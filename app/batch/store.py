"""Starea unui lot de conversie, persistata in SQLite.

De ce o baza de date si nu un JSON: la 500 de intrari vrem sa scriem dupa
fiecare fisier (ca o intrerupere sa nu piarda tot), sa interogam ("arata-mi
esecurile"), si sa nu ramanem cu un fisier trunchiat daca procesul e omorat in
mijlocul unei scrieri. `sqlite3` e in biblioteca standard, deci nu adauga nicio
dependenta.

Reluarea e implicita, nu o comanda separata: la deschidere, tot ce era `running`
se marcheaza inapoi `pending`, fiindca un job "in curs" intr-o baza inchisa
inseamna un proces care a murit.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    path        TEXT NOT NULL,
    page        INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL,
    digest      TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',
    systems     INTEGER,
    staves      INTEGER,
    errors      INTEGER,
    warnings    INTEGER,
    seconds     REAL,
    message     TEXT,
    musicxml    TEXT,
    detail      TEXT,
    PRIMARY KEY (path, page)
);
"""


class State(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    path: Path
    page: int
    kind: str
    state: State
    systems: int | None = None
    staves: int | None = None
    errors: int | None = None
    warnings: int | None = None
    seconds: float | None = None
    message: str | None = None
    musicxml: str | None = None
    detail: str | None = None

    @property
    def label(self) -> str:
        return self.path.name if self.page == 0 else f"{self.path.name}#{self.page + 1}"


def digest_of(path: Path) -> str:
    """Amprenta fisierului, ca sa observam daca a fost inlocuit intre rulari.

    Citim in bucati: unele scanuri sunt de zeci de megaocteti si nu are rost sa
    le tinem in memorie doar ca sa le numaram.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._recover()

    def _recover(self) -> None:
        """Un job `running` intr-o baza proaspat deschisa = proces mort."""
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET state = ?, message = ? WHERE state = ?",
                (State.PENDING.value, "reluat dupa o intrerupere", State.RUNNING.value),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- adaugare -------------------------------------------------------

    def add(self, path: Path, page: int, kind: str, digest: str | None = None) -> None:
        """Inregistreaza o unitate de lucru.

        Un fisier deja terminat se re-pune in coada doar daca s-a schimbat pe
        disc; altfel relansarea lotului l-ar reface degeaba.

        `digest` poate fi dat de apelant ca sa nu recitim acelasi fisier o data
        pentru fiecare pagina a lui: un PDF de 200 de pagini ar fi citit integral
        de 200 de ori doar ca sa intre in coada.
        """
        current = digest if digest is not None else digest_of(path)
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(
                "SELECT digest, kind FROM jobs WHERE path = ? AND page = ?",
                (str(path), page),
            )
            row = cursor.fetchone()

        with self.connection:
            if row is None:
                self.connection.execute(
                    "INSERT INTO jobs (path, page, kind, digest) VALUES (?, ?, ?, ?)",
                    (str(path), page, kind, current),
                )
            elif row["digest"] != current:
                self.connection.execute(
                    "UPDATE jobs SET digest = ?, kind = ?, state = ?, message = ? "
                    "WHERE path = ? AND page = ?",
                    (current, kind, State.PENDING.value, "fisierul s-a schimbat",
                     str(path), page),
                )
            elif row["kind"] != kind:
                # Acelasi fisier, re-adaugat cu alt tip. Tipul decide pe ce
                # conducta merge, deci trebuie sa se aplice — altfel randul ar
                # ramane etichetat gresit si ar fi transcris cu motorul gresit.
                self.connection.execute(
                    "UPDATE jobs SET kind = ?, state = ?, message = ? "
                    "WHERE path = ? AND page = ?",
                    (kind, State.PENDING.value, "tipul partiturii s-a schimbat",
                     str(path), page),
                )

    def add_failed(self, path: Path, page: int, kind: str, message: str) -> None:
        """Inregistreaza un fisier care nu a putut fi nici macar citit.

        Nu-i calculam amprenta — tocmai citirea lui a esuat. Fara amprenta,
        randul se va reincerca la fiecare rulare, ceea ce e comportamentul dorit:
        daca inlocuiesti scanul stricat, urmatorul lot il ia.
        """
        with self.connection:
            self.connection.execute(
                "INSERT INTO jobs (path, page, kind, digest, state, message) "
                "VALUES (?, ?, ?, '', ?, ?) "
                "ON CONFLICT(path, page) DO UPDATE SET state = ?, message = ?",
                (str(path), page, kind, State.FAILED.value, message,
                 State.FAILED.value, message),
            )

    # -- consum ---------------------------------------------------------

    def pending(self) -> list[Job]:
        return self._select("WHERE state IN (?, ?)", (State.PENDING.value, State.FAILED.value))

    def all(self) -> list[Job]:
        return self._select("", ())

    def _select(self, where: str, params: tuple) -> list[Job]:
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(f"SELECT * FROM jobs {where} ORDER BY path, page", params)
            return [self._to_job(row) for row in cursor.fetchall()]

    @staticmethod
    def _to_job(row: sqlite3.Row) -> Job:
        return Job(
            path=Path(row["path"]),
            page=row["page"],
            kind=row["kind"],
            state=State(row["state"]),
            systems=row["systems"],
            staves=row["staves"],
            errors=row["errors"],
            warnings=row["warnings"],
            seconds=row["seconds"],
            message=row["message"],
            musicxml=row["musicxml"],
            detail=row["detail"],
        )

    # -- actualizare ----------------------------------------------------

    def mark_running(self, path: Path, page: int) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET state = ?, message = NULL WHERE path = ? AND page = ?",
                (State.RUNNING.value, str(path), page),
            )

    def mark_done(self, path: Path, page: int, **fields) -> None:
        self._finish(path, page, State.DONE, **fields)

    def mark_failed(self, path: Path, page: int, message: str) -> None:
        self._finish(path, page, State.FAILED, message=message)

    def _finish(self, path: Path, page: int, state: State, **fields) -> None:
        columns = ", ".join(f"{name} = ?" for name in fields)
        assignment = f", {columns}" if columns else ""
        with self.connection:
            self.connection.execute(
                f"UPDATE jobs SET state = ?{assignment} WHERE path = ? AND page = ?",
                (state.value, *fields.values(), str(path), page),
            )

    # -- rezumat --------------------------------------------------------

    def counts(self) -> dict[str, int]:
        with closing(self.connection.cursor()) as cursor:
            cursor.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
            return {row["state"]: row["n"] for row in cursor.fetchall()}
