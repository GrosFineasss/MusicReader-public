"""Fereastra de corectare: transcrierea gresita devine transcriere buna.

Etapa 12. Deasupra lui `app/correction/editare.py`, care tine toata logica —
aici e doar aratatul si apasatul.

Ce **nu** e: un editor de partituri. Nu se deseneaza portative si nu se trag note
cu mausul. E un tabel: fiecare rand o nota, cu inaltimea si durata de schimbat, si
poza paginii deasupra ca sa ai la ce te uita. Un editor grafic ar fi luat
saptamani si ar fi facut mai prost ce face deja MuseScore; ce nu face MuseScore e
**sa lege corectia de imaginea din care a iesit**, si aia e tot rostul ferestrei.

Ce iese: MusicXML corectat. Din el, `scripts/dataset.py` face perechi de
antrenare pe drumul care exista deja — nu se scrie un al doilea.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.correction.editare import (
    Adresa,
    Istoric,
    catre_nota,
    catre_pauza,
    evenimente,
    eveniment_la,
    probleme,
    schimba_durata,
    schimba_inaltimea,
    sterge,
)
from app.music_model.model import Duration, Note, NoteType, Pitch, Rest

TREPTE = ["C", "D", "E", "F", "G", "A", "B"]
ALTERATII = {"": 0, "#": 1, "b": -1}

COLOANE = ["voce", "masura", "poz.", "ce e", "inaltime", "durata", "puncte"]


def numele_inaltimii(pitch: Pitch) -> str:
    """`C#4` — cum se scrie o inaltime intr-un tabel, pe scurt."""
    semn = {1: "#", -1: "b", 0: ""}.get(pitch.alter, f"({pitch.alter:+d})")
    return f"{pitch.step}{semn}{pitch.octave}"


def citeste_inaltimea(text: str) -> Pitch:
    """Inversul lui `numele_inaltimii`. Ridica `ValueError` pe orice altceva.

    Mesajele spun **ce** n-a mers, nu doar ca n-a mers: cine corecteaza o suta de
    note pe pagina nu are chef sa ghiceasca formatul.
    """
    text = text.strip()
    if len(text) < 2:
        raise ValueError(f"nu inteleg inaltimea {text!r}; scrie de exemplu C4 sau F#3")
    step = text[0].upper()
    if step not in TREPTE:
        raise ValueError(f"treapta {step!r} nu exista; foloseste una din {''.join(TREPTE)}")
    semn, octava = ("", text[1:]) if text[1].isdigit() else (text[1], text[2:])
    if semn not in ALTERATII:
        raise ValueError(f"alteratia {semn!r} nu exista; foloseste # sau b")
    if not octava.isdigit():
        raise ValueError(f"octava {octava!r} nu e un numar")
    return Pitch(step=step, octave=int(octava), alter=ALTERATII[semn])


def citeste_durata(tip: str, puncte: str) -> Duration:
    """Durata dintr-un rand de tabel, cu aceleasi mesaje lamurite."""
    tip, puncte = tip.strip(), puncte.strip()
    try:
        note_type = NoteType(tip)
    except ValueError:
        stiute = ", ".join(t.value for t in NoteType)
        raise ValueError(f"durata {tip!r} nu exista; scrie una din: {stiute}") from None
    if not puncte.isdigit() or int(puncte) > 2:
        raise ValueError(f"punctele {puncte!r} trebuie sa fie 0, 1 sau 2")
    return Duration(note_type, dots=int(puncte))


def salveaza_corectat(score, musicxml: Path) -> Path:
    """Partitura corectata, langa cea originala, cu `-corectat` in nume.

    Nu se suprascrie iesirea modelului: diferenta dintre ce a citit si ce era de
    fapt e chiar informatia din care se invata. Cine sterge originalul pierde
    exact masura greselii.
    """
    # `export.write`, adica exact drumul pe care lotul scrie transcrierile
    # (`app/batch/runner.py`). Un al doilea drum de scriere ar putea produce alt
    # fisier decat cel pe care il citeste restul conductei.
    from app.musicxml import export

    return export.write(score, musicxml.with_name(f"{musicxml.stem}-corectat.musicxml"))


class FereastraCorectare(QDialog):
    """Poza sus, notele jos, ce nu se inchide alaturi."""

    def __init__(self, imagine: Path, musicxml: Path, parent=None) -> None:
        super().__init__(parent)
        self.imagine = Path(imagine)
        self.musicxml = Path(musicxml)
        self.setWindowTitle(f"Corectare — {self.imagine.name}")
        self.resize(1100, 800)

        from app.evaluation.reference import load

        self.istoric = Istoric(load(self.musicxml))
        self.adrese: list[Adresa] = []
        self._construieste()
        self._reincarca()

    # --------------------------------------------------------------- constructie

    def _construieste(self) -> None:
        layout = QVBoxLayout(self)
        despicator = QSplitter(Qt.Vertical)

        poza = QLabel()
        pixmap = QPixmap(str(self.imagine))
        if not pixmap.isNull():
            poza.setPixmap(pixmap.scaledToWidth(1000, Qt.SmoothTransformation))
        else:
            poza.setText(f"(nu s-a putut deschide {self.imagine.name})")
        poza.setAlignment(Qt.AlignCenter)
        derulare = QScrollArea()
        derulare.setWidget(poza)
        derulare.setWidgetResizable(True)
        despicator.addWidget(derulare)

        jos = QWidget()
        jos_layout = QHBoxLayout(jos)

        self.tabel = QTableWidget(0, len(COLOANE))
        self.tabel.setHorizontalHeaderLabels(COLOANE)
        self.tabel.itemChanged.connect(self._pe_schimbare)
        jos_layout.addWidget(self.tabel, 3)

        laterala = QVBoxLayout()
        laterala.addWidget(QLabel("Ce nu se inchide:"))
        self.lista_probleme = QListWidget()
        laterala.addWidget(self.lista_probleme)
        cutie = QWidget()
        cutie.setLayout(laterala)
        jos_layout.addWidget(cutie, 2)

        despicator.addWidget(jos)
        despicator.setSizes([300, 500])
        layout.addWidget(despicator, 1)

        butoane = QHBoxLayout()
        self.buton_inapoi = QPushButton("Inapoi")
        self.buton_inapoi.clicked.connect(self._inapoi)
        self.buton_inainte = QPushButton("Inainte")
        self.buton_inainte.clicked.connect(self._inainte)
        self.buton_pauza = QPushButton("Nota / pauza")
        self.buton_pauza.clicked.connect(self._comuta_pauza)
        self.buton_sterge = QPushButton("Sterge")
        self.buton_sterge.clicked.connect(self._sterge)
        for b in (self.buton_inapoi, self.buton_inainte, self.buton_pauza,
                  self.buton_sterge):
            butoane.addWidget(b)
        butoane.addStretch(1)
        self.eticheta_stare = QLabel()
        butoane.addWidget(self.eticheta_stare)
        self.buton_salveaza = QPushButton("Salveaza corectat")
        self.buton_salveaza.clicked.connect(self._salveaza)
        butoane.addWidget(self.buton_salveaza)
        layout.addLayout(butoane)

    # ------------------------------------------------------------------- afisare

    def _reincarca(self) -> None:
        """Tabelul si lista de constatari, din partitura curenta."""
        score = self.istoric.curenta
        gasite = probleme(score)

        # Umplerea programata nu e o modificare a omului; fara asta, fiecare
        # celula scrisa aici ar declansa `_pe_schimbare` si ar reintra la
        # nesfarsit.
        self.tabel.blockSignals(True)
        self.adrese = []
        perechi = evenimente(score)
        self.tabel.setRowCount(len(perechi))
        for rand, (adresa, event) in enumerate(perechi):
            self.adrese.append(adresa)
            este_nota = isinstance(event, Note)
            valori = [
                adresa.part_id,
                str(adresa.measure),
                str(adresa.index),
                "nota" if este_nota else "pauza",
                numele_inaltimii(event.pitch) if este_nota else "",
                event.duration.note_type.value,
                str(event.duration.dots),
            ]
            for coloana, text in enumerate(valori):
                item = QTableWidgetItem(text)
                if coloana < 4:  # unde sta nota nu se muta din tabel
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if adresa in gasite:
                    item.setBackground(Qt.yellow)
                self.tabel.setItem(rand, coloana, item)
        self.tabel.blockSignals(False)

        self.lista_probleme.clear()
        for lista in gasite.values():
            for finding in lista:
                self.lista_probleme.addItem(str(finding))

        constatari = sum(len(v) for v in gasite.values())
        self.eticheta_stare.setText(
            f"{self.istoric.modificari} corectii · {constatari} constatari"
        )
        self.buton_inapoi.setEnabled(self.istoric.se_poate_inapoi)
        self.buton_inainte.setEnabled(self.istoric.se_poate_inainte)

    # ---------------------------------------------------------------- modificari

    def _adresa_selectata(self) -> Adresa | None:
        rand = self.tabel.currentRow()
        return self.adrese[rand] if 0 <= rand < len(self.adrese) else None

    def _pe_schimbare(self, item: QTableWidgetItem) -> None:
        """O celula editata de om. Ce nu se intelege se refuza si se spune."""
        adresa = self.adrese[item.row()]
        coloana = COLOANE[item.column()]
        try:
            if coloana == "inaltime":
                self.istoric.aplica(schimba_inaltimea(
                    self.istoric.curenta, adresa, citeste_inaltimea(item.text())))
            elif coloana in ("durata", "puncte"):
                durata = citeste_durata(
                    self.tabel.item(item.row(), COLOANE.index("durata")).text(),
                    self.tabel.item(item.row(), COLOANE.index("puncte")).text(),
                )
                self.istoric.aplica(schimba_durata(self.istoric.curenta, adresa, durata))
        except Exception as eroare:
            QMessageBox.warning(self, "Nu am inteles", str(eroare))
        self._reincarca()

    def _comuta_pauza(self) -> None:
        adresa = self._adresa_selectata()
        if adresa is None:
            return
        event = eveniment_la(self.istoric.curenta, adresa)
        if isinstance(event, Rest):
            self.istoric.aplica(catre_nota(
                self.istoric.curenta, adresa, Pitch(step="C", octave=4)))
        else:
            self.istoric.aplica(catre_pauza(self.istoric.curenta, adresa))
        self._reincarca()

    def _sterge(self) -> None:
        adresa = self._adresa_selectata()
        if adresa is not None:
            self.istoric.aplica(sterge(self.istoric.curenta, adresa))
            self._reincarca()

    def _inapoi(self) -> None:
        self.istoric.inapoi()
        self._reincarca()

    def _inainte(self) -> None:
        self.istoric.inainte()
        self._reincarca()

    # ------------------------------------------------------------------ salvarea

    def _salveaza(self) -> None:
        try:
            cale = salveaza_corectat(self.istoric.curenta, self.musicxml)
        except Exception as eroare:
            QMessageBox.critical(self, "Nu s-a putut salva", str(eroare))
            return
        QMessageBox.information(
            self, "Salvat",
            f"Scris in {cale.name}.\n\n"
            "Ca sa devina material de antrenare:\n"
            f"  python scripts/dataset.py --imagine {self.imagine.name} "
            f"--ref {cale.name} --out data/antrenare",
        )
        self.accept()
