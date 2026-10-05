"""Raport HTML pentru un lot.

Un singur fisier autonom: fara CDN, fara resurse externe. Se deschide cu
dublu-clic si merge si daca il muti pe alt calculator.

Decizia care conteaza: pentru randurile cu probleme, **decupajele portativelor
sunt incorporate in pagina**. Cand o transcriere iese prost, prima intrebare nu
e "ce a citit modelul", ci "ce i-am dat sa citeasca" — daca detectia de
portative a taiat gresit, restul nu mai conteaza. Raportul trebuie sa raspunda
la intrebarea asta fara sa cauti fisiere prin foldere.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from app.batch.store import Job, State

MAX_EMBEDDED_CROPS = 40  # peste atat, pagina devine greu de deschis

STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: .25rem; }
.sub { color: #6b7280; margin-top: 0; }
.cards { display: flex; gap: .75rem; flex-wrap: wrap; margin: 1.5rem 0; }
.card { border: 1px solid #d1d5db; border-radius: .5rem; padding: .75rem 1rem; min-width: 7rem; }
.card b { display: block; font-size: 1.6rem; line-height: 1.2; }
.card span { color: #6b7280; font-size: .85rem; }
.ok b { color: #15803d; } .bad b { color: #b91c1c; } .warn b { color: #b45309; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #e5e7eb; }
th { cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { background: rgba(127,127,127,.12); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.failed td:first-child, tr.haserr td:first-child { border-left: 3px solid #b91c1c; }
details { margin: .3rem 0; }
summary { cursor: pointer; }
.msg { color: #b91c1c; font-size: .85rem; }
.crops img { max-width: 100%; border: 1px solid #d1d5db; margin: .25rem 0; display: block; }
@media (prefers-color-scheme: dark) {
  th, td { border-color: #374151; } .card { border-color: #374151; }
  .sub, .card span { color: #9ca3af; }
}
"""

SCRIPT = """
document.querySelectorAll('th').forEach((th, column) => {
  th.addEventListener('click', () => {
    const body = th.closest('table').tBodies[0];
    const rows = [...body.rows];
    const asc = th.dataset.asc !== 'true';
    th.dataset.asc = asc;
    rows.sort((a, b) => {
      const x = a.cells[column].dataset.sort ?? a.cells[column].textContent;
      const y = b.cells[column].dataset.sort ?? b.cells[column].textContent;
      const n = parseFloat(x) - parseFloat(y);
      return (asc ? 1 : -1) * (isNaN(n) ? x.localeCompare(y) : n);
    });
    rows.forEach(row => body.appendChild(row));
  });
});
"""


def _embed(path: Path) -> str | None:
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{data}"


def _cell(value, numeric: bool = False) -> str:
    if value is None:
        return '<td class="num">—</td>' if numeric else "<td>—</td>"
    text = html.escape(str(value))
    return f'<td class="num" data-sort="{text}">{text}</td>' if numeric else f"<td>{text}</td>"


def _detail_block(job: Job, budget: list[int]) -> str:
    """Sectiunea pliata a unui rand: probleme, erori, si decupajele."""
    if not job.detail:
        return ""
    try:
        detail = json.loads(job.detail)
    except json.JSONDecodeError:
        return ""

    parts: list[str] = []
    for label, items in (("Probleme", detail.get("problems")),
                         ("Erori de validare", detail.get("errors")),
                         ("Avertismente", detail.get("warnings"))):
        if not items:
            continue
        listed = "".join(f"<li>{html.escape(str(i))}</li>" for i in items[:20])
        extra = f"<li>… si inca {len(items) - 20}</li>" if len(items) > 20 else ""
        parts.append(f"<b>{label}</b><ul>{listed}{extra}</ul>")

    # Decupajele se incorporeaza doar unde chiar ajuta: randuri cu probleme.
    troubled = job.state is State.FAILED or (job.errors or 0) > 0 or bool(detail.get("problems"))
    if troubled and budget[0] > 0:
        images = []
        for crop in detail.get("crops", []):
            if budget[0] <= 0:
                break
            source = _embed(Path(crop))
            if source:
                images.append(f'<img src="{source}" alt="portativ">')
                budget[0] -= 1
        if images:
            parts.append(f'<div class="crops"><b>Portative decupate</b>{"".join(images)}</div>')

    if not parts:
        return ""
    return f'<details><summary>detalii</summary>{"".join(parts)}</details>'


def render(jobs: list[Job], title: str = "Raport de conversie") -> str:
    done = [j for j in jobs if j.state is State.DONE]
    failed = [j for j in jobs if j.state is State.FAILED]
    clean = [j for j in done if not j.errors and not j.message]
    seconds = sum(j.seconds or 0 for j in jobs)

    budget = [MAX_EMBEDDED_CROPS]
    rows = []
    for job in jobs:
        classes = []
        if job.state is State.FAILED:
            classes.append("failed")
        elif job.errors:
            classes.append("haserr")

        message = f'<div class="msg">{html.escape(job.message)}</div>' if job.message else ""
        # Doar numele fisierului, nu calea absoluta: raportul sta in acelasi
        # folder cu MusicXML-urile, iar o cale Windows cu backslash-uri si
        # spatii nu e o adresa valida — linkul nu ar deschide nimic.
        link = (f' · <a href="{html.escape(Path(job.musicxml).name)}">MusicXML</a>'
                if job.musicxml else "")
        first = (f'<td><b>{html.escape(job.label)}</b>{message}'
                 f'{_detail_block(job, budget)}{link}</td>')

        rows.append(
            f'<tr class="{" ".join(classes)}">{first}'
            + _cell(job.kind)
            + _cell(job.state.value)
            + _cell(job.staves, numeric=True)
            + _cell(job.systems, numeric=True)
            + _cell(job.errors, numeric=True)
            + _cell(job.warnings, numeric=True)
            + _cell(f"{job.seconds:.1f}" if job.seconds else None, numeric=True)
            + "</tr>"
        )

    headers = ("Fisier", "Tip", "Stare", "Portative", "Sisteme", "Erori", "Avertismente", "Secunde")
    return f"""<!doctype html>
<html lang="ro"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(jobs)} unitati de lucru · {seconds / 60:.1f} minute de procesare</p>
<div class="cards">
  <div class="card ok"><b>{len(clean)}</b><span>fara probleme</span></div>
  <div class="card warn"><b>{len(done) - len(clean)}</b><span>cu observatii</span></div>
  <div class="card bad"><b>{len(failed)}</b><span>esuate</span></div>
  <div class="card"><b>{seconds / max(len(jobs), 1):.1f}s</b><span>media pe pagina</span></div>
</div>
<div class="scroll"><table><thead><tr>
{"".join(f"<th>{h}</th>" for h in headers)}
</tr></thead><tbody>
{"".join(rows)}
</tbody></table></div>
<script>{SCRIPT}</script>
</body></html>
"""


def write(jobs: list[Job], path: Path, title: str = "Raport de conversie") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(jobs, title), encoding="utf-8")
    return path
