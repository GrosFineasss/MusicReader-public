"""Modelul de sinteza care deseneaza cu glifele **noastre**, nu cu MUSCIMA++.

Aici iese MUSCIMA++ din proiect (PROJECT.md §3.6, §15.16). `BaseHandwrittenModel`
leaga implicit sintetizatoare care descarca active NonCommercial; noi rescriem
trei legaturi si ramanem cu:

| ce | implicit la ei | la noi | cere descarcare |
|---|---|---|---|
| glife | `MuscimaPPGlyphSynthesizer` | **al nostru**, din foile scrise de oameni | nu |
| linii | `MuscimaPPLineSynthesizer` | `NaiveLineSynthesizer` (drepte) | nu |
| hartie | `MzkQuiltingPaperSynthesizer` | `SolidColorPaperSynthesizer` | nu |
| portative | `NaiveStafflinesSynthesizer` | la fel | nu |

Verificat: doar sintetizatoarele MUSCIMA++ si OmniOMR ating `AssetRepository`.
Cu legaturile astea, randarea merge **fara nicio descarcare** — ceea ce e si
proba ca datele produse sunt integral ale noastre.

Ca `PaperSynthesizer` sa fie hartie alba nu e o economie, e o alegere: pana nu
avem texturi proprii, o hartie imprumutata ar readuce exact problema de licenta
pe care o rezolvam.
"""

# **Fara `from __future__ import annotations` aici, dinadins.** Containerul de
# dependinte al lui Smashcima (`punq`) rezolva parametrii dupa adnotarile de
# tip, iar PEP 563 le-ar transforma in siruri pe care nu le-ar mai putea lega
# de clasele construite in functiile de mai jos.
import random
from pathlib import Path
from typing import Union

# Ce cere Smashcima, si ce am strans noi pe foaie. Fara harta asta, o eticheta
# pe care n-am colectat-o ar opri randarea; cu ea, se deseneaza cu ruda cea mai
# apropiata. Perechile sunt alese vizual: un steag de saisprezecime seamana cu
# unul de optime mai mult decat cu orice altceva de pe foaie.
#
# Cheile sunt **fara prefix**: Smashcima cere `smufl::gClef`, iar noi pastram
# `gClef`. Prefixul se taie in `normalize_label`, si nu invers, fiindca eticheta
# scurta e cea tiparita pe foaia pe care o citeste omul — `smufl::gClef` scris
# langa o casuta n-ar ajuta pe nimeni.
LABEL_MAP = {
    "gClefSmall": "gClef",
    "fClefSmall": "fClef",
    "cClef": "gClef",
    "cClefSmall": "gClef",
    "flag16thUp": "flag8thUp",
    "flag16thDown": "flag8thDown",
    "flag32ndUp": "flag8thUp",
    "flag32ndDown": "flag8thDown",
    "rest16th": "rest8th",
    "rest32nd": "rest8th",
    "noteheadDoubleWhole": "noteheadWhole",
}


def normalize_label(label: str) -> str:
    """Eticheta lui Smashcima -> eticheta din repository-ul nostru.

    Taie prefixul de spatiu de nume (`smufl::`, `smashcima::`) si aplica harta
    de inlocuiri. Ordinea conteaza: harta e scrisa cu nume scurte.
    """
    scurt = label.split("::")[-1]
    return LABEL_MAP.get(scurt, scurt)


def load_repository(path: Union[str, Path]):
    """Repository-ul de glife scris de `scripts/glife_repository.py`."""
    import pickle

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nu exista repository-ul de glife: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_style_domain_class():
    """Domeniul de stiluri: multimea mainilor din repository-ul nostru.

    Se construieste in functie, nu la import, fiindca mostenirea cere Smashcima
    incarcat — iar `.venv`-ul proiectului nu-l are (§15.14) si nici nu trebuie.
    """
    from smashcima.assets.glyphs.mung.repository.MungSymbolRepository import (
        MungSymbolRepository,
    )
    from smashcima.synthesis.style.RepositoryStyleDomain import RepositoryStyleDomain

    class GlifeStyleDomain(RepositoryStyleDomain):
        """Mainile care au completat foi. Una se alege pentru fiecare pagina."""

        def __init__(
            self, symbol_repository: MungSymbolRepository, rng: random.Random
        ):
            super().__init__(symbol_repository=symbol_repository, rng=rng)

    return GlifeStyleDomain


def build_synthesizer_class(style_domain_class):
    from smashcima.assets.glyphs.mung.repository.MungSymbolRepository import (
        MungSymbolRepository,
    )
    from smashcima.synthesis.glyph.RepositoryGlyphSynthesizer import (
        RepositoryGlyphSynthesizer,
    )

    class GlifeGlyphSynthesizer(RepositoryGlyphSynthesizer):
        """Alege glife din foile scrise de oameni, dupa eticheta si stil."""

        def __init__(
            self,
            symbol_repository: MungSymbolRepository,
            style_domain: style_domain_class,
            rng: random.Random,
        ):
            super().__init__(
                symbol_repository=symbol_repository,
                style_domain=style_domain,
                rng=rng,
            )

        def supports_label(self, label: str) -> bool:
            return super().supports_label(normalize_label(label))

        def create_glyph(self, label: str):
            glyph = super().create_glyph(normalize_label(label))
            # Eticheta ceruta, nu cea desenata: harta nu e unu-la-unu, iar
            # adnotarea trebuie sa spuna ce reprezinta simbolul in muzica.
            glyph.label = label
            return glyph

    return GlifeGlyphSynthesizer


def build_repository_line_synthesizer_class(style_domain_class, fallback_class):
    """Liniile desenate cu glifele **noastre**, cu revenire la cele drepte.

    `beam`, `legerLine`, `stem` si `beamHook` se intind intre doua capete, deci
    Smashcima le cere din `line_glyphs_index`, nu din cel obisnuit. Capetele le
    detecteaza `app/training/repository.py`.

    Rezerva nu e o precautie de forma: daca o mana n-a scris o bara de grup, sau
    trasatura ei a iesit prea scurta ca sa i se gaseasca doua capete, glifa nu
    intra in repository — iar fara rezerva randarea ar cadea in mijlocul unei
    pagini. Asa iese o linie dreapta si pagina se termina.
    """
    from smashcima.assets.glyphs.mung.repository.MungSymbolRepository import (
        MungSymbolRepository,
    )
    from smashcima.synthesis.glyph.RepositoryLineSynthesizer import (
        RepositoryLineSynthesizer,
    )

    class GlifeRepositoryLineSynthesizer(RepositoryLineSynthesizer):
        def __init__(
            self,
            symbol_repository: MungSymbolRepository,
            style_domain: style_domain_class,
            rng: random.Random,
        ):
            super().__init__(
                symbol_repository=symbol_repository,
                style_domain=style_domain,
                rng=rng,
            )
            self.fallback_synthesizer = fallback_class()

        def supports_label(self, label: str) -> bool:
            return super().supports_label(normalize_label(label))

        def create_glyph(self, label: str, delta):
            glyph = super().create_glyph(normalize_label(label), delta)
            glyph.label = label
            return glyph

    return GlifeRepositoryLineSynthesizer


def build_line_synthesizer_class():
    """Liniile drepte — rezerva, si ce se foloseste cand n-avem glife de linie.

    `NaiveLineSynthesizer` deseneaza o linie dreapta pentru orice, dar e
    **abstracta**: nu implementeaza `supports_label`, fiindca autorii au
    gandit-o ca baza pentru sintetizatoare proprii, nu ca implementare de-a
    gata. Noi spunem „le accept pe toate", fiindca o linie dreapta se poate
    desena pentru orice eticheta de linie — bara de grup, linie suplimentara,
    legato.

    E o solutie de asteptare, si se vede pe pagina: notele ies scrise de mana,
    liniile ies drepte. Se inlocuieste cand `line_glyphs_index` se umple cu
    glife adevarate (golul din §15.16).
    """
    from smashcima.synthesis.glyph.NaiveLineSynthesizer import NaiveLineSynthesizer

    class GlifeLineSynthesizer(NaiveLineSynthesizer):
        def supports_label(self, label: str) -> bool:
            return True

    return GlifeLineSynthesizer


class _DomeniuImprumutat:
    """Ce cere `BaseHandwrittenModel.__call__` ca sa stampileze metadate.

    Modelul lor scrie in scena `mpp_writer` si `mzk_background_patch`, luate din
    doua domenii de stil MUSCIMA++/MZK pe care noi nu le avem si nu le vrem.
    Sunt doar metadate — nu influenteaza desenul — dar lipsa lor ar da
    `AttributeError` in mijlocul randarii. Deci le tinem locul cu stilul nostru.
    """

    def __init__(self, style_domain=None):
        self._domain = style_domain

    @property
    def current_writer(self):
        return getattr(self._domain, "current_style", None)

    @property
    def current_patch(self):
        return None


def build_paper_synthesizer_class(dpi: float):
    """Hartie alba, randata la **aceeasi** rezolutie ca scena.

    `SolidColorPaperSynthesizer` isi fixeaza `dpi = 72` in constructor. Scena se
    randeaza la 300, deci sprite-ul de hartie se intinde de patru ori — iar
    marginea lui lasa o **dunga intunecata pe toata inaltimea paginii**, la
    ~4 px de marginea din dreapta.

    Masurat: la 72 dpi apare o coloana aproape plina la 4 px de margine; la 300
    nu mai apare niciuna. Nu e cosmetic — o dunga verticala in fiecare decupaj
    de sistem ar fi cerneala pe care modelul ar invata-o ca facand parte din
    muzica.
    """
    from smashcima.synthesis import SolidColorPaperSynthesizer

    class GlifePaperSynthesizer(SolidColorPaperSynthesizer):
        def __init__(self):
            super().__init__()
            self.dpi = dpi

    return GlifePaperSynthesizer


def build_model(repository, seed: int = 0, dpi: float = 300.0):
    """Un `BaseHandwrittenModel` legat la glifele noastre, fara MUSCIMA++.

    **Nu se cheama `super().register_services()`**, si asta e partea delicata.
    Inregistrarea lor pune in container `MuscimaPPStyleDomain` si
    `MzkPaperStyleDomain`, iar apoi doua lucruri le rezolva **avid**, ceea ce
    porneste descarcarea a 2 GB de date NonCommercial:

    * `Model.resolve_services` cheama `styler.register_domains_from_container()`,
      care rezolva **orice** `StyleDomain` inregistrat;
    * `BaseHandwrittenModel.resolve_services` le cere pe amandoua pe nume.

    Deci nu e destul sa rescriem legaturile de sintetizatoare — tipurile alea
    nu trebuie sa ajunga in container. Repetam aici lista lor, fara ele.

    Pretul e ca lista trebuie tinuta la zi daca Smashcima si-o schimba pe a lui.
    Alternativa ar fi fost sa lasam descarcarea sa se intample si sa aruncam
    rezultatul, ceea ce ar fi readus exact problema de licenta din §3.6.
    """
    from smashcima.assets.glyphs.mung.repository.MungSymbolRepository import (
        MungSymbolRepository,
    )
    from smashcima.orchestration import BaseHandwrittenModel, Model
    from smashcima.synthesis import (
        BeamStemSynthesizer,
        ColumnMusicNotationSynthesizer,
        GlyphSynthesizer,
        LineSynthesizer,
        NaiveStafflinesSynthesizer,
        PaperSynthesizer,
        SimplePageSynthesizer,
        StafflinesSynthesizer,
    )
    from smashcima.synthesis.MusicNotationSynthesizer import MusicNotationSynthesizer
    from smashcima.synthesis.PageSynthesizer import PageSynthesizer

    style_domain_class = build_style_domain_class()
    synthesizer_class = build_synthesizer_class(style_domain_class)
    naive_line_class = build_line_synthesizer_class()
    paper_synthesizer_class = build_paper_synthesizer_class(dpi)
    # Cu glife de linie in repository se deseneaza cu ele; fara, raman drepte.
    line_synthesizer_class = (
        build_repository_line_synthesizer_class(style_domain_class, naive_line_class)
        if repository.get_all_line_glyph_labels()
        else naive_line_class
    )

    class ModelGlife(BaseHandwrittenModel):
        def register_services(self):
            # Bunicul, nu parintele: vezi docstring-ul lui `build_model`.
            Model.register_services(self)
            c = self.container

            c.instance(MungSymbolRepository, repository)
            c.instance(random.Random, random.Random(seed))

            c.interface(
                StafflinesSynthesizer, NaiveStafflinesSynthesizer, register_impl=True
            )
            c.interface(GlyphSynthesizer, synthesizer_class, register_impl=True)
            c.interface(LineSynthesizer, line_synthesizer_class, register_impl=True)

            c.type(style_domain_class)
            c.type(ColumnMusicNotationSynthesizer)
            c.type(BeamStemSynthesizer)
            c.type(SimplePageSynthesizer)
            c.type(paper_synthesizer_class)

            c.interface(MusicNotationSynthesizer, ColumnMusicNotationSynthesizer)
            c.interface(PageSynthesizer, SimplePageSynthesizer)
            c.interface(PaperSynthesizer, paper_synthesizer_class)

        def resolve_services(self):
            Model.resolve_services(self)
            c = self.container
            self.notation_synthesizer = c.resolve(MusicNotationSynthesizer)
            self.page_synthesizer = c.resolve(PageSynthesizer)
            self.glife_style_domain = c.resolve(style_domain_class)
            self.mpp_style_domain = _DomeniuImprumutat(self.glife_style_domain)
            self.mzk_paper_style_domain = _DomeniuImprumutat()

    return ModelGlife()


def render_system(
    model,
    musicxml: Union[str, Path],
    staves: int,
    dpi: float = 300.0,
    margin: float = 0.8,
):
    """Randeaza o felie si o **decupeaza pe portativele ei**, ca la inferenta.

    Modelul primeste la inferenta decupaje de la `GrandstaffEngine.detect_systems`;
    sinteticul trebuie sa arate la fel, altfel antrenam pe altceva decat citim.

    Portativele nu se **detecteaza** — la sintetic stim unde sunt. Se iau primele
    `staves` din `page.staves` (sintetizatorul de pagina le umple de sus in jos),
    iar dreptunghiul lor din `glyph.get_bbox_in_space`. Asa nu bagam eroarea
    detectorului in datele de antrenare, ceea ce ar fi invatat modelul sa
    citeasca decupaje pe care nu le vede niciodata.

    `margin` e in **inaltimi de portativ**, nu in pixeli, si asta conteaza:
    dreptunghiul unui portativ e al celor cinci linii, iar notele cu linii
    suplimentare si coditele ies mult peste ele. O margine fixa in pixeli ar
    taia altfel la fiecare rezolutie; una masurata in portative taie la fel
    peste tot. E acelasi rationament ca la `music_zones` din preprocesare, unde
    marginea se da in spatii de portativ.
    """
    import numpy as np
    from PIL import Image

    scene = model(file=str(musicxml))
    scene.dpi = dpi
    page = scene.pages[0]
    grey = _to_grey(scene.render(page))

    if staves > len(page.staves):
        raise ValueError(
            f"felia cere {staves} portative, pagina are {len(page.staves)}"
        )

    space = page.view_box.space
    boxes = [s.glyph.get_bbox_in_space(space) for s in page.staves[:staves]]
    view = page.view_box.rectangle
    scale = dpi / 25.4  # mm -> pixeli
    pad = margin * max(s.staff_height for s in page.staves[:staves])

    left = (min(b.x for b in boxes) - pad - view.x) * scale
    right = (max(b.x + b.width for b in boxes) + pad - view.x) * scale
    top = (min(b.y for b in boxes) - pad - view.y) * scale
    bottom = (max(b.y + b.height for b in boxes) + pad - view.y) * scale

    height, width = grey.shape
    box = (
        max(int(round(left)), 0),
        max(int(round(top)), 0),
        min(int(round(right)), width),
        min(int(round(bottom)), height),
    )
    return Image.fromarray(grey).crop(box), len(scene.pages)


def _to_grey(bitmap):
    """BGRA -> tonuri de gri, cu compunere peste alb **dupa canalul alfa**.

    Aruncandu-l, pixelii complet transparenti de pe margine ies negri si pagina
    capata o rama pe care decupajele ar duce-o mai departe ca pe cerneala.
    """
    import numpy as np

    bitmap = np.asarray(bitmap)
    colour = bitmap[:, :, :3].astype(np.float32)
    alpha = bitmap[:, :, 3:4].astype(np.float32) / 255.0
    composed = colour * alpha + 255.0 * (1.0 - alpha)
    return composed.mean(axis=2).round().clip(0, 255).astype(np.uint8)


def render(model, musicxml: Union[str, Path], page: int = 0, dpi: float = 300.0):
    """O pagina randata, in tonuri de gri, plus cate pagini a produs partitura.

    Scena stie singura sa se randeze (`scene.render(page)`); noi doar alegem
    rezolutia si aducem imaginea la tonurile de gri pe care le asteapta restul
    conductei — aceeasi conventie ca la decupajele din `dataset.py`.
    """
    import numpy as np
    from PIL import Image

    scene = model(file=str(musicxml))
    if page >= len(scene.pages):
        raise IndexError(
            f"partitura are {len(scene.pages)} pagini, ceruta e {page + 1}"
        )

    scene.dpi = dpi
    return Image.fromarray(_to_grey(scene.render(scene.pages[page]))), len(scene.pages)
