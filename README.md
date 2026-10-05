# MusicReader — OMR local pentru partituri de cor

Poza unei partituri corale, tipărite sau scrise de mână → MusicXML de deschis în MuseScore.
Rulează **integral local**, pe procesor, fără servicii în cloud.

Proiect personal, făcut ca partiturile corului de la biserică să se scrie mai repede. Aici e doar
codul: pozele, referințele, datele de antrenare și modelele antrenate nu sunt în repository.

## Ce citește acum

Note citite corect, pe pagini pe care modelul nu le-a văzut la antrenare (octombrie 2026):

| ce | modelul publicat | după antrenările noastre |
|---|---|---|
| o pagină corală tipărită, 4 variante (2 scanuri de hârtie) | 33% | **91%** |
| corale generate (PDMX, Bach), ținute deoparte | 66–70% | **97–98%** |
| un manuscris nevăzut, fără nicio pagină a lui la antrenare | 15% | **29%** (din alte 72 de mâini) |
| același manuscris, antrenat pe cealaltă pagină a lui | 15% | **41–48%** |

Scrisul de mână e partea grea: fiecare partitură e scrisă de altă mână, iar ce urcă cifra sunt
pagini reale de mână cu varianta lor din MuseScore.

## Cum funcționează

Modelul e **SMT** (Sheet Music Transformer, varianta GrandStaff,
[`PRAIG/smt-grandstaff`](https://huggingface.co/PRAIG/smt-grandstaff)), preantrenat pe pian tipărit
și reantrenat aici pe partituri corale. Citește câte un **sistem** (2–3 portative) și scrie kern
(Humdrum), token cu token.

1. Poza → alb-negru, îndreptare, ștergerea versurilor (la scris de mână).
2. Portativele găsite și grupate în sisteme.
3. Fiecare sistem decupat → model → kern.
4. Kern → model muzical intern (voci, măsuri, chei) → reguli de validare → MusicXML.

```
app/
  music_model/    Score / Part / Measure / Note — reprezentarea internă
  musicxml/       export MusicXML 4.0
  validation/     reguli muzicale
  recognition/    motorul (grandstaff_engine.py, incremental.py), kern.py, bekern.py, page.py
  evaluation/     comparația cu o referință (metrics.py, align.py, reference.py)
  preprocessing/  îndreptare, linii de portativ, ștergerea textului
  training/       bucla de antrenare, perechile, generatorul de pagini corale (bach.py, satb.py)
  batch/, ui/, correction/   lotul reluabil, fereastra, corectarea
scripts/          generare de date, antrenare, măsurare (vezi mai jos)
tests/            ~865 de teste; cele care cer date locale sau vendor/ se sar dacă lipsesc
```

`docs/ce-face-aplicatia.md` și `docs/cum-functioneaza-miezul.md` explică pe larg aplicația și miezul.

## Instalare (Windows)

```bash
python -m venv .venv
# torch, build-ul CPU (fara --index-url iei varianta CUDA, 2,5 GB inutili)
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0
.venv/Scripts/python.exe -m pip install -e ".[recognition,batch,ui,conversion,dev]"

# codul tert, nu se versioneaza (MIT)
git clone https://github.com/antoniorv6/SMT.git vendor/SMT
git clone https://github.com/JuanCarlosMartinezSevilla/ISMIR-Jazzmus.git vendor/ISMIR-Jazzmus
.venv/Scripts/python.exe scripts/ia_bravura.py      # fontul Bravura (OFL)

.venv/Scripts/python.exe -m pytest -q
```

Greutățile modelului publicat se descarcă singure la prima rulare. Pentru paginile generate e nevoie
de **MuseScore 4**. Fereastra: dublu-clic pe `OMR-Cor.bat`.

## Rulare

```bash
# o pagina
.venv/Scripts/python.exe scripts/transcribe.py poza.jpg
# un lot (poze, PDF-uri, foldere), reluabil
.venv/Scripts/python.exe scripts/batch.py folder --tip handwritten
# acuratete fata de o referinta scrisa in MuseScore
.venv/Scripts/python.exe scripts/evaluate.py --ref referinta.musicxml --imagine poza.png --curata
# perechi de antrenare dintr-o pagina scrisa de mana + varianta ei din MuseScore
.venv/Scripts/python.exe scripts/dataset.py --imagine pagina.png --ref pagina.musicxml \
    --out data/antrenare --curata --scriitor cineva
# pagini corale tiparite generate (PDMX), si cu un portativ de solo deasupra (--solo)
.venv/Scripts/python.exe scripts/bach.py --lista acceptate.json --out data/pdmx --dpi 150,200
# antrenare, reluabila; --pasi e numarul TOTAL de pasi
.venv/Scripts/python.exe scripts/finetune.py --date data/antrenare/date.jsonl --out models/x \
    --mod tot --pasi 8000
```

## Date folosite (nu sunt în repository)

- **PDMX** ([Zenodo 15571083](https://zenodo.org/records/15571083)), partituri MuseScore în domeniul
  public / CC0 → coralele SATB, selectate de `scripts/pdmx_satb.py`.
- **Coralele Bach** din corpusul `music21` (ediția Greentree; licență neclară pentru uz comercial).
- **MusiCorpus v1.0** ([LINDAT](http://hdl.handle.net/20.500.12800/1-6147), CC BY-NC-SA 4.0),
  manuscrise cu MusicXML pe sistem → `scripts/musicorpus.py`.
- **GrandStaff** (pian), cum îl folosește SMT.

Câteva dintre aceste surse sunt **necomerciale**: modelele antrenate pe ele nu se pot folosi într-un
produs vândut.

## Licență

Codul acestui repository nu are încă o licență declarată. SMT și ISMIR-Jazzmus sunt MIT, `music21`
BSD, PyTorch BSD, Bravura OFL.
