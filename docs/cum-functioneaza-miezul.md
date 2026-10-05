# Cum funcționează miezul

*Continuarea lui [ce-face-aplicatia.md](ce-face-aplicatia.md). Aici e lanțul propriu-zis, pas cu
pas, tot în cuvinte simple — dar cu numele fișierelor, ca să poți sări în cod.*

---

## Lanțul, dintr-o privire

```
fotografie
   │
   ├─ 1. îndreptare          pagina strâmbă devine dreaptă        app/preprocessing/
   ├─ 2. găsirea rândurilor  unde sunt portativele, câte pe sistem app/staff_detection/
   ├─ 3. citirea             imagine → șir de simboluri            app/recognition/
   ├─ 4. modelul intern      șirul devine muzică structurată        app/music_model/
   ├─ 5. verificarea         ce nu se închide se semnalează         app/validation/
   └─ 6. scrierea            MusicXML pentru MuseScore              app/musicxml/
   │
fișier editabil
```

Peste ele, două învelișuri: **fereastra** (`app/ui/`) pentru o pagină pe rând, și **lotul**
(`app/batch/`) pentru un teanc întreg, reluabil dacă îl oprești.

---

## 1. Îndreptarea

O poză de telefon e strâmbă, are umbre și e făcută oblic. Pasul ăsta o aduce la o pagină dreaptă,
cu alb curat și negru curat.

Tot aici se încearcă **ștergerea versurilor** scrise sub note. Cu o rezervă spusă pe față: pe
manuscrisele reale, textul e scris uneori la câțiva milimetri de portativ, iar acolo ștergerea e
imposibilă geometric fără să tai și din note. Ce se poate se șterge; ce nu, rămâne.

## 2. Găsirea rândurilor

Un „portativ" e grupul de cinci linii orizontale pe care stau notele. Un „sistem" e mănunchiul de
portative care se cântă în același timp — la cor, de obicei două (sopran+alt sus, tenor+bas jos).

Numărul de portative dintr-un sistem **nu e mereu același**, nici măcar în aceeași pagină: unele
cântări au o voce de solo pe portativ separat, deci sisteme de trei.

Aici s-a luat o hotărâre care se vede peste tot mai departe: **gruparea se declară, nu se ghicește.**
Programul nu are cum să deosebească portativul de solo de cel de sopran — au aceeași cheie. Deci i
se spune, ca argument. Informația care se știe se scrie; nu se deduce mai târziu cu aproximație.
(`app/staff_detection/staves.py`)

## 3. Citirea — partea cu inteligență artificială

Aici intră modelul. Și aici e prima decizie care chiar contează:

> **Modelul citește un sistem întreg dintr-odată, nu notă cu notă.**

Motivul: pe același portativ sunt scrise două voci suprapuse, deosebite prin încotro merge codița.
Dacă tai imaginea în note și le dai una câte una, pierzi tocmai legătura care spune care notă e a
cui. Citind tot rândul deodată, modelul poate ține cele două șiruri paralele.

Ce iese nu e direct muzică, ci un **șir de simboluri** într-un format text (Humdrum `**kern`), unde
sunt scrise și lucruri de tipul „aici rândul se desparte în două voci" și „aici se unesc la loc".
(`app/recognition/`, în special `kern.py` și `bekern.py`)

## 4. Modelul intern

Șirul de simboluri se transformă în structuri proprii: partitură → voce → măsură → notă.

Două alegeri de aici se simt până la capăt:

- **Duratele se țin ca fracții exacte, niciodată ca zecimale.** O treime dintr-o măsură scrisă ca
  0,333… nu se mai adună niciodată exact la 1, iar verificarea de la pasul următor ar da alarme
  false pentru totdeauna. Cu fracții, se adună exact.
- **Nucleul nu știe nimic despre imagini și nimic despre inteligență artificială.** `music_model/`,
  `musicxml/` și `validation/` nu importă nimic din `recognition/`.

A doua e mai importantă decât pare. Când rezultatul e greșit, întrebarea e mereu „a citit prost, sau
am scris noi prost mai departe?". Fiindcă partea determinată de reguli e separată și testată
singură, răspunsul se află imediat. Dacă totul ar fi amestecat, fiecare greșeală ar fi o anchetă.

## 5. Verificarea — fără inteligență artificială

Un set de reguli muzicale se plimbă peste rezultat și semnalează ce nu se închide: o măsură care ar
trebui să conțină patru pătrimi și conține una, o voce care dispare, o notă în afara ambitusului.

Nu repară nimic. **Arată.** Un program care „repară" singur ce n-a înțeles produce un rezultat care
pare corect și nu e — cel mai prost fel de greșeală. (`app/validation/rules.py`)

## 6. Scrierea

Rezultatul devine MusicXML — formatul pe care îl citesc toate programele de note. Se deschide în
MuseScore și se editează.

Un amănunt cu motiv: partitura se scrie cu **patru părți separate**, câte una de voce, deși pe
hârtie erau strânse pe două rânduri. Așa se pot corecta vocile una câte una, iar MuseScore le poate
strânge la loc cu o comandă. Formatul e ales pentru cine îl folosește, nu pentru ce e comod
înăuntru.

---

## Cum se măsoară dacă merge

Nu prin impresii. Se ia o pagină al cărei răspuns corect e scris de om, se dă prin lanț, și se
compară notă cu notă. Ies cifre: câte note nimerite din câte, câte măsuri ieșite exact, și o rată de
greșeală.

**Cifra de acum, pe pagina grea de manuscris: 16 note nimerite din 182, o singură măsură exactă din
56.** Numărul ăsta e ținut fix ca reper: orice schimbare în cod care îl mișcă, în orice direcție, se
oprește și se explică înainte de a merge mai departe. Un reper care se mișcă pe tăcute nu mai e
reper. (`scripts/masoara.py`, `app/evaluation/`)

---

## 7. Corectarea — și de ce nu e doar o comoditate

Când transcrierea iese greșită, o poți repara în fereastră: pagina sus, notele într-un tabel jos,
câte una pe rând, cu înălțimea și durata de schimbat. Validatorul de la pasul 5 rulează la fiecare
schimbare și îți arată ce nu se închide încă.

Partea care contează nu e că repari fișierul. E că **reparația devine material de învățat**: din
pagina corectată plus poza din care a ieșit se face exact același fel de pereche „întrebare +
răspuns" ca cele descrise mai jos. Ce corectezi azi e ce nu mai greșește mâine.

Ce **nu** e: un editor de partituri. Nu se desenează note cu mausul — pentru asta e MuseScore, și o
face mai bine. Ce nu face MuseScore, și e tot rostul ferestrei, e să țină legătura dintre corecție și
imaginea din care a ieșit. (`app/correction/editare.py`, `app/ui/corectare.py`)

---

## Cum se face materialul de învățat

Partea explicată în celălalt document — cu foile de căsuțe — arată în cod așa:

```
foaia completată (fotografie)
   │
   ├─ decupare        fiecare căsuță devine o glifă      app/training/glife.py
   ├─ colecție        glifele unui om = un „stil"        app/training/repository.py
   ├─ compunere       pagini noi, scrise cu acel stil    app/training/sinteza.py
   ├─ perechi         imagine + răspunsul corect         scripts/sintetic.py
   └─ învățare        modelul se ajustează pe ele        app/training/loop.py
```

Trei precauții din lanțul ăsta merită spuse, fiindcă fiecare a fost plătită cu o greșeală:

- **Se compune un sistem odată, dintr-o felie.** Dacă s-ar compune o pagină întreagă și abia apoi
  s-ar întreba ce note au căzut pe ce rând, imaginea și răspunsul ar putea să nu corespundă — iar
  modelul ar învăța, cu încredere, lucruri false. Așa, amândouă ies din aceleași măsuri și n-au de
  unde să se contrazică.
- **Fiecare răspuns e verificat dus-întors înainte să intre în set.** Ce nu se închide exact nu
  intră, și se raportează. Mai bine niciun material decât material care minte.
- **Împărțirea între „învață" și „se testează pe" se face pe om, nu pe pagină.** Dacă două pagini
  ale aceleiași persoane ajung una la învățat și una la testat, cifra iese frumoasă și e minciună:
  ai măsurat cât de bine recunoaște un scris pe care tocmai l-a învățat.

---

## Ce nu face, și nu va face în versiunea asta

- Nu citește **versurile**. Ele se scriu la calculator; aplicația trebuie doar să nu se
  încurce din cauza lor.
- Nu ascultă și nu cântă. Asta o face MuseScore.
- Nu trimite nimic pe internet. Niciodată — e o regulă a proiectului, nu o setare.
