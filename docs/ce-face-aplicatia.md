# Ce face aplicația asta

*Scris pentru cineva care nu se pricepe la muzică. Nu trebuie să știi să citești note ca să
înțelegi ce urmează.*

---

## Problema, în două propoziții

Există un teanc de aproximativ **500 de partituri de cor scrise de mână**. Ca să poată fi
folosite mai departe — tipărite curat, corectate, cântate mai sus sau mai jos, trimise altcuiva —
trebuie rescrise la calculator, notă cu notă.

Rescrisă de mână, o partitură ia **una-două ore**. Cinci sute de partituri înseamnă câteva luni de
om, la program întreg.

## Ce face aplicația

Fotografiezi pagina cu telefonul. Aplicația îți dă un fișier pe care îl deschizi în programul de
note (MuseScore, gratuit) și pe care îl poți edita ca pe un document Word: schimbi ce vrei,
tipărești, îl transpui mai sus sau mai jos, îl asculți.

Totul se întâmplă **pe laptopul tău**. Nicio poză nu pleacă pe internet, nu există abonament și nu
există un serviciu care într-o zi se închide. E o hotărâre luată la început și ținută de atunci.

---

## De ce e greu — partea care se înțelege cel mai greșit

Prima reacție a oricui este: „păi există de mult programe care scanează texte, nu-i tot aia?"

Nu e. Și merită explicat de ce, pentru că exact aici stă toată munca.

### 1. La text contează forma literei. La muzică, contează și **unde stă pe pagină**

Litera „a" înseamnă „a" oriunde ai pune-o pe rând. Un semn muzical nu funcționează așa. **Același
semn, urcat cu un rând mai sus, e un alt sunet.** Forma nu s-a schimbat deloc — poziția s-a
schimbat, și poziția *este* informația.

E ca și cum, într-un text, cuvântul „casă" ar însemna altceva dacă îl scrii cu doi milimetri mai
sus. Programul trebuie să măsoare, nu doar să recunoască.

### 2. Se citesc patru lucruri deodată, suprapuse

Într-o partitură de cor cântă patru voci în același timp: sopran, alt, tenor, bas. Pe hârtie, ele
nu stau una sub alta ca patru paragrafe. Stau **suprapuse pe aceleași rânduri**, deosebite prin
detalii mici — încotro merge codița notei, în sus sau în jos.

Deci programul nu citește un rând de la stânga la dreapta. Citește un rând și trebuie să iasă din
el **patru șiruri paralele**, fiecare cu sensul lui, fără să le încurce între ele.

### 3. Fiecare pagină e scrisă de altcineva

Asta e partea grea de tot. Cele ~500 de partituri nu sunt scrise de o singură persoană. Sunt scrise
de zeci de oameni diferiți, de-a lungul anilor, unii necunoscuți — o partitură e din 1994.

Un program antrenat pe scrisul unui singur om învață **scrisul acelui om**, și se împiedică la
următorul. Ce ne trebuie e ceva care să facă față și unui scris pe care nu l-a văzut niciodată.

### 4. Sub note e scris textul cântării

Pe hârtie, sub fiecare rând de note sunt scrise versurile, cu mâna, uneori la câțiva milimetri de
note. Nu trebuie citite — dar sunt acolo, în poză, și programul trebuie să nu se încurce din
cauza lor.

---

## De ce nu se rezolvă cu ceva gata făcut

Există programe care fac asta. Niciunul nu se potrivește, și fiecare din alt motiv:

- Cele foarte bune citesc note **tipărite**. Scrisul de mână e o cu totul altă problemă, iar ele
  se descurcă prost cu el.
- Cele care citesc scris de mână sunt făcute pentru cercetare, iar **licențele lor interzic**
  folosirea într-un produs vândut. Le poți folosi ca să scrii un articol, nu ca să vinzi.
- Iar cel mai precis dintre toate are o licență care ar obliga la publicarea întregului cod al
  produsului.

Așa că partea de citire e construită pornind de la un model cu licență curată, antrenat pe note
**tipărite**, care trebuie învățat să citească scris de mână.

---

## Unde suntem acum, cinstit

Pe o pagină grea de manuscris, cu 182 de note, programul nimerește în acest moment **16**.

E puțin, și e normal să fie: modelul din spate n-a văzut în viața lui scris de mână. N-a fost
învățat încă.

Ce **este** gata: tot lanțul din jur. Fotografia se îndreaptă singură, rândurile se găsesc,
rezultatul se verifică după regulile muzicii (o măsură trebuie să conțină exact atâtea note câte
îi trebuie — dacă nu, se semnalează), se scrie fișierul care se deschide în MuseScore, iar
aplicația poate lucra pe un teanc întreg de pagini și continua de unde a rămas dacă o oprești.

Și mai e ceva gata, care schimbă felul în care se recuperează greșelile: **corectarea**. Când
transcrierea iese greșit, o repari în fereastră — pagina sus, notele într-un tabel jos — iar
reparația nu se pierde: din ea se face material de învățat. Fiecare pagină corectată e o pagină pe
care programul n-o mai greșește la fel.

Ce lipsește nu e programul. E **materialul de învățat**.

---

## De ce foile alea cu căsuțe

Ca să învețe, unui asemenea program îi trebuie mii de pagini scrise de mână, în cât mai multe
scrisuri diferite, și pentru fiecare pagină trebuie să știi dinainte răspunsul corect. Nimeni nu
poate scrie mii de partituri.

Soluția e ocolită, și e miezul întregii idei:

**Un om nu scrie o partitură. Scrie o dată fiecare semn, în căsuțe.** O foaie, un sfert de oră.

Apoi calculatorul ia semnele acelea și **compune el pagini întregi** — pagini de partitură care
n-au existat niciodată, dar care sunt scrise cu mâna acelui om. Și, fiindcă el a compus pagina, știe
exact ce e pe ea. Are și întrebarea, și răspunsul.

Din foaia unui singur om ies sute de pagini de exercițiu. Din foile a zece oameni ies zece scrisuri
diferite, ceea ce e exact ce trebuie ca programul să facă față și unui scris nou.

Asta e ce se cere celor care completează o foaie: un sfert de oră, o dată. Nu să scrie partituri.

---

## Pe scurt

| | |
|---|---|
| **Ce intră** | o fotografie a unei pagini scrise de mână |
| **Ce iese** | un fișier de partitură editabil (MusicXML), deschis în MuseScore |
| **Unde rulează** | pe laptop, fără internet, fără abonament |
| **Cât durează** | secunde pe pagină; e mai lentă corectarea decât citirea |
| **Ce merge azi** | tot lanțul, de la poză la fișier |
| **Ce nu merge încă** | citirea e slabă (16 note din 182), fiindcă modelul n-a fost încă învățat scris de mână |
| **Ce urmează** | foile cu semne, de la cât mai mulți oameni, și învățarea propriu-zisă |
