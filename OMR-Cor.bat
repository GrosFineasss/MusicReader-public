@echo off
rem Lanseaza aplicatia. Dublu-clic pe acest fisier.
rem
rem %~dp0 e folderul in care sta fisierul asta, cu backslash la final. Toate
rem caile pornesc de la el, deci lansatorul merge si daca muti proiectul, si
rem daca il apelezi dintr-un scurtatura de pe Desktop.

set "ROOT=%~dp0"
set "PYTHONW=%ROOT%.venv\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
    rem Fara consola nu s-ar vedea nimic, asa ca o pastram deschisa aici.
    echo.
    echo   Mediul Python nu a fost gasit:
    echo     %PYTHONW%
    echo.
    echo   Creeaza-l cu:
    echo     uv venv --python 3.11
    echo     uv pip install -p .venv\Scripts\python.exe -e .[recognition,batch,ui,conversion]
    echo.
    pause
    exit /b 1
)

rem `start ""` porneste procesul si elibereaza fereastra de comanda imediat;
rem primul argument gol e titlul ferestrei, cerut de sintaxa lui `start`.
start "" "%PYTHONW%" "%ROOT%scripts\app.py"
