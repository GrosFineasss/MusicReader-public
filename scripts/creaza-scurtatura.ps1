# Creeaza o scurtatura "OMR Cor" pe Desktop.
#
# Se ruleaza o singura data, manual:
#     powershell -ExecutionPolicy Bypass -File scripts\creaza-scurtatura.ps1
#
# E un script separat, nu ceva ce face aplicatia la pornire: nimic nu ar trebui
# sa scrie pe Desktop-ul tau fara sa i-o ceri explicit.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root 'OMR-Cor.bat'

if (-not (Test-Path $target)) {
    Write-Error "Nu gasesc lansatorul: $target"
    exit 1
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'OMR Cor.lnk'

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $target
# Directorul de lucru trebuie sa fie radacina proiectului: de acolo isi gaseste
# aplicatia `.venv`, `models/` si `data/`.
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'Conversie partituri de cor in MusicXML'
$shortcut.WindowStyle = 7   # pornit minimizat: fereastra reala o deschide aplicatia

# `.bat` nu are pictograma proprie; imprumutam una din shell32 ca sa nu arate
# ca un fisier text pe Desktop.
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,70"
$shortcut.Save()

Write-Host "Scurtatura creata: $link"
Write-Host "O poti fixa pe bara de activitati cu clic dreapta."
