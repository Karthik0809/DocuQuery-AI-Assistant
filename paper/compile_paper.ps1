# Full LaTeX + BibTeX build. Requires pdflatex and bibtex on PATH (MiKTeX / TeX Live).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-Pdflatex([string]$label) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    & pdflatex -interaction=nonstopmode docuquery_ieee.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
}

try {
    Run-Pdflatex "pdflatex (1/4)"
    Write-Host "=== bibtex ===" -ForegroundColor Cyan
    & bibtex docuquery_ieee
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
    Run-Pdflatex "pdflatex (2/4)"
    Run-Pdflatex "pdflatex (3/4)"
    Write-Host "`nOK: docuquery_ieee.pdf" -ForegroundColor Green
    Write-Host "After changing references.bib, copy the new docuquery_ieee.bbl into git for one-pass pdflatex."
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
