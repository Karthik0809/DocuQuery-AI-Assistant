@echo off
setlocal
cd /d "%~dp0"
echo === pdflatex (1/4) ===
pdflatex -interaction=nonstopmode docuquery_ieee.tex
if errorlevel 1 goto :fail
echo === bibtex ===
bibtex docuquery_ieee
if errorlevel 1 goto :fail
echo === pdflatex (2/4) ===
pdflatex -interaction=nonstopmode docuquery_ieee.tex
if errorlevel 1 goto :fail
echo === pdflatex (3/4) ===
pdflatex -interaction=nonstopmode docuquery_ieee.tex
if errorlevel 1 goto :fail
echo.
echo OK: docuquery_ieee.pdf
echo If references changed, copy docuquery_ieee.bbl from this folder to keep single-pass pdflatex working.
goto :eof
:fail
echo FAILED — ensure MiKTeX or TeX Live is installed and in PATH.
exit /b 1
