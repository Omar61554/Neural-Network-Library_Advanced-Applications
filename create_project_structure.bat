@echo off
REM create_project_structure.bat
REM Creates the requested project tree for a neural network project.

REM Change to script directory
cd /d "%~dp0"

echo Creating project directories...
if not exist lib mkdir lib
if not exist notebooks mkdir notebooks
if not exist report mkdir report

echo Creating top-level files if missing...
if not exist .gitignore echo __pycache__/ > .gitignore
if not exist README.md echo # Project Title > README.md
if not exist requirements.txt echo numpy>requirements.txt

echo Creating lib module files (placeholders)...
if not exist lib\__init__.py type nul > lib\__init__.py
if not exist lib\layers.py echo # layers placeholder > lib\layers.py
if not exist lib\activations.py echo # activations placeholder > lib\activations.py
if not exist lib\losses.py echo # losses placeholder > lib\losses.py
if not exist lib\optimizer.py echo # optimizer placeholder > lib\optimizer.py
if not exist lib\network.py echo # network placeholder > lib\network.py

echo Creating notebook placeholder (if missing)...
if not exist notebooks\project_demo.ipynb (
  copy /Y NUL notebooks\project_demo.ipynb >nul
)

echo Creating report placeholder (if missing)...
if not exist report\project_report.pdf (
  REM create a tiny PDF header so tools recognize it as a PDF
  (echo %%PDF-1.4)>report\project_report.pdf
  (echo %%âãÏÓ)>>report\project_report.pdf
)

echo Done. To customize files, edit the placeholders under lib\ and notebooks\.
pause
