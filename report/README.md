# Report Artifacts

This directory contains the ACL-style report project. It is separate from
`colab_notebooks/` so the Colab runtime bundle stays focused on the executable
notebook and its README.

- `latex/report.tex`: report source.
- `latex/report.pdf`: compiled report PDF.
- `latex/overleaf_acl_report.zip`: Overleaf upload package.
- `latex/figures/`: tracked figures used by the report.
- `latex/scripts/generate_acl_figures.py`: script for regenerating report
  figures from local meeting and diagnostic artifacts.

To rebuild the report locally:

```bash
cd report/latex
python scripts/generate_acl_figures.py
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```
