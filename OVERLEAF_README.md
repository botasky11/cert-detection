# Overleaf Build Notes

This repository is Overleaf-ready for the English LaTeX report.

## Main document

Use `main.tex` as the main document.

## Compiler

Compile with **pdfLaTeX**. The included `latexmkrc` sets pdfLaTeX for Overleaf/latexmk automatically.

## Required files

Upload these paths together:

- `main.tex`
- `latexmkrc`
- `outputs/fig_*.png`
- `outputs/comparison/fig_*.png`

The image paths in `main.tex` are relative to the project root, so the `outputs/` folder must remain at the root level.
