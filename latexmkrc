# Overleaf/latexmk configuration: use pdfLaTeX for the pandoc-generated report.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
