all: clean informe

informe: informe.tex
	pdflatex informe.tex
	bibtex informe
	pdflatex informe.tex
	pdflatex informe.tex

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.synctex.gz
	rm -f *.out
	rm -f *.xml
	rm -f *.bcf
	rm -f *.blg
	rm -f *.bbl
	rm -f informe-blx.bib
	rm -f *.toc
