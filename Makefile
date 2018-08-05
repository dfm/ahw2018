.SUFFIXES:
.SUFFIXES: .pdf .tex

LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
RM          = rm -rf

TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
SUFF        = pdf

RM_TMP      = $(foreach d, ${TEX_FILES}, ${RM} $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun $*.log

# Enter the name(s) of the .tex file(s) that you want to compile.
TEX_FILES = mcmc
DIRS      = figures

# You shouldn't need to edit below here.
default: mcmc.pdf

.PHONY: figures

figures: force_look
	$(foreach d, ${DIRS}, (echo "Looking into ${d}:"; cd ${d}; ${MAKE} ${MFLAGS}) )

mcmc.pdf: mcmc.tex figures
	${LATEX} mcmc.tex
	( ${CHECK_RERUN} && ${LATEX} mcmc.tex ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} mcmc.tex ) || echo "Done."

force_look:
	true

clean:
	$(RM_TMP)
	$(foreach d, ${DIRS}, (echo "Cleaning ${d}:"; cd ${d}; $(MAKE) clean) )
