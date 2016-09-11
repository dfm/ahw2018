ALL_FILES=traces.pdf ensemble.pdf

default: $(ALL_FILES)

%.pdf: %.py plotting.py
	python -W ignore $<

clean:
	rm -f $(ALL_FILES)
