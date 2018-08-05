ALL_FILES=traces.pdf ensemble.pdf data.pdf line1.pdf line2.pdf

default: $(ALL_FILES)

%.pdf: %.py plotting.py
	python -W ignore $<

clean:
	rm -f $(ALL_FILES)
