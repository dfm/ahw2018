DIRS      = mcmc

# You shouldn't need to edit below here.
default: all

.PHONY: all

all: force_look
	$(foreach d, ${DIRS}, (echo "Looking into ${d}:"; cd ${d}; ${MAKE} ${MFLAGS}) )

force_look:
	true

clean:
	$(foreach d, ${DIRS}, (echo "Cleaning ${d}:"; cd ${d}; $(MAKE) clean) )
