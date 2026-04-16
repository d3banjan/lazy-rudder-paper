.PHONY: paper clean values tables lean-status all
all: paper
%:
	$(MAKE) -C manuscript $@
