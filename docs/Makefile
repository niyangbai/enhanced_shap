# Minimal Makefile for Sphinx documentation
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html
html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html."

.PHONY: clean
clean:
	rm -rf "$(BUILDDIR)"
