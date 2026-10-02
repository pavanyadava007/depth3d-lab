PY ?= ../.venv/bin/python
TECTONIC ?= $(HOME)/.local/bin/tectonic
CERT ?= /etc/pki/tls/certs/ca-bundle.crt

.PHONY: all tables figs pdf check clean

all: pdf check

tables:
	$(PY) make_tables.py

figs:
	$(PY) make_figs.py

pdf: tables figs
	SSL_CERT_FILE=$(CERT) $(TECTONIC) main.tex

# no unresolved citations, no long dashes, page count in range
check:
	@pdfinfo main.pdf | grep Pages
	@if pdftotext main.pdf - | grep -n '\[?\]\|??'; then echo "unresolved reference"; exit 1; fi
	@if pdftotext main.pdf - | grep -nP '\x{2013}|\x{2014}'; then echo "long dash in pdf"; exit 1; fi
	@if grep -n -- '--' main.tex; then echo "double hyphen in main.tex"; exit 1; fi
	@echo "checks passed"

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
