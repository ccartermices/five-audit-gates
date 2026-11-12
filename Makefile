# Thin wrapper over run_checks.py, for systems where make is available.
# Python alone is enough: python run_checks.py
PY ?= python
LEAN ?=

.PHONY: check demo trace record-hashes clean

check:
	$(PY) run_checks.py

demo:
	$(PY) checks/check_numbers.py checks/fixtures/manuscript_ok.md
	@$(PY) checks/check_numbers.py checks/fixtures/manuscript_bad.md > /dev/null 2>&1 \
	  && { echo "expected the bad fixture to fail, it passed"; exit 1; } \
	  || echo "[PASS] the bad fixture fails the trace, as it must"

trace:
	@test -n "$(MANUSCRIPT)" || { echo "usage: make trace MANUSCRIPT=path/to/manuscript.md"; exit 1; }
	$(PY) checks/check_numbers.py "$(MANUSCRIPT)"

record-hashes:
	$(PY) checks/check_hashes.py --write

clean:
	rm -rf checks/__pycache__
