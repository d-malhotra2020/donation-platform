.PHONY: bench bench-fast bench-ablations bench-clean bench-install bench-test help

VENV := .venv-bench
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Donation Platform — Recommender Benchmark (Slice 1)"
	@echo ""
	@echo "  make bench-install   Install pinned deps into $(VENV)"
	@echo "  make bench           Run the full benchmark (~15 min CPU)"
	@echo "  make bench-fast      Smoke test on tiny dataset (<30 sec)"
	@echo "  make bench-ablations Run the 3x3x2 hyperparam sweep (~1 hr)"
	@echo "  make bench-test      Run pytest unit tests for metrics + invariants"
	@echo "  make bench-clean     Remove bench/results/ outputs"

$(PY):
	python3.11 -m venv $(VENV)

bench-install: $(PY)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r bench/requirements.txt

bench: bench-install
	$(PY) -m bench.eval.run

bench-fast: bench-install
	BENCH_FAST=1 $(PY) -m bench.eval.run

bench-ablations: bench-install
	$(PY) -m bench.eval.ablations

bench-test: bench-install
	$(PY) -m pytest bench/tests -v

bench-clean:
	rm -rf bench/results/*.json bench/results/*.md bench/results/*.png
	touch bench/results/.gitkeep
