PY ?= .venv/bin/python

.PHONY: venv collect build analyze all test

venv:
	uv venv .venv && uv pip install --python $(PY) -e '.[dev]'

collect:
	$(PY) -m perun collect

build:
	$(PY) -m perun build

analyze:
	$(PY) -m perun analyze

all:
	$(PY) -m perun all

test:
	$(PY) -m pytest -q
