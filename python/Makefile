.PHONY: build lint typecheck test docs clean

build:
	pip install -e .

lint:
	flake8 .

typecheck:
	mypy src/greybox/ --ignore-missing-imports

test:
	pytest

docs:
	sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build

clean: docs-clean
