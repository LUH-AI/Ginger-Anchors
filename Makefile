# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := Ginger Anchors
PACKAGE_NAME := ginger_anchors

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests

.PHONY: help install-dev check format clean clean-build build publish test

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* build            to build a dist"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
FLAKE8 ?= flake8

install-dev:
	$(PIP) install -e ".[dev]"
	check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	$(BLACK) ${TESTS_DIR} --check || :

check-isort:
	$(ISORT) ${SOURCE_DIR} --check || :
	$(ISORT) ${TESTS_DIR} --check || :

check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-isort check-flake8

format-black:
	$(BLACK) ${SOURCE_DIR}
	$(BLACK) ${TESTS_DIR}

format-isort:
	$(ISORT) ${SOURCE_DIR}
	$(ISORT) ${TESTS_DIR}

format: format-black format-isort

test:
	$(PYTEST) ${TESTS_DIR}

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	read -p "Did you update the version number?"
	
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${NAME}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"

# Clean up any builds in ./dist as well as doc, if present
clean: clean-build 