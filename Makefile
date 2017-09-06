.PHONY: clean-pyc clean-build

help:
	@echo "clean-build -- remove build scaffolding"
	@echo "clean-pyc -- remove auxiliary python files"
	@echo "clean-cython -- remove cython auxiliary files"
	@echo "clean -- total cleaning of project files"
	@echo "test - run main tests"
	@echo "build - build project using virtualenv, useful for development purposes"
	@echo "build-ext - locally build and compile the cython sources"
	@echo "docs - build sphinx documentation"
	@echo "coverage - report on code coverage"
	@echo "test-lisp - run the lisp unitests"

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -rf zubr_env
	rm -rf html_env

clean-pyc:
	find zubr/ -name '*.pyc' -exec rm -f {} +
	find zubr/ -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-cython:
	find zubr/ -name '*.so' -exec rm -f {} +
	find zubr/ -name '*.c' -exec rm -f {} +

clean: clean-build clean-pyc clean-cython

build-ext:
	python setup.py build_ext --inplace

build-local:
	virtualenv -p python zubr_env &&\
	source zubr_env/bin/activate &&\
	pip install -r requirements.txt &&\
	python setup.py install

docs:
	sphinx-build -aE docs build/docs > /dev/null

test:
	python setup.py test

test-lisp:
	python -m zubr.zubr_lisp --test zubr/zubr_lisp/test

coverage:
	coverage run --source zubr setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html
