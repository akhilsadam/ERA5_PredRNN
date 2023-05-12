# Makefile for installation, setup, and running...
.ONESHELL:
SHELL := /bin/bash
user=$(shell whoami)
dir=$(shell pwd)
BASE=$(shell builtin cd $(dir)/../; pwd)

venv:
	@python3 -m venv ../venv --prompt "ERA5_PredRNN"
install_requirements: 
	@source ../venv/bin/activate
	@pip install -r ../install/requirements.txt
	@pip install -r ../PDE/requirements.txt
pypath:
	@echo "PYTHONPATH=$$PYTHONPATH:$(BASE)" >> ../venv/bin/activate
	@echo "$$PYTHONPATH"
deactivate:
	@deactivate

install: venv install_requirements pypath