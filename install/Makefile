# Makefile for installation, setup, and running...
.ONESHELL:
SHELL := /bin/bash
user=$(shell whoami)

venv:
	@python3 -m venv ../venv --prompt "ERA5_PredRNN"
install_requirements: 
	@source ../venv/bin/activate
	@pip install -r ../install/requirements.txt
	@pip install -r ../PDE/requirements.txt
deactivate:
	@deactivate

install: venv install_requirements