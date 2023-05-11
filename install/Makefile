# Makefile for installation, setup, and running...
# installation directory, working directory, and data directories, respectively (in secrets file)
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
mkd:
	@while IFS= read -r file; 		\
	 do 							\
		mkdir -p "$file";			\
	done < "${user}_secrets.txt";	\

install: venv install_requirements mkd