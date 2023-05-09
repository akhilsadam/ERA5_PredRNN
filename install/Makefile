# Makefile for installation, setup, and running...
# installation directory, working directory, and data directories, respectively (in secrets file)
user=$(shell whoami)

venv:
	python3 -m venv ../venv --prompt "ERA5_PredRNN"
activate:
	source ../venv/bin/activate
install_requirements:
	pip install -r ../install/requirements.txt
	pip install -r ../PDE/requirements.txt
deactivate:
	deactivate
mkd:
	@while IFS= read -r file; do
	@	mkdir -p $file
	@done < "${user}_secrets.txt"

install: venv activate install_requirements mkd