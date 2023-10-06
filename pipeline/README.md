To automatically handle data collection & conversion from the CDS repo.
Important pipeline files:
- `pipeline/README.md`: this file
- `pipeline/download.py` and `pipeline/convert.py`: download data from CDS (or PDE) and convert to npz (while generating snapshots)
- `pipeline/param.py`: parameters for dataset (also select PDE/CDS)