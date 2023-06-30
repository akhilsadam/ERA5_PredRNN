import os,sys, importlib
user='as_tacc'

spec = importlib.util.spec_from_file_location("module.name", f'./user/{user}_param.py')
userparam = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = userparam
spec.loader.exec_module(userparam)

_dir = userparam.param['work_dir']

# copy back all .png files using rsync from user@ls6.tacc.utexas.edu
os.makedirs('../model_ls6/', exist_ok=True)
os.system(f'rsync -zarv --include="*/" --include="*.png" --exclude="*" {user}@ls6.tacc.utexas.edu:{_dir}/model_ls6/* ../model_ls6/')
os.system(f'rsync -zarv --include="*/" --include="*.png" --exclude="*" {user}@ls6.tacc.utexas.edu:{_dir}/data/* ../data_ls6/')