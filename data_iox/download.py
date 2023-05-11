import cdsapi
import uuid, os
import param
import convert
import importlib

# TODO add logging to log.txt, so same file is not requested twice
uid = str(uuid.uuid4())

user=os.popen('whoami').read().replace('\n','')
userparam=importlib.import_module('param',f'user.{user}_param')
os.makedirs(userparam.data_dir, exist_ok=True)
os.makedirs(f'{userparam.data_dir}/{uid}/', exist_ok=True)

c = cdsapi.Client()

c.retrieve('reanalysis-era5-single-levels',
    param.data,
    f'{userparam.data_dir}/{uid}/data.grib')

convert.convert(f'{userparam.data_dir}/{uid}/data.grib', f'{userparam.data_dir}/{uid}/')