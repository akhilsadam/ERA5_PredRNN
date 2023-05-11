import cdsapi
import uuid, os
import param
import convert
import importlib
import logging
logging.basicConfig(level = logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M', handlers = [logging.FileHandler('filename.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# TODO add logging to log.txt, so same file is not requested twice
uid = str(uuid.uuid4())

user=os.popen('whoami').read().replace('\n','')
userparam=importlib.import_module('param',f'user.{user}_param')
os.makedirs(userparam.data_dir, exist_ok=True)
os.makedirs(f'{userparam.data_dir}/{uid}/', exist_ok=True)

logger.info(f'User: {user}')
logger.info(f'UID: {uid}')
logger.info(f'Opening CDS API ... saving to {userparam.data_dir}/{uid}/data.grib')

c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels',
    param.data,
    f'{userparam.data_dir}/{uid}/data.grib')

logger.info('Converting CDS data...')
convert.convert(f'{userparam.data_dir}/{uid}/data.grib', f'{userparam.data_dir}/{uid}/', logger.getLogger('convert'))