import cdsapi
import uuid, os, importlib
import logging
###########
import param
import convert
###########
logging.basicConfig(level = logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M', handlers = [logging.FileHandler('filename.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)
###########

# TODO add logging to log.txt, so same file is not requested twice
user=os.popen('whoami').read().replace('\n','')
uid = str(uuid.uuid4())
logger.info(f'User: {user}')
logger.info(f'UID: {uid}')

userparam=importlib.import_module(f'user.{user}_param')
datadir = userparam.param['data_dir']
os.makedirs(datadir, exist_ok=True)

if 'year' in param.data:
    cdatadir = f'{datadir}/CDS_{uid}/'
    os.makedirs(cdatadir, exist_ok=True)
    
    logger.info(f'Opening CDS API ... saving to {cdatadir}/data.grib')

    c = cdsapi.Client()
    c.retrieve('reanalysis-era5-single-levels',
        param.data,
        f'{cdatadir}/data.grib')

    logger.info('Converting CDS data...')
    convert.convert(f'{cdatadir}/data.grib', cdatadir, logging.getLogger('convert'))
else:
    cdatadir = f'{datadir}/PDE_{uid}/'
    os.makedirs(cdatadir, exist_ok=True)
    gen=importlib.import_module('PDE', package='ERA5_PredRNN')
    
    logger.info(f'Operating PDE API ... will save to {cdatadir}')  
    assert 't_step' in param.data, logger.critical('t_step must be provided for PDE data')
    assert 'dt' in param.data, logger.critical('dt must be provided for PDE data')
    assert 'nvar' in param.data, logger.critical('nvar must be provided for PDE data')
    assert 'gshape' in param.data, logger.critical('gshape must be provided for PDE data')
    
    final_data = gen.gen_data(param.data['t_step'], param.data['dt'], param.data['nvar'], param.data['gshape'], cdatadir, logging.getLogger('pde'))
    convert.convert(f'{cdatadir}/data.grib', cdatadir, logging.getLogger('convert'), pygrib_fmt=False, final_data=final_data)
    
    