import cdsapi
import uuid
import param
import convert

# TODO add logging to log.txt, so same file is not requested twice

c = cdsapi.Client()
uid = str(uuid.uuid4())
c.retrieve('reanalysis-era5-single-levels',
    param.data,
    f'{uid}.grib')

convert.convert(f'{uid}.grib')