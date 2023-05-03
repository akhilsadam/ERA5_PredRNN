import os
from time import sleep
from osgeo import gdal
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib

data_ds = pygrib.open('download.grib')