import numpy
from matplotlib import pyplot as plt
import pde

def gen_data(t_step, dt, nvar, gshape, data_path, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger()

    assert type(t_step) == int and t_step > 0, logger.critical('N of time steps must be a positive integer')
    assert type(dt) == float and dt > 0, logger.critical('Time step must be a positive float')
    assert nvar == 1, logger.critical('Number of variables must be 1 - other numbers not supported')
    assert type(gshape) == list, logger.critical('Grid shape must be a list')
    assert len(gshape) == 2, logger.critical('Grid shape must be 2D')
    assert type(gshape[0]) == int, logger.critical('Grid shape [0] must be an integer')
    assert type(gshape[1]) == int, logger.critical('Grid shape [1] must be an integer')
    assert t_step % 2 == 0, logger.critical('Time step must be even')
    assert gshape[0] % 2 == 0, logger.critical('Grid shape [0] must be even')
    assert gshape[1] % 2 == 0, logger.critical('Grid shape [1] must be even')

    data_shape = [t_step, nvar, *gshape]
    ############################################## Make data
    logger.info('Generating data...')

    grid = pde.UnitGrid(gshape) # generate grid
    state = pde.ScalarField.random_uniform(grid)  # generate initial condition

    # eq = pde.DiffusionPDE(diffusivity=0.1) 
    # eq = pde.PDE({'c': 'laplace(c**3 - c - laplace(c))'})
    eq = pde.PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"}) # define the pde
    storage = pde.MemoryStorage()

    eq.solve(state, t_range=t_step-1, dt=dt, tracker=["progress", storage.tracker(1)]) # solve the pde

    logger.info('Saving PDE Movie...')
    pde.movie(storage, f'{data_path}/movie.mp4') # create a movie
    ############################################## Make data
    logger.info('Making data...')
    data = numpy.zeros(data_shape)
    for i, (time, field) in enumerate(storage.items()):
        data[i,0,:,:] = field.data
    return data
    
    
