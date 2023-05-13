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

    grid = pde.UnitGrid(gshape, periodic=[False, False]) # generate grid
    state = pde.ScalarField.random_uniform(grid)  # generate initial condition

    # set boundary conditions `bc` for all axes
    bc_x_left = {"value": "-0.1*sin(y / 2)"}
    bc_x_right = {"value_expression": "0.1*sin(y / 2 + t/10)"}
    bc_y_right = {"value_expression": "sin(x / 2 + t/5)"}
    bc_y_left = "neumann" #{"value": "0.1*sin(x / 2)"} # 
    bc_x = [bc_x_left, bc_x_right]
    bc_y =  [bc_y_left, bc_y_right]# "auto_periodic_neumann" #
    bc=[bc_x, bc_y]


    # eq = pde.DiffusionPDE(diffusivity=0.1) 
    eq = pde.PDE({'c': 'laplace(c**3 - c - laplace(c)) + 0.1*d_dy(c)'}, bc=bc)
    # eq = pde.PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"}, bc=bc) # define the pde
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
    
    
