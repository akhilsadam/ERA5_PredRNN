import pde

grid = pde.UnitGrid([64, 64])                 # generate grid
state = pde.ScalarField.random_uniform(grid)  # generate initial condition

# eq = pde.DiffusionPDE(diffusivity=0.1)        # define the pde
# eq = pde.PDE({'c': 'laplace(c**3 - c - laplace(c))'})
eq = pde.PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"})
storage = pde.MemoryStorage()

result = eq.solve(state, t_range=1000, dt=1e-3, tracker=["progress", storage.tracker(1)])          # solve the pde
# result.plot()                                 # plot the resulting field
pde.movie(storage, 'movie.mp4')        # create a movie
