import pde

grid = pde.UnitGrid([64, 64])                 # generate grid
state = pde.ScalarField.random_uniform(grid)  # generate initial condition

# eq = pde.DiffusionPDE(diffusivity=0.1)        # define the pde
eq = pde.PDE({'c': 'laplace(c**3 - c - laplace(c))'})
result = eq.solve(state, t_range=10)          # solve the pde
result.plot()                                 # plot the resulting field
