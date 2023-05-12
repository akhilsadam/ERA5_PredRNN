import numpy as np
data = dict(np.load('data.npz'))

data['dims'] = np.array([data['dims'].tolist(),]).astype(np.int32)

np.savez('data.npz', **data)