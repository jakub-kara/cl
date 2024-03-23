import density_routines
import numpy as np

data = np.array([[0,0],[1,1],[2,2]], dtype=float)
feature_vars = np.array([1,1], dtype=float)/np.sqrt(2)
use = np.array([0,1], dtype=int) + 1
coords = np.array([0,0], dtype=int)
coeff = 1
density = np.float64(0)

print(density)
density = density_routines.get_density_i4(density, \
    data.T, feature_vars, use, coords, coeff)
print(density)
