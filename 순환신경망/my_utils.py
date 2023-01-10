PYTHONIOENCODING='utf8'

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'

PRECISION = 3

def svd(M):
    """Returns the Singular Value Decomposition of M (via numpy), with all components returned in 
    matrix format
    """
    U, s, Vt = np.linalg.svd(M)
    
    # Put the vector singular values into a padded matrix
    S = np.zeros(M.shape)
    np.fill_diagonal(S, s)
    
    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)

