import numpy as np


def prd_uncertainty(mu_mcs):
    # = np.var(mu_mcs, 0)
    return np.var(mu_mcs, 1)
