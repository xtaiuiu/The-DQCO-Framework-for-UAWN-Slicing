from network_classes.network_slice import Slice
from scenarios.UE_creators import create_UE_set
import numpy as np


def create_slice(n_UEs, tilde_R_l, tilde_R_u, b_width, network_radius):
    UE_set = create_UE_set(n_UEs, network_radius, tilde_R_l, tilde_R_u)
    return Slice(UE_set, b_width)


def create_slice_set(n_slices, network_radius):
    # create a randomly generated slice set
    slice_set = []
    bandwidths = np.array([0.1, 0.5, 1])
    tilde_R_ls = np.array([0.1, 0.2])
    tilde_R_us = np.array([0.2, 0.4])
    num_UEs = np.array([1])
    for i in range(n_slices):
        n_UEs = np.random.choice(num_UEs)
        b_width = np.random.choice(bandwidths)
        tilde_R_l = np.random.choice(tilde_R_ls)
        tilde_R_u = np.random.choice(tilde_R_us)
        s_tmp = create_slice(n_UEs, tilde_R_l, tilde_R_u, b_width, network_radius)
        slice_set.append(s_tmp)
    return slice_set