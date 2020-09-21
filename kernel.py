import numpy as np
import nibabel as nb
import os
import h5py, hdf5plugin
import multiprocessing
from functools import partial

def load_z(filename, mean, std):
    eps = 0.001
    with h5py.File(filename, 'r') as f:
            x = f["data"][:] - mean
            x /= (std + eps)
    return x


def compute_residual(h5file, train_h5files, stdfile, meanfile, outfile):
    with h5py.File(stdfile, 'r') as f:
        std = f["data"][:]
    with h5py.File(meanfile, 'r') as f:
        mean = f["data"][:]
    mask = 1.0*(nb.load("/workspace/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii").get_fdata()!=0)
    zi = load_z(h5file, mean, std)
    outdata = zi.astype(np.float)
    for j, filenamei in enumerate(train_h5files):
        zj = load_z(filenamei, mean, std).astype(np.float)*mask
        normj = np.linalg.norm(zj)
        coef = (outdata*zj).sum()/normj
        print(filenamei)
        outdata -= coef*zj/normj
    
    with h5py.File(outfile, 'w') as f:
        f.create_dataset("data", data=outdata, dtype="f2", **hdf5plugin.Blosc())