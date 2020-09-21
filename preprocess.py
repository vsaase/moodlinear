import nibabel as nb
import numpy as np
import os
import h5py, hdf5plugin
from scipy.stats import norm
import multiprocessing
from functools import partial

def nii2int16(path):
    nii = nb.load(path)
    data = np.round(nii.get_fdata()).astype(np.int16)
    nii = nb.Nifti1Image(data, header=nii.header, affine=nii.affine)
    nii.header.set_data_dtype(np.int16)
    nb.save(nii, path)

def register_MNI_linear(filename, outdir):
    if not os.path.isdir(outdir):
        return False
    if not os.path.isfile(filename):
        return False
    
    print(f"registering {filename}")
    mni = f"/workspace/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_skullstripped.nii"

    mprage = filename
    mpragemni = f"{outdir}/mni_linear.nii.gz"
    if not os.path.isfile(mpragemni):
        if not os.path.isfile(f"{outdir}/MPRAGE_to_MNI0GenericAffine.mat"):
            cmd = (
                f"antsRegistration --dimensionality 3 -v --output {outdir}/MPRAGE_to_MNI "
                f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mni},{mprage},1] --use-histogram-matching"
                f" --transform Rigid[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
                f" --transform Affine[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
            )
            os.system(cmd)
        cmd = (
            f"antsApplyTransforms --interpolation Linear -v -d 3 -u float"
            f" -i {mprage} -r {mni} -o {mpragemni} -t {outdir}/MPRAGE_to_MNI0GenericAffine.mat"
        )
        os.system(cmd)
        nii2h5(mpragemni, f"{outdir}/mni_linear.h5")
    return f"{outdir}/mni_linear.h5"

def MNI_to_world_linear(infile, reffile, transformdir, outfile):
    if not os.path.isdir(transformdir):
        return False
    if not os.path.isfile(infile):
        return False
    if not os.path.isfile(reffile):
        return False
    
    cmd = (
        f"antsApplyTransforms --interpolation Linear -v -d 3 -u float"
        f" -i {infile} -r {reffile} -o {outfile} -t [{transformdir}/MPRAGE_to_MNI0GenericAffine.mat,1] "
    )
    #print(cmd)
    os.system(cmd)

def residual_to_score(infile, outfile):
    with h5py.File(infile, 'r') as f:
            x = f["data"][:]
    #x = 2*norm.cdf(np.abs(x)/16) - 1
    x = np.clip(x,-4,4)**2/16
    #x = np.abs(x)

    with h5py.File(outfile, 'w') as f:
        f.create_dataset("data", data=x, dtype=x.dtype, **hdf5plugin.Blosc())

def residual_to_globalscore(infile, outfile):
    with h5py.File(infile, 'r') as f:
            x = f["data"][:]
            
    x = (x.astype(np.float)**2/16)
    score = np.clip(x[x>0].mean(),0,1)

    with open(outfile,'w') as f:
        f.write(str(score))

def reg(i, indir, outdir):
    print(i)
    filename = f"{indir}/{str(i).zfill(5)}.nii.gz"
    outdiri = f"{outdir}/{str(i).zfill(5)}"
    if not os.path.exists(outdiri):
        os.makedirs(outdiri)
    register_MNI_linear(filename, outdiri)

def register_dir(indir, outdir, nproc=1):
    if not os.path.isdir(outdir):
        return False
    if not os.path.isdir(indir):
        return False
    if nproc == 1:
        for i in range(800):
            reg(i, indir, outdir)
    else:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count()//nproc)
        pool = multiprocessing.Pool(nproc)
        pool.map(partial(reg, indir=indir, outdir=outdir, linear=linear), range(800))


def nii2h5(infile, outfile, dtype=None):
    nii = nb.load(infile)
    x = nii.get_fdata()
    with h5py.File(outfile, 'w') as f:
        if dtype is None:
            f.create_dataset("data", data=x, dtype=x.dtype, **hdf5plugin.Blosc())
        else:
            f.create_dataset("data", data=x, dtype=dtype, **hdf5plugin.Blosc())
        f.create_dataset("niftiheader", data = nii.header.structarr)

def h52nii(infile, outfile, niireference):
    with h5py.File(infile, 'r') as f:
            x = f["data"][:]
    nii = nb.load(niireference)
    nii = nb.Nifti1Image(x.astype(np.float32), nii.affine)
    nii.to_filename(outfile)

#niifiles = [f"/mnt/e/data/brainmni/{str(i).zfill(5)}/mni.nii.gz" for i in range(800)]
#batch_nii2h5(niifiles)
def batch_nii2h5(infiles):
    for infile in infiles:
        if not os.path.isfile(infile):
            print(infile + " not found")
            continue
        outfile = infile.split(".nii")[0] + ".h5"
        if os.path.isfile(outfile):
            print(infile + " found")
            continue
        nii2h5(infile, outfile)

#h5files = [f"/home/vsaase/Desktop/brain_train/brainmni/{str(i).zfill(5)}/mni_linear.h5" for i in range(800)]
#compute_onlinestats(h5files, "/home/vsaase/Desktop/brain_train/brainmni/linear")
def compute_onlinestats(h5files, outdir):
    if not os.path.isdir(outdir):
        return False
    count = 0
    for i, filename in enumerate(h5files):
        if not os.path.isfile(filename):
            break
        print("loading" + filename)
        count +=1
        with h5py.File(filename, 'r') as f:
            x = f["data"][:]
        if count == 1:
            mean = np.zeros(x.shape)
            M2 = np.zeros(x.shape)
        print("updating stats")
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

        count += 1
        delta = np.flipud(x) - mean
        mean += delta / count
        delta2 = np.flipud(x) - mean
        M2 += delta * delta2
    std = np.sqrt(M2 / count)
    with h5py.File(f"{outdir}/std.h5", 'w') as f2:
        f2.create_dataset("data", data=std, dtype=np.float16, **hdf5plugin.Blosc())
    
    with h5py.File(f"{outdir}/mean.h5", 'w') as f2:
        f2.create_dataset("data", data=mean, dtype=np.float16, **hdf5plugin.Blosc())
