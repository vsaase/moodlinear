from preprocess import  register_MNI_linear, h52nii, MNI_to_world_linear, residual_to_score, residual_to_globalscore
from kernel import compute_residual
import os
import multiprocessing
from functools import partial

def predict_folder_pixel_abs(input_folder, target_folder, sampleonly=False, docker=False):
    files = [(os.path.join(input_folder, f),os.path.join(target_folder, f)) for f in os.listdir(input_folder)]
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())
    for f in files:
        register(f, basedir=target_folder)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    nproc = min(multiprocessing.cpu_count(), len(files))
    pool = multiprocessing.Pool(nproc)
    pool.map(partial(pixelwise, basedir=target_folder, sampleonly=sampleonly, docker=docker), files)
    if docker:
        cmd = (
            f"rm -rf {target_folder}/brainmni"
        )
        #print(cmd)
        os.system(cmd)

def register(files,  basedir = "/mnt/e/data",):
    infile, _ = files
    basename = infile.split("/")[-1].split(".nii")[0]
    outdir = f"{basedir}/brainmni/output/{basename}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    h5mni = f"{outdir}/mni.h5"
    if not os.path.isfile(h5mni):
        register_MNI_linear(infile, outdir)

def pixelwise(files,  basedir = "/mnt/e/data", sampleonly=False, docker=False):
    infile, target_file = files
    h5files = [f"/workspace/brainmni/{str(i).zfill(5)}/mni_linear.h5" for i in range(800)]
    basename = infile.split("/")[-1].split(".nii")[0]
    outdir = f"{basedir}/brainmni/output/{basename}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    h5mni = f"{outdir}/mni_linear.h5"
    if not os.path.isfile(h5mni):
        register_MNI_linear(infile, outdir)
    if not os.path.isfile(f"{outdir}/{basename}.h5"):
        compute_residual(h5mni, h5files, 
            f"/workspace/brainmni/linear/std.h5",
            f"/workspace/brainmni/linear/mean.h5",
            f"{outdir}/{basename}.h5")
    if not os.path.isfile(f"{outdir}/{basename}_score.h5"):
        if sampleonly:
            residual_to_globalscore(f"{outdir}/{basename}.h5", target_file+".txt")
            if docker:
                cmd = (
                    f"rm -rf {outdir}"
                )
                #print(cmd)
                os.system(cmd)
            return
        residual_to_score(f"{outdir}/{basename}.h5", f"{outdir}/{basename}_score.h5")
        h52nii(f"{outdir}/{basename}_score.h5", f"{outdir}/{basename}_score.nii.gz", f"{outdir}/mni_linear.nii.gz")
    MNI_to_world_linear(f"{outdir}/{basename}_score.nii.gz", infile, outdir, target_file)
    
    if docker:
        cmd = (
            f"rm -rf {outdir}"
        )
        #print(cmd)
        os.system(cmd)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-t", type=str, default="brain", help="can be either 'brain' or 'abdom'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir, docker=True)
    elif mode == "sample":
        predict_folder_pixel_abs(input_dir, output_dir, sampleonly=True, docker=True)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")