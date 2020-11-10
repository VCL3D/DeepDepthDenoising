import argparse
import os
import sys
import argparse
import torch
import models
import utils
import dataset
import importers

from supervision import *
from exporters import *
from importers import *

import datetime

def parse_arguments(args):
    usage_text = (
        "Depth Denoising method predictions."
        "Usage:  python inference.py [options],"
        "   with [options]: (as described below)"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--model_path", type=str , help="Path to saved model to load.")
    parser.add_argument("--input_path", type=str, help="Path to files for inference.")  
    parser.add_argument("--output_path", type=str, help="Path to directory to save the infered files.")
    parser.add_argument("--pointclouds", type=bool, default=False, help = "Save original and denoised pointclouds for RealSense input.")
    parser.add_argument("--autoencoder", type=bool, default=False, help = "Set model to autoencoder mode (i.e. trained without multi-view supervision, but as a depth map autoencoder).")
    parser.add_argument("-g","--gpu", type=str, default="0", help="The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.")
    # other
    parser.add_argument("--scale", type=float, default="0.001", help="How much meters does one bit represent in the input data.")
    return parser.parse_known_args(args)

def run_model(
    model_path          : str, #path to trained model
    input_path          : str, #path containing data to be denoised
    output_path         : str,
    device              : str, #device on which the network will run
    scale               : float
):
    assert os.path.exists(input_path), "{} does not exist\n".format(input_path)
    assert os.path.exists(model_path), "{} does not exist\n".format(model_path)

    if not os.path.exists(output_path):
        print("{} does not exist,creating\n".format(input_path))
        os.makedirs(output_path)

    ndf = 16 if args.autoencoder else 8
    model_params = {
        'width': 640,
        'height': 360,
        'ndf': ndf,
        'dilation': 1,
        'norm_type': "elu",
        'upsample_type': "nearest"
    }

    model = models.get_model(model_params).to(device)

    utils.init.initialize_weights(model, model_path)

    files = [os.path.join(input_path,file) for file in os.listdir(input_path)]
    print("{} files loaded".format(len(files)))

    uv_grid_t = create_image_domain_grid(model_params['width'], model_params['height'])

    if args.pointclouds:
        device_repo_path = os.path.join(args.input_path,"device_repository.json")
        device_repository = importers.intrinsics.load_intrinsics_repository(device_repo_path)
    
    for file in files:
        filename, extension = os.path.basename(file).split('.')
        if extension == "json":
            continue
        depthmap = load_depth(
            filename = file,
            scale = scale
        )
        if depthmap.shape[3] != model_params['width'] or depthmap.shape[2] != model_params['height']:
            depthmap = crop_depth(# for inference /w InteriorNet (640x480), center cropped to 640x360
                filename = file,
                scale = scale
            )

        mask, _ = get_mask(depthmap)
        
        mask, depthmap = mask.to(device), depthmap.to(device)

        predicted_depth, _ = model(depthmap, mask)

        masked_predicted_depth = predicted_depth * mask

        # save denoising depthmap
        output_file = os.path.join(output_path, filename + "_denoised." + extension)
        save_depth(output_file, masked_predicted_depth, 1/scale)
        print("{} denoised depthmap saved.".format(filename))

        # save original (noisy) and denoised depthmaps as pointclouds
        if args.pointclouds:
            device_name = filename.split('_')[1]
            _, intrinsics_inv = importers.intrinsics.get_intrinsics(\
                    device_name, device_repository, scale=2)

            source_points3d = deproject_depth_to_points(depthmap.cpu(), uv_grid_t, intrinsics_inv)
            save_ply(os.path.join(args.output_path, filename + "_original_#.ply"), source_points3d, 1000.0, color='red')

            masked_predicted_points3d = deproject_depth_to_points(masked_predicted_depth.cpu(), uv_grid_t, intrinsics_inv)
            save_ply(os.path.join(args.output_path, filename + "_masked_denoised_#.ply"), masked_predicted_points3d, 1000.0, color='blue')
            print("{} pointclouds saved.".format(filename))

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device("cuda:{}" .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu")

    run_model(
        args.model_path,
        args.input_path,
        args.output_path,
        device,
        args.scale
    )