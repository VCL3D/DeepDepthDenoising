import torch
import cv2
import os
import sys
import argparse
import numpy
from shutil import copyfile

from importers import *
from exporters import *
from evaluation import noise

def parse_arguments(args):
    usage_text = (
        "Synthetic Noise Addition to InteriorNet samples."
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--interiornet_path", type=str, 
        default="..\\InteriorNet\\Scenes", 
        help="Path to the Scenes InteriorNet folder containing the files."
    )
    parser.add_argument("--output_path", type=str, 
        default="..\\InteriorNet\\noisy", 
        help="Path to the output folder where the resulting files will be saved."
    )
    parser.add_argument("--sigma_depth", type=float, default=0.56, 
        help = "Depth standard deviation parameter for disparity noise."
    )
    parser.add_argument("--sigma_space", type=float, default=0.9, 
        help = "Space standard deviation parameter for disparity noise."
    )
    parser.add_argument("--depth_fraction", type=float, default=0.6, 
        help = "Depth fraction parameter for ToF noise."
    )
    parser.add_argument("--copy_original", required=False, default=True,
        action="store_true", help="Save original depth maps as well (prefixed with +gt)."
    )
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    for scene_name in os.listdir(args.interiornet_path):
        scene_folder = os.path.join(args.interiornet_path, scene_name)
        depth_data_folder = os.path.join(scene_folder, "depth0", "data")
        for depth_name in os.listdir(depth_data_folder):
            depth_filename = os.path.join(depth_data_folder, depth_name)
            depth = load_depth(depth_filename)
            noisy_filename =  scene_name + "_noisy_" + depth_name
            algorithm_selector = torch.rand(1).float() < 0.5
            if algorithm_selector:
                sigma_space = args.sigma_space * (1.0 + torch.rand(1).float())
                sigma_depth = args.sigma_depth * (1.0 + torch.rand(1).float())
                noisy, _ = noise.disparity_noise(depth, sigma_depth=sigma_depth,\
                    sigma_space=sigma_space)
            else:
                depth_fraction = args.depth_fraction * (1.0 + torch.rand(1).float())
                noisy = noise.tof_noise(depth, sigma_fraction=depth_fraction)
            output_filename = os.path.join(args.output_path, noisy_filename)
            save_depth(output_filename, noisy, scale=1000.0)            
            if args.copy_original:
                copyfile(depth_filename, output_filename.replace(".png", "+gt.png"))                
