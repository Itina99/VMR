import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import tqdm

import esim_torch


def generate_events(input_dir, output_file, contrast_threshold_neg=0.2, contrast_threshold_pos=0.2, refractory_period_ns=0):
    esim = esim_torch.ESIM(contrast_threshold_neg=contrast_threshold_neg,
                            contrast_threshold_pos=contrast_threshold_pos,
                            refractory_period_ns=refractory_period_ns)

    image_files = sorted(glob.glob(f"{input_dir}/imgs/*.png"))
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
    timestamps_s = np.genfromtxt(f"{input_dir}/timestamps.txt")
    timestamps_ns = (timestamps_s * 1e9).astype("int64")

    log_images = np.log(images.astype("float32") / 255 + 1e-4)

    # generate torch tensors
    device = "cuda:0"
    log_images = torch.from_numpy(log_images).to(device)
    timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

    # generate events with GPU support

    events = esim.forward(log_images, timestamps_ns)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file,
                        x=events['x'].cpu().numpy(),
                        y=events['y'].cpu().numpy(),
                        t=events['t'].cpu().numpy(),
                        p=events['p'].cpu().numpy())

def main():
    upsampled_rgb_dir = "output/upsampled_rgb"
    
    # Get all subdirectories in upsampled_rgb
    subdirs = [d for d in os.listdir(upsampled_rgb_dir) 
               if os.path.isdir(os.path.join(upsampled_rgb_dir, d))]
    
    progress_bar = tqdm.tqdm(subdirs, desc="Processing directories")
    for subdir in progress_bar:
        input_dir = os.path.join(upsampled_rgb_dir, subdir)
        output_file = f"output/events/{subdir}.npz"
        progress_bar.set_description(f"Processing {subdir}")
        generate_events(input_dir, output_file)

if __name__ == "__main__":
    main()