import torch
import numpy as np
import glob
import cv2
import os
import tqdm
import esim_torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate events from image sequences")
    parser.add_argument("--input_dir", type=str, default="output/upsampled_rgb",
                       help="Directory containing input image sequences")
    parser.add_argument("--output_dir", type=str, default="output/events",
                       help="Directory to save generated events")
    return parser.parse_args()

def generate_events(input_dir, output_file, contrast_threshold_neg=0.2, contrast_threshold_pos=0.2, refractory_period_ns=0):     
    esim = esim_torch.ESIM(contrast_threshold_neg=contrast_threshold_neg,
                            contrast_threshold_pos=contrast_threshold_pos,
                            refractory_period_ns=refractory_period_ns)

    image_files = sorted(glob.glob(f"{input_dir}/imgs/*.png"))
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
    timestamps_s = np.genfromtxt(f"{input_dir}/timestamps.txt")
    timestamps_ns = (timestamps_s * 1e9).astype("int64")

    log_images = np.log(images.astype("float32") / 255 + 1e-5)

    # generate torch tensors
    # generate torch tensors - esim_torch requires CUDA tensors
    
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
    args = parse_args()
    upsampled_rgb_dir = args.input_dir  
    output_dir = args.output_dir
    
    # Get all subdirectories in upsampled_rgb
    subdirs = [d for d in os.listdir(upsampled_rgb_dir) 
               if os.path.isdir(os.path.join(upsampled_rgb_dir, d))]
    
    progress_bar = tqdm.tqdm(subdirs, desc="Processing directories")
    for subdir in progress_bar:
        input_dir = os.path.join(upsampled_rgb_dir, subdir)
        output_file = os.path.join(output_dir, f"{subdir}.npz")
        progress_bar.set_description(f"Processing upsampled sequences")
        generate_events(input_dir, output_file)

if __name__ == "__main__":
    main()