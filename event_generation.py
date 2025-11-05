import gc
import torch
import numpy as np
import glob
import cv2
import os
import esim_torch
import argparse

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate events from a single image sequence")
    
    # This is now the specific input directory (e.g., .../upsampled_rgb/seq_0001)
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input image sequence (imgs/*.png and timestamps.txt)")
    
    # This is now the specific output file path (e.g., .../events/seq_0001.npz)
    parser.add_argument("--output_file", type=str, required=True,
                        help="File path to save the generated events (.npz)")
    
    return parser.parse_args()

def generate_events(input_dir, output_file, contrast_threshold_neg=0.2, contrast_threshold_pos=0.2, refractory_period_ns=0):
    """Generates and saves events for a single image sequence."""
    
    esim = esim_torch.ESIM(contrast_threshold_neg=contrast_threshold_neg,
                           contrast_threshold_pos=contrast_threshold_pos,
                           refractory_period_ns=refractory_period_ns)

    image_files = sorted(glob.glob(f"{input_dir}/imgs/*.png"))
    
    # Check if images were found
    if not image_files:
        print(f"⚠️  Warning: No images found in {input_dir}. Skipping.")
        return
        
    timestamps_file = f"{input_dir}/timestamps.txt"
    if not os.path.exists(timestamps_file):
        print(f"⚠️  Warning: No timestamps.txt found in {input_dir}. Skipping.")
        return

    try:
        images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
        timestamps_s = np.genfromtxt(timestamps_file)
        timestamps_ns = (timestamps_s * 1e9).astype("int64")

        log_images = np.log(images.astype("float32") / 255 + 1e-5)

        # generate torch tensors - esim_torch requires CUDA tensors
        device = "cuda:0" 
        log_images = torch.from_numpy(log_images).to(device)
        timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

        # generate events with GPU support
        events = esim.forward(log_images, timestamps_ns)
        
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        np.savez_compressed(output_file,
                            x=events['x'].cpu().numpy(),
                            y=events['y'].cpu().numpy(),
                            t=events['t'].cpu().numpy(),
                            p=events['p'].cpu().numpy())

    except Exception as e:
        print(f"❌ Error processing {input_dir}: {e}")
    
    finally:
        # Explicitly delete tensors and clear cache to be extra safe
        if 'events' in locals(): del events
        if 'log_images' in locals(): del log_images
        if 'timestamps_ns' in locals(): del timestamps_ns
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main function to parse args and call event generator."""
    args = parse_args()
    
    # The loop is removed. This script now only processes one directory.
    print(f"Processing {args.input_dir} -> {args.output_file}")
    generate_events(args.input_dir, args.output_file)
    print(f"✅ Completed processing for {args.input_dir}")

if __name__ == "__main__":
    main()