import logging
from rpg_vid2e.upsampling.utils import Upsampler
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Upsample RGB frames")
    parser.add_argument("--input_dir", default="output/cleaned_rgb", 
                       help="Input directory containing RGB sequences")
    parser.add_argument("--output_dir", default="output/upsampled_rgb",
                       help="Output directory for upsampled frames")
    return parser.parse_args()

def upsample():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    logging.info("📈 Starting upsampling for all sequences...")
    upsampler = Upsampler(input_dir=input_dir, output_dir=output_dir)
    upsampler.upsample()



if __name__ == "__main__":
    upsample()