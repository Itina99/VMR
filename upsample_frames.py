import logging
from rpg_vid2e.upsampling.utils import Upsampler


def upsample():
    output_dir = "output/upsampled_rgb"
    # Upsample ALL RGB sequences from the simulation
    logging.info("📈 Avvio upsampling per tutte le sequenze...")
    upsampler = Upsampler(input_dir="output/cleaned_rgb", output_dir=output_dir)
    upsampler.upsample()



if __name__ == "__main__":
    upsample()