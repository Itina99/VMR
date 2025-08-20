import subprocess
import os
import shutil
import logging
from rpg_vid2e.upsampling.utils import Upsampler
from event_generator import EventGenerator


def generation():

    output_dir = "output/upsampled_rgb"
    # Upsample ALL RGB sequences from the simulation
    logging.info("📈 Avvio upsampling per tutte le sequenze...")
    upsampler = Upsampler(input_dir="output/rgb", output_dir=output_dir)
    upsampler.upsample()
    
    # Generate events for ALL upsampled sequences
    logging.info("⚡ Generazione eventi per tutte le sequenze...")
    generator = EventGenerator(
        base_dir="output/upsampled_rgb",
        output_base_dir="output/events"
    )
    successful, failed = generator.generate_all()
    
    logging.info(f"✅ Pipeline completata: {successful} sequenze elaborate con successo, {failed} fallimenti")



if __name__ == "__main__":
    generation()