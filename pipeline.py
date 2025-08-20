import subprocess
import os
import shutil
import logging
from rpg_vid2e.upsampling.utils import Upsampler
from event_generator import EventGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_simulation(simulation_type = "gso"):
    user_id = os.getuid()
    group_id = os.getgid()
    current_dir = os.getcwd()

    if (simulation_type == "gso"):
        cmd = [
            "docker", "run", "--rm", "--interactive",
            "--user", f"{user_id}:{group_id}",
            "--volume", f"{current_dir}:/kubric",
            "kubricdockerhub/kubruntu",
        "/usr/bin/python3", "kubric/examples/simulator.py"]
    elif (simulation_type == "shapenet"):
        cmd = [
            "docker", "run", "--rm", "--interactive",
            "--user", f"{user_id}:{group_id}",
            "--volume", f"{current_dir}:/kubric",
            "kubricdockerhub/kubruntu",
            "/usr/bin/python3", "generator_shapenet.py"
        ]

    subprocess.run(cmd)

def pipeline():
    logging.info("🚀 Avvio pipeline completa")
    
    # Start the Kubric simulation to generate initial RGB frames
    logging.info("📹 Avvio simulazione Kubric...")
    start_simulation("shapenet")
    
    # Clean up any existing upsampled output directory
    output_dir = "output/upsampled_rgb"
    if os.path.exists(output_dir):
        logging.info(f"🧹 Pulizia directory esistente: {output_dir}")
        shutil.rmtree(output_dir)

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
    pipeline()
