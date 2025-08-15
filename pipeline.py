import subprocess
import os
import shutil
from rpg_vid2e.upsampling.utils import Upsampler
from event_generator import EventGenerator

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
    # Start the Kubric simulation to generate initial RGB frames
    start_simulation("shapenet")
    # Clean up any existing upsampled output directory
    output_dir = "output/upsampled_rgb"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Upsample the RGB frames from the simulation
    upsampler = Upsampler(input_dir="output/rgb", output_dir=output_dir)
    #upsampler.upsample()
    generator = EventGenerator(
        image_dir="output/upsampled_rgb/seq0/imgs",
        timestamp_file="output/upsampled_rgb/seq0/timestamps.txt",
        output_file="output/events/seq0.npz"
    )
    #generator.generate()


if __name__ == "__main__":
    pipeline()
