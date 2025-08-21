import gc
import logging
import os
import numpy as np
import kubric as kb
import bpy
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
from kubric.assets.asset_source import AssetSource
from kubric.file_io import (
    write_rgb_batch,
    write_rgba_batch,
    write_depth_batch,
    write_uv_batch,
    write_normal_batch,
    write_flow_batch,
    write_forward_flow_batch,
    write_backward_flow_batch,
    write_segmentation_batch,
    write_coordinates_batch,
)
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict
import random as Random
import argparse

# ============================================================
# --- CONFIGURAZIONE GLOBALE ---
# ============================================================

logging.basicConfig(level="INFO")

writer_map = {
    "rgb": write_rgb_batch,
    "rgba": write_rgba_batch,
    "depth": write_depth_batch,
    "uv": write_uv_batch,
    "normal": write_normal_batch,
    "flow": write_flow_batch,
    "forward_flow": write_forward_flow_batch,
    "backward_flow": write_backward_flow_batch,
    "segmentation": write_segmentation_batch,
    "object_coordinates": write_coordinates_batch,
}

# Parametri di base
RESOLUTION = (256, 256)
FRAME_END = 24
FRAME_RATE = 12
STEP_RATE = 240
MIN_STATIC, MAX_STATIC = 0, 0
MIN_DYNAMIC, MAX_DYNAMIC = 1, 1
SPAWN_REGION_STATIC = [[-3, -3, 0], [3, 3, 5]]
SPAWN_REGION_DYNAMIC = [[-3, -3, 1], [3, 3, 5]]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]
CAMERA_TYPES = ["fixed_random", "linear_movement", "linear_movement_linear_lookat"]
MAX_CAMERA_MOVEMENT = 4.0

# Percorsi ai manifest
SHAPENET_MANIFEST = "gs://kubric-unlisted/assets/ShapeNetCore.v2.json"
KUBASIC_MANIFEST = "gs://kubric-public/assets/KuBasic/KuBasic.json"
HDRI_MANIFEST = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"

# ============================================================
# --- CARICAMENTO RISORSE UNA VOLTA SOLA ---
# ============================================================

print("📂 Caricamento dataset...")
source_path = os.getenv("SHAPENET_GCP_BUCKET", SHAPENET_MANIFEST)
ASSET_SOURCE = kb.AssetSource.from_manifest(source_path)
HDRI_SOURCE = kb.AssetSource.from_manifest(HDRI_MANIFEST)
KUBASIC_SOURCE = kb.AssetSource.from_manifest(KUBASIC_MANIFEST)


# settings
shape_ids = sorted(ASSET_SOURCE._assets.keys())
classes_all = ["airplane", "ashcan", "bag", "basket", "bathtub", "bed", "bench", "birdhouse", "bookshelf", "bottle", "bowl", "bus", "cabinet", "camera", "can", "cap", "car", "cellular telephone", "chair", "clock", "computer keyboard", "dishwasher", "display", "earphone", "faucet", "file", "guitar", "helmet", "jar", "knife", "lamp", "laptop", "loudspeaker", "mailbox", "microphone", "microwave", "motorcycle", "mug", "piano", "pillow", "pistol", "pot", "printer", "remote control", "rifle", "rocket", "skateboard", "sofa", "stove", "table", "telephone", "tower", "train", "vessel", "washer"]

light_levels_all = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0–100%

light_orientations_all = {
    "front": (0., 0., 0.),
    "side_45": (0., 0., np.pi/4),
    "side_90": (0., 0., np.pi/2),
    "back_135": (0., 0., 3*np.pi/4),
    "top": (np.pi/2, 0., 0.),
    "bottom": (-np.pi/2, 0., 0.)
}

camera_positions_all = {
    "front": (0, -8, 0),            # 0° frontale
    "tilt_30": (4, -7, 3),          # 30° inclinata
    "tilt_60": (7, -4, 5),          # 60° obliqua
    "side_90": (8, 0, 0),           # 90° laterale puro
    "retro_120": (7, 4, 3),         # 120° retro-inclinata
    "back_180": (0, 8, 0),          # 180° dietro
    "top": (0, 0, 8),               # zenitale
    "bottom": (0, 0, -8),           # vista dal basso
}
light_colors_all = {
    "white":   (1.0, 1.0, 1.0, 1.0),
    "red":     (1.0, 0.0, 0.0, 1.0),
    "green":   (0.0, 1.0, 0.0, 1.0),
    "blue":    (0.0, 0.0, 1.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0, 1.0),
    "purple":  (0.5, 0.0, 0.5, 1.0),
    "cyan":    (0.0, 1.0, 1.0, 1.0),
    "orange":  (1.0, 0.5, 0.0, 1.0),
}


print(f"✅ ShapeNet: {len(ASSET_SOURCE._assets)} modelli caricati")
print(f"✅ HDRI: {len(HDRI_SOURCE._assets)} mappe caricate")
print(f"✅ KuBasic asset disponibili")


# ============================================================
# --- PARSING DATI ---
# ============================================================

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Kubric ShapeNet generator")

    parser.add_argument("--light_levels", type=str, default="0.25,1.0",
                        help="Lista intensità luce separate da virgola")
    parser.add_argument("--light_colors", type=str, default="white:1.0,1.0,1.0,1.0",
                        help="Lista colori luce formato name:r,g,b,a separati da ;")
    parser.add_argument("--camera_positions", type=str, default="tilt_60:7,-4,5",
                        help="Lista posizioni camera formato name:x,y,z separati da ;")
    parser.add_argument("--classes", type=str, default="airplane,car",
                        help="Classi ShapeNet da processare")
    parser.add_argument("--light_orientations", type=str, default="side_90:0.0,0.0,1.570796",
                        help="Lista orientazioni luce formato name:x,y,z separati da ;")
    parser.add_argument("--output_root", type=Path, default=Path("output"),
                        help="Cartella di output per i risultati")
    

    return parser.parse_args()


# ============================================================
# --- FUNZIONE DI GENERAZIONE SEQUENZA ---
# ============================================================

def generate_sequence(seq_id: int, shape_id:str, light_intensity: float, orientation: tuple, camera_position: tuple, output_root: Path = Path("output")):
    parser = kb.ArgumentParser()
    parser.set_defaults(
        resolution=RESOLUTION,
        frame_end=FRAME_END,
        frame_rate=FRAME_RATE,
        step_rate=STEP_RATE,
    )
    FLAGS = parser.parse_args()
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)    

    # --- Scene background HDRI ---
    hdri_id = rng.choice(list(HDRI_SOURCE._assets.keys()))
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = HDRI_SOURCE.create(asset_id=hdri_id)
    renderer._set_ambient_light_hdri(background_hdri.filename, hdri_rotation=orientation, strength=light_intensity)

    # --- Dome ---
    dome = KUBASIC_SOURCE.create(asset_id="dome", friction=1.0, restitution=0.0, static=True, background=True)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    # --- Camera ---
    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)
    scene.camera.position = camera_position
    scene.camera.look_at((0, 0, 0))

    

    # === STATIC OBJECTS ===
    num_static = rng.randint(MIN_STATIC, MAX_STATIC + 1)
    print(f"📦 Generating {num_static} static objects...")
    for _ in range(num_static):
        #shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])  # Normalize scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_STATIC, rng=rng)
        print(f"📦 Static object {shape_id} at {obj.position}")

    # === DYNAMIC OBJECTS ===
    num_dynamic = rng.randint(MIN_DYNAMIC, MAX_DYNAMIC + 1)
    print(f"🚀 Generating {num_dynamic} dynamic objects...")
    for _ in range(num_dynamic):
        #shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_DYNAMIC, rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
        print(f"🚀 Dynamic object {shape_id} with velocity {obj.velocity}")

    # === Simulation ===
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)


    # === Rendering ===
    renderer.save_state(output_root / f"states/seq{seq_id}.blend")
    frames_dict = renderer.render()

    # === Post-processing ===
    kb.compute_visibility(frames_dict["segmentation"], scene.assets)
    frames_dict["segmentation"] = kb.adjust_segmentation_idxs(
        frames_dict["segmentation"], scene.assets, [obj]).astype(np.uint8)

    # === Saving frames ===
    print(f"💾 Salvataggio frame per seq{seq_id}...")
    for key in tqdm(frames_dict.keys(), desc=f"Scrittura Frame seq{seq_id}", unit="tipo"):
        value = frames_dict[key]
        base_dir = output_root / key / f"seq{seq_id}"
        imgs_dir = base_dir / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)

        if key == "rgba":
            writer_map["rgba"](value, imgs_dir)
            rgb = value[..., :3]
            rgb_base_dir = output_root / "rgb" / f"seq{seq_id}"
            rgb_imgs_dir = rgb_base_dir / "imgs"
            rgb_imgs_dir.mkdir(parents=True, exist_ok=True)
            writer_map["rgb"](rgb, rgb_imgs_dir)
            with open(rgb_base_dir / "fps.txt", "w") as f:
                f.write(str(scene.frame_rate))

        elif key in writer_map:
            writer_map[key](value, imgs_dir)
            with open(base_dir / "fps.txt", "w") as f:
                f.write(str(scene.frame_rate))

    # === Metadata ===
    exclude_names = {"floor", "camera", "sun"}
    scene_objects = [obj for obj in scene.assets if obj.name not in exclude_names]
    data = {
        "scene_metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "object": kb.get_instance_info(scene, scene_objects)
    }
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = annotations_dir / f"seq{seq_id}_metadata.json"
    kb.file_io.write_json(filename=metadata_path, data=data)
    gc.collect()  # Garbage collection to free memory


# ============================================================
# --- CHOOSE IDS ---
# ============================================================
def chooseClass(class_name):
    return [name for name, spec in ASSET_SOURCE._assets.items() if spec["metadata"]["category"] == class_name]


# ============================================================
# --- MAIN ---
# ============================================================

def main():
    
    args = parse_args()
    
    # Parse light levels
    light_levels = [float(x.strip()) for x in args.light_levels.split(",")]
    
    # Parse light orientations
    light_orientations = {}
    for orient_spec in args.light_orientations.split(";"):
        name, xyz = orient_spec.split(":")
        x, y, z = map(float, xyz.split(","))
        light_orientations[name] = (x, y, z)
    
    # Parse light colors
    light_colors = {}
    for color_spec in args.light_colors.split(";"):
        name, rgba = color_spec.split(":")
        r, g, b, a = map(float, rgba.split(","))
        light_colors[name] = (r, g, b, a)
    
    # Parse camera positions
    camera_positions = {}
    for pos_spec in args.camera_positions.split(";"):
        name, xyz = pos_spec.split(":")
        x, y, z = map(float, xyz.split(","))
        camera_positions[name] = (x, y, z)
    
    # Parse selected classes
    classes = [x.strip() for x in args.classes.split(",")]
    
    # Use output_root from arguments
    output_root = args.output_root
    
    seq_id = 0
    
    for shape_class in classes:
        shape_ids = chooseClass(shape_class)
        shape_id = Random.choice(shape_ids)
        for intensity in light_levels:
            for orient_name, orientation in light_orientations.items():
                for cam_name, cam_pos in camera_positions.items():
                    for color_name, color_value in light_colors.items():
                        print(f"\n🚀 Generazione sequenza {seq_id} | shape={shape_class} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
                        generate_sequence(seq_id, shape_id, intensity, orientation, cam_pos, output_root)
                        seq_id += 1
    print("\n✅ Tutte le sequenze sono state generate.")

    # Generate additional sequences with multiple objects and random parameters
    print(f"\n🎲 Generating additional sequences with random multiple objects...")

    # Modified parameters for multiple objects
    global MIN_STATIC, MAX_STATIC, MIN_DYNAMIC, MAX_DYNAMIC
    MIN_STATIC, MAX_STATIC = 1, 2
    MIN_DYNAMIC, MAX_DYNAMIC = 1, 2
    
    # Generate 10 additional sequences with random parameters
    for i in range(5):
        # Random shape selection
        random_class = Random.choice(classes_all)
        shape_ids = chooseClass(random_class)
        shape_id = Random.choice(shape_ids)
        
        # Random light parameters
        intensity = Random.choice(light_levels_all)
        orient_name, orientation = Random.choice(list(light_orientations_all.items()))
        cam_name, cam_pos = Random.choice(list(camera_positions_all.items()))
        color_name, color_value = Random.choice(list(light_colors_all.items()))

        print(f"\n🎲 Random sequence {seq_id} | shape={random_class} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
        generate_sequence(seq_id, shape_id, intensity, orientation, cam_pos, output_root)
        seq_id += 1

    kb.done()


if __name__ == "__main__":
    main()
