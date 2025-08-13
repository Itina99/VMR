import logging
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




# Setup logging
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


def generate_sequence(seq_id: int, output_root: Path = Path("output")):

    # --- Parameter settings ---
    kubasic_manifest = "gs://kubric-public/assets/KuBasic/KuBasic.json"
    gso_manifest = "gs://kubric-public/assets/GSO/GSO.json"
    hdri_manifest = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
    resolution = (256, 256)
    frame_end = 24
    frame_rate = 12
    step_rate = 240
    min_static, max_static = 2, 4
    min_dynamic, max_dynamic = 1, 3
    spawn_region_static = [[-7, -7, 0], [7, 7, 10]]
    spawn_region_dynamic = [[-5, -5, 1], [5, 5, 5]]
    VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]
    camera_tipes = ["fixed_random", "linear_movement", "linear_movement_linear_lookat"]
    max_camera_movement = 4.0
    rng = np.random.default_rng()

    # --- Scene initialization ---
    scene = kb.Scene(resolution=resolution)
    scene.frame_end = frame_end
    scene.frame_rate = frame_rate
    scene.step_rate = step_rate

    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)


    # --- Asset sources ---
    asset_source = kb.AssetSource.from_manifest(gso_manifest)
    hdri_source = kb.AssetSource.from_manifest(hdri_manifest)
    kubasic_source = kb.AssetSource.from_manifest(kubasic_manifest)
    
    # --- Scene background HDRI ---
    hdri_id = rng.choice(list(hdri_source._assets.keys()))
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = hdri_source.create(asset_id=hdri_id)
    print(f"📸 Loading HDRI: {background_hdri.filename}")
    renderer._set_ambient_light_hdri(background_hdri.filename)
    #TODO: AGGIUNGERE BACKGROUND AI METADATA

    # --- Dome ---
    dome = kubasic_source.create(asset_id="dome",friction = 1.0, restitution = 0.0, static= True, background= True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)
    
    # --- Camera ---
    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)
    scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
    scene.camera.look_at((0, 0, 0))
    #TODO: AGGIUNGERE MOVIMENTO DI CAMERA


    # === STATIC OBJECTS ===
    num_static = rng.integers(min_static, max_static + 1)
    print(f"📦 Generating {num_static} static objects...")
    shape_ids = sorted(asset_source._assets.keys())  # ✅ alternativa equivalente
    for _ in range(num_static):
        shape_id = rng.choice(shape_ids)
        obj = asset_source.create(
            shape_id
        )
        print(f"📦 Creating static object: {shape_id} at position :{obj.position}")
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1]- obj.bounds[0])  # Normalize scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region_static, rng=rng)
        
        obj.friction = 1.0
        obj.restitution = 0.0
        #TODO AGGIUNGERE STATICO F/T AI METADATA
        print(f"📦 Static objects added: {num_static} ({shape_id}) at position: {obj.position}")

    # --- Run static simulation ---
    simulator.run(frame_start=-100, frame_end=0)
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.friction, obj.restitution = 0.5, 0.5
    
    dome.friction = 0.3
    dome.restitution = 0.5

    
    # === DYNAMIC OBJECTS ===
    num_dynamic = rng.integers(min_dynamic, max_dynamic + 1)
    print(f"🚀 Generating {num_dynamic} dynamic objects...")
    for _ in range(num_dynamic):
        shape_id = rng.choice(shape_ids)
        obj = asset_source.create(
            shape_id)
        print(f"🚀 Creating dynamic object: {shape_id} at position :{obj.position}")
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region_dynamic, rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                  [obj.position[0], obj.position[1], 0])
        #obj.mass = 1.0
        #obj.static = False
        print(f"🚀 Dynamic object {shape_id} velocity set to {obj.velocity} at position: {obj.position}")
        #scene.gravity = (0, 0, -9.81)

    # === Main simulation run ===
    animation,collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)


    # === Saving metadata frame by frame ===
    metadata_output = []
    exclude_names = {"floor", "camera", "sun"}
    movable_objects = [obj for obj in scene.assets if obj.name not in exclude_names]

    print(f"📦 Trovati {len(movable_objects)} oggetti mobili nella scena.")

    for frame_idx in tqdm(range(scene.frame_end + 1), desc=f"📸 Raccoglimento metadati"):
        simulator.run(frame_start=frame_idx, frame_end=frame_idx)
        frame_data = collect_frame_metadata(scene, frame_idx, movable_objects)
        metadata_output.append(frame_data)

    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = annotations_dir / f"seq{seq_id}_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump({"frames": metadata_output}, f, indent=2)

    print(f"✅ Metadata salvati in {metadata_path.resolve()}")


    # === Rendering ===
    renderer.save_state(output_root / f"states/seq{seq_id}.blend")
    frames_dict = renderer.render()

    # === Saving frames ===
    print(f"💾 Salvataggio frame per seq{seq_id} in corso...")
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
        else:
            logging.warning(f"⚠️ Nessuna funzione di salvataggio per '{key}' — ignorato.")
    
    kb.done()

def collect_frame_metadata(scene, frame_idx, movable_objects):
    """
    Raccoglie i metadati per un singolo frame della scena.
    Aggiunge nomi univoci del tipo 'cube_0', 'cube_1', ecc.
    """
    frame_data = {
        "frame": frame_idx,
        "camera": {
            "position": scene.camera.position.tolist(),
            "quaternion": scene.camera.quaternion.tolist()
        },
        "objects": []
    }

    # Contatori per ogni asset_id
    class_counts = defaultdict(int)

    for obj in movable_objects:
        class_name = obj.asset_id  # es. "cube", "sphere"
        count = class_counts[class_name]
        class_counts[class_name] += 1

        unique_name = f"{class_name}_{count}"

        obj_data = {
            "name": unique_name,
            "id": obj.asset_id,
            "position": obj.position.tolist(),
            "quaternion": obj.quaternion.tolist()
        }
        frame_data["objects"].append(obj_data)

    return frame_data


def main(num_sequences: int = 5):
    for seq_id in range(num_sequences):
        print(f"\n🚀 Generazione sequenza {seq_id}")
        generate_sequence(seq_id)
    print("\n✅ Tutte le sequenze sono state generate.")
    


if __name__ == "__main__":
    main(num_sequences=1)