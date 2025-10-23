import gc
import logging
import os
import re
import numpy as np
import kubric as kb
import bpy
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
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
from kubric import utils as kb_utils
from tqdm import tqdm
from pathlib import Path
import random as Random
from HDRISelector import HDRISelector
from datetime import datetime
import time
import math
import hashlib



# ============================================================
# --- GLOBAL CONFIGURATION ---
# ============================================================

logging.basicConfig(level="WARNING")
os.environ["KUBRIC_USE_GPU"] = "1"

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

CAMERA_TYPES = ["fixed", "linear_movement", "panning"]

# Percorsi ai manifest
SHAPENET_MANIFEST = "gs://kubric-unlisted/assets/ShapeNetCore.v2.json"
KUBASIC_MANIFEST = "gs://kubric-public/assets/KuBasic/KuBasic.json"
HDRI_MANIFEST = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"

# ============================================================
# --- CARICAMENTO RISORSE UNA VOLTA SOLA ---
# ============================================================

print("📂 Loading dataset...")
source_path = os.getenv("SHAPENET_GCP_BUCKET", SHAPENET_MANIFEST)
ASSET_SOURCE = kb.AssetSource.from_manifest(source_path)
HDRI_SOURCE = kb.AssetSource.from_manifest(HDRI_MANIFEST)
KUBASIC_SOURCE = kb.AssetSource.from_manifest(KUBASIC_MANIFEST)
selector = HDRISelector(source=HDRI_SOURCE, json_path="hdri.json")


# settings
SPAWNING_REGION_STATIC= [[-5, -5, 0], [5, 5, 0]]
SPAWNING_REGION_DYNAMIC= [[-5, -5, 1], [5, 5, 6]]
VELOCITY_RANGE= [[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0]]
shape_ids = sorted(ASSET_SOURCE._assets.keys())
classes_all = ["airplane", "ashcan", "bag", "basket", "bathtub", "bed", "bench", "birdhouse", "bookshelf", "bottle", "bowl", "bus", "cabinet", "camera", "can", "cap", "car", "cellular telephone", "chair", "clock", "computer keyboard", "dishwasher", "display", "earphone", "faucet", "file", "guitar", "helmet", "jar", "knife", "lamp", "laptop", "loudspeaker", "mailbox", "microphone", "microwave", "motorcycle", "mug", "piano", "pillow", "pistol", "pot", "printer", "remote control", "rifle", "rocket", "skateboard", "sofa", "stove", "table", "telephone", "tower", "train", "vessel", "washer"]

light_levels_all = [0.25, 0.5, 0.75, 1.0]  

light_orientations_all = {
    "side_45": (0., 0., np.pi/4),
    "side_90": (0., 0., np.pi/2),
    "back_135": (0., 0., 3*np.pi/4),
    "top": (np.pi/2, 0., 0.),}

camera_positions_all = {
    "tilt_30": (4, -7, 3),          
    "tilt_60": (7, -4, 5),
    "retro_120": (7, 4, 3),
    "top": (0, 0, 8),

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


print(f"✅ ShapeNet: {len(ASSET_SOURCE._assets)} models loaded")
print(f"✅ HDRI: {len(HDRI_SOURCE._assets)} maps loaded")
print(f"✅ KuBasic assets available")


# ============================================================
# --- LIGHT DIRECTION AND COLOR SELECTION---
# ============================================================

def get_light_direction(luminosity: float, rng: np.random.RandomState, distance: float = 10.0) -> np.ndarray:
    """
    Calculate a random position for the light on a hemisphere,
    keeping the height linked to the luminosity.

    Args:
        luminosity (float): The luminosity level [0.0, 1.0], which controls the height.
        rng (np.random.RandomState): The random number generator for reproducibility.
        distance (float): The distance of the light from the center of the scene.

    Returns:
        np.ndarray: A 3D position vector for the directional light.
    """
    luminosity = np.clip(luminosity, 0.0, 1.0)

    elevation_rad = luminosity * (math.pi / 2.0)

    azimuth_rad = rng.uniform(0, 2 * math.pi)
    

    xy_projection = distance * math.cos(elevation_rad)
    
    x = xy_projection * math.cos(azimuth_rad)
    y = xy_projection * math.sin(azimuth_rad)
    z = distance * math.sin(elevation_rad)

    if z < 0.1:
        z = 0.1
        
    return np.array([x, y, z])

def get_light_color_by_intensity(light_intensity: float) -> tuple:
    """
    Choose a light color based on intensity to mimic time of day progression.
    
    Args:
        light_intensity (float): Light intensity value between 0.0 and 1.0
        
    Returns:
        tuple: RGBA color tuple representing the time of day
    """
    intensity = np.clip(light_intensity, 0.0, 1.0)
    
    if intensity >= 0.9:  
        return (1.0, 1.0, 1.0, 1.0)
    elif intensity >= 0.7:  
        return (1.0, 0.95, 0.85, 1.0)
    elif intensity >= 0.5:  
        return (1.0, 0.9, 0.6, 1.0)
    elif intensity >= 0.25:  
        return (1.0, 0.7, 0.3, 1.0)
    elif intensity >= 0.1:  
        return (1.0, 0.5, 0.2, 1.0)
    else:  
        return (0.3, 0.4, 0.8, 1.0)

# ============================================================
# --- CAMERA MOVEMENT FUNCTIONS---
# ============================================================

def get_linear_camera_motion_start_end(movement_speed: float, inner_radius: float = 7., outer_radius: float = 9., z_offset: float = 0.1):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        camera_start = np.array(kb.sample_point_in_half_sphere_shell(
            inner_radius, outer_radius, z_offset))
        direction = Random.random() * 3 - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z_offset):
            return camera_start, camera_end

def get_linear_lookat_motion_start_end(inner_radius: float = 0.5, outer_radius: float = 2.0):
    """Sample a linear path for the look-at point."""
    while True:
        camera_through = np.array(
            kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0))
        while True:
            camera_start = np.array(
                kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0))
            if camera_start[-1] < inner_radius:
                break
        continuation = Random.random() * 0.5
        camera_end = camera_through + continuation * (camera_through - camera_start)
        if Random.random() < 0.5:
            camera_start, camera_end = camera_end, camera_start
        return camera_start, camera_end

# ============================================================
# --- PARSING DATA ---
# ============================================================

def parse_args():
    parser = kb.ArgumentParser()
    """     parser.set_defaults(
            resolution=RESOLUTION,
            frame_end=FRAME_END,
            frame_rate=FRAME_RATE,
            step_rate=STEP_RATE,
        ) """

    parser.add_argument("--refresh_item_number",type=int, default=5)
    parser.add_argument("--light_levels", nargs="+", type=float, default=[1.0])
    parser.add_argument("--camera_modes", nargs="+", choices=["fixed", "linear_movement", "panning"], default=["fixed"])
    parser.add_argument("--camera_positions", nargs="+", default=["tilt_30"])
    parser.add_argument("--light_colors", nargs="+", default=["white"])
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--standard_mode", type=lambda x: x.lower() == 'true', default=True, help="Generate standard sequences with one object per scene and fixed parameters")
    parser.add_argument("--rand_gen", type=lambda x: x.lower() == 'true', default=False, help="Generate additional sequences with random parameters and multiple objects")
    parser.add_argument("--number_of_random_sequences", type=int, default=5, help="Number of random sequences to generate if --rand_gen is active")

    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--max_camera_movement", type=float, default=4.0)
    parser.add_argument("--max_dynamic_objects", type=int, default=3)
    parser.add_argument("--max_static_objects", type=int, default=3)
    parser.add_argument("--min_dynamic_objects", type=int, default=1)
    parser.add_argument("--min_static_objects", type=int, default=1)

    return parser.parse_args()

# ============================================================
# --- COCO STYLE ANNOTATIONS ---
# ============================================================

def initialize_coco_dict():
    """Create the basic, empty structure of a COCO annotations file."""
    return {
        "info": { "year": datetime.now().year, "version": "1.0", "description": "Simulated dataset with Kubric sequences featuring objects from the ShapeNet dataset and variable lighting conditions", "contributor": "Matteo Tinacci" },
        "licenses": [], "categories": [], "images": [], "annotations": []
    }

def update_coco_from_metadata(coco_dict, kubric_metadata, seq_name, annotation_id, image_id):
    """
    Uses the REAL data structure (a list of objects with "bbox" key)
    to update the COCO dictionary.
    """
    # --- 1. Category Management ---

    objects_metadata_list = kubric_metadata["object"]
    existing_categories = {cat['name']: cat['id'] for cat in coco_dict['categories']}

    for obj_info in objects_metadata_list:
        category_name = obj_info["asset_id"]
        if category_name not in existing_categories:
            category_id = len(existing_categories) + 1
            coco_dict['categories'].append({
                "id": category_id,
                "name": category_name,
                "supercategory": "oggetto_simulato"
            })
            existing_categories[category_name] = category_id

    # --- 2. Image and Annotation Management ---
    scene_info = kubric_metadata["scene_metadata"]
    num_frames = scene_info["num_frames"]
    height, width = scene_info["resolution"]
    
    print(f"📊 Processing {len(objects_metadata_list)} objects for {num_frames} frames")
    for i, obj_info in enumerate(objects_metadata_list):
        vis_len = len(obj_info["visibility"]) if "visibility" in obj_info else 0
        bbox_len = len(obj_info["bboxes"]) if "bboxes" in obj_info else 0
        print(f"  Object {i}: visibility={vis_len}, bboxes={bbox_len} frames")
    
    for frame_idx in range(num_frames):
        current_image_id = image_id + frame_idx
        image_info = {
            "id": current_image_id,
            "file_name": f"{seq_name}/imgs/{frame_idx:05d}.png", 
            "width": width,
            "height": height
        }
        coco_dict['images'].append(image_info)
        
        for obj_info in objects_metadata_list:
            if ("visibility" not in obj_info or 
                len(obj_info["visibility"]) == 0 or
                all(vis == 0 for vis in obj_info["visibility"])):
                annotation_info = {
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": existing_categories[obj_info["asset_id"]],
                    "bbox": [0.0, 0.0, 0.0, 0.0],  
                    "area": 0.0,
                    "iscrowd": 0,
                    "segmentation": [],
                    "visibility": "not_visible"  
                }
                coco_dict['annotations'].append(annotation_info)
                annotation_id += 1
                continue
                

            if (frame_idx < len(obj_info["visibility"]) and 
                obj_info["visibility"][frame_idx] > 0):
                
                if ("bboxes" in obj_info and 
                    frame_idx < len(obj_info["bboxes"])):
                    bbox = obj_info["bboxes"][frame_idx]
                    ymin, xmin, ymax, xmax = bbox
                    coco_bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                    area = float((xmax - xmin) * (ymax - ymin))

                    annotation_info = {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": existing_categories[obj_info["asset_id"]],
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    coco_dict['annotations'].append(annotation_info)
                    annotation_id += 1
                else:
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": existing_categories[obj_info["asset_id"]],
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "area": 0.0,
                        "iscrowd": 0,
                        "segmentation": [],
                        "visibility": "visible_no_bbox"  
                    }
                    coco_dict['annotations'].append(annotation_info)
                    annotation_id += 1
            elif frame_idx < len(obj_info["visibility"]):
                annotation_info = {
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": existing_categories[obj_info["asset_id"]],
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "area": 0.0,
                    "iscrowd": 0,
                    "segmentation": [],
                    "visibility": "not_visible_frame"  
                }
                coco_dict['annotations'].append(annotation_info)
                annotation_id += 1
            
    next_image_id = image_id + num_frames
    
    return coco_dict, annotation_id, next_image_id

# ============================================================
# --- UTILS ---
# ============================================================

def chooseClass(class_name):
    return [name for name, spec in ASSET_SOURCE._assets.items() if spec["metadata"]["category"] == class_name]

def get_seed():
    """Return a random seed for generation."""
    return int(time.time() * 1000) % 2**32

def get_sequence_name():
    """Generate a unique sequence name using timestamp and random hash."""
    
    timestamp = int(time.time() * 1000)
    random_bytes = os.urandom(4)  
    hash_component = hashlib.md5(random_bytes).hexdigest()[:6]
    
    return f"{timestamp}_{hash_component}"
# ============================================================
# --- SCENE GENERATION ---
# ============================================================

def compute_bboxes_manually(segmentation_map, asset):
    """
    Manually calculate bounding boxes for an asset
    from the segmentation map.
    
    Args:
        segmentation_map (np.ndarray): The segmentation array (num_frames, H, W, 1).
        asset (kb.Asset): The asset for which to calculate the BBoxes.

    Returns:
        np.ndarray: An array (num_frames, 4) of normalized BBoxes [ymin, xmin, ymax, xmax].
    """
    seg_id = asset.segmentation_id
    
    if segmentation_map.ndim == 4:

        if segmentation_map.shape[3] == 1:
            segmentation_map = segmentation_map.squeeze(axis=3)
        else:
            raise ValueError(f"Segmentation map has unexpected shape: {segmentation_map.shape}")
            
    num_frames, height, width = segmentation_map.shape
    
    all_bboxes = np.zeros((num_frames, 4), dtype=np.float32)

    for frame_idx in range(num_frames):
        rows, cols = np.where(segmentation_map[frame_idx] == seg_id)
        
        if rows.size > 0:
            ymin = np.min(rows)
            xmin = np.min(cols)
            ymax = np.max(rows) + 1 
            xmax = np.max(cols) + 1
            
            all_bboxes[frame_idx] = [
                float(ymin) / height,
                float(xmin) / width,
                float(ymax) / height,
                float(xmax) / width
            ]

    return all_bboxes

def generate_scene_layout(FLAGS):
    """
    Uses a seed to generate a list of objects with their 
    "spawn" properties (initial position, scale, etc.). 
    Does NOT execute the settling simulation.
    """
    print(f"🔩 Generating spawn list with seed {FLAGS.seed}...")
    rng = np.random.RandomState(FLAGS.seed)

    temp_scene, _, _, _ = kb.setup(FLAGS)
    simulator = KubricSimulator(temp_scene)
    
    dome = KUBASIC_SOURCE.create(asset_id="dome", name="dome", static=True, background=True)
    temp_scene += dome
    
    layout_data = []
    current_segmentation_id = 1

    # === 1. Initial Static Objects Placement ===
    num_static = rng.randint(FLAGS.min_static_objects, FLAGS.max_static_objects + 1)
    print(f"  📦 Generating {num_static} static objects...")
    
    for idx in range(num_static):
        oggetto_posizionato_con_successo = False
        max_retries = 5  

        for attempt in range(max_retries):
            random_class = rng.choice(classes_all)
            shape_ids = chooseClass(random_class)
            shape_id = rng.choice(shape_ids)
            
            try:
                obj = ASSET_SOURCE.create(shape_id)

            except kb.core.traitlets.TraitError as e:

                print(f"    MASS WARNING: Attempt {attempt + 1}/{max_retries} failed. Object {shape_id} has a problematic mesh. Retrying with another object.")
                continue  
            scale = rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            
            temp_scene += obj
            
            try:
                kb.move_until_no_overlap(
                    obj, 
                    simulator, 
                    spawn_region=SPAWNING_REGION_STATIC, 
                    max_trials=200,
                    rng=rng
                )
                
                layout_data.append({
                    "asset_id": obj.asset_id, 
                    "segmentation_id": current_segmentation_id,
                    "position": tuple(obj.position), 
                    "quaternion": tuple(obj.quaternion),
                    "scale": tuple(obj.scale), 
                    "static": True
                })
                current_segmentation_id += 1
                oggetto_posizionato_con_successo = True
                break 

            except RuntimeError:
                print(f"    PLACEMENT WARNING: Attempt {attempt + 1}/{max_retries} failed for object {shape_id}.")
                temp_scene.remove(obj)
        
        if not oggetto_posizionato_con_successo:
            print(f"    FINAL WARNING: Unable to place a static object in slot #{idx+1} after {max_retries} attempts.")

    # === 2. Dynamic Objects Placement with Retry Logic ===

    num_dynamic = rng.randint(FLAGS.min_dynamic_objects, FLAGS.max_dynamic_objects + 1)
    print(f"  🚀 Generating up to {num_dynamic} dynamic objects...")
    
    for idx in range(num_dynamic):
        oggetto_posizionato_con_successo = False
        max_retries = 5

        for attempt in range(max_retries):
            random_class = rng.choice(classes_all)
            shape_ids = chooseClass(random_class)
            shape_id = rng.choice(shape_ids)
            try:
                obj = ASSET_SOURCE.create(shape_id)
            except kb.core.traitlets.TraitError as e:
                print(f"    MASS WARNING: Attempt {attempt + 1}/{max_retries} failed. Object {shape_id} has a problematic mesh. Retrying with another object.")
                continue 
            scale = rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            
            temp_scene += obj
            try:
                kb.move_until_no_overlap(
                    obj,
                    simulator,
                    spawn_region=SPAWNING_REGION_DYNAMIC,
                    max_trials=200,
                    rng=rng
                )

                velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
                layout_data.append({
                    "asset_id": obj.asset_id,
                    "segmentation_id": current_segmentation_id,
                    "position": tuple(obj.position),
                    "quaternion": tuple(obj.quaternion),
                    "scale": tuple(obj.scale),
                    "velocity": tuple(velocity),
                    "angular_velocity": (0., 0., 0.),
                    "static": False
                })
                current_segmentation_id += 1
                oggetto_posizionato_con_successo = True
                break

            except RuntimeError:
                print(f"    PLACEMENT WARNING: Attempt {attempt + 1}/{max_retries} failed for dynamic object {shape_id}.")
                temp_scene.remove(obj)

        if not oggetto_posizionato_con_successo:
            print(f"    FINAL WARNING: Unable to place a dynamic object in slot #{idx+1} after {max_retries} attempts.")
            
    print(f"  -> Spawn list for {len(layout_data)} objects created successfully.")
    del temp_scene, simulator
    gc.collect()
    return layout_data

def render_variation(layout_data: list, light_intensity: float, light_color: tuple, light_orientation: tuple, camera_position: tuple, camera_mode:str, FLAGS, output_root: Path = Path("output")):
    """Creates a scene, populates it, sets variation parameters (camera, lights) and includes complete logic for dome, HDRI, rendering and saving."""
    # 1. INITIAL SETUP
    clean_blender_scene()

    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)


    # 2. SCENE SETUP (LIGHTS, BACKGROUND, CAMERA)
    hdri_id = selector.pick(light_intensity)
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = HDRI_SOURCE.create(asset_id=hdri_id)
    scene.metadata["background"] = hdri_id

    logging.info(f"Using light intensity: {light_intensity:.2f}")
    logging.info(f"Using light color: {light_color}")

    # --- LIGHT CALIBRATION ---
    LIGHT_SOURCE_GAMMA = 1.8
    light_source_intensity = light_intensity ** LIGHT_SOURCE_GAMMA
    BACKGROUND_GAMMA = 1.8
    background_visual_intensity = light_intensity ** BACKGROUND_GAMMA
    AMBIENT_LIGHT_FACTOR = 0.1
    print(f"INFO: Light intensity: {light_source_intensity:.4f} | Background intensity: {background_visual_intensity:.4f} | Ambient light: {AMBIENT_LIGHT_FACTOR}")

    # --- Ambient Light of the Blender World ---
    light_color = get_light_color_by_intensity(light_intensity) if light_color is None else light_color
    world_background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_background_node:
        # Usiamo direttamente la variabile light_color, che dovrebbe essere un RGBA
        world_background_node.inputs["Color"].default_value = light_color
        world_background_node.inputs["Strength"].default_value = AMBIENT_LIGHT_FACTOR * light_intensity
        print("INFO: Ambient Light of the World configured with custom color.")

    # --- Adding a Sun ---
    SUN_BASE_INTENSITY = 0.25
    sun_pos = get_light_direction(light_intensity, rng) if light_orientation is None else light_orientation

    sun = kb.DirectionalLight(
        name="sun",
        position=sun_pos,
        look_at=(0, 0, 0),
        color=light_color[:3],
        intensity=SUN_BASE_INTENSITY * light_source_intensity
    )
    scene.add(sun)
    print("INFO: Sun Light configured with custom color.")

    # --- Dome ---
    dome = KUBASIC_SOURCE.create(asset_id="dome", name="dome", static=True, background=True)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node_ref = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node_ref.image = bpy.data.images.load(background_hdri.filename)
    material = dome_blender.data.materials[0]
    node_tree = material.node_tree
    texture_node = node_tree.nodes.get("Image Texture")

    if texture_node:
        if texture_node.outputs["Color"].links:
            original_destinations = []
            for link in texture_node.outputs["Color"].links:
                original_destinations.append(link.to_socket)

            for link in list(texture_node.outputs["Color"].links):
                node_tree.links.remove(link)

            color_dimmer_node = node_tree.nodes.new(type='ShaderNodeVectorMath')
            color_dimmer_node.operation = 'MULTIPLY'
            bg_vec = [c * background_visual_intensity for c in light_color[:3]]
            color_dimmer_node.inputs[1].default_value = bg_vec

            node_tree.links.new(texture_node.outputs["Color"], color_dimmer_node.inputs[0])

            for dest_socket in original_destinations:
                node_tree.links.new(node_tree.nodes["Vector Math"].outputs["Vector"], dest_socket)

    # --- Camera ---
    max_camera_speed = FLAGS.max_camera_movement
    print(f"🎥 Setting up Camera in '{camera_mode}' mode...")

    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)

    if camera_mode == "fixed":
        scene.camera.position = camera_position
        scene.camera.look_at((0, 0, 0))
        logging.info(f"Camera position fixed at {scene.camera.position}")

    elif camera_mode == "linear_movement":
        speed = rng.uniform(0., max_camera_speed)
        camera_start, camera_end = get_linear_camera_motion_start_end(movement_speed=speed)
        logging.info(f"Camera will move from {camera_start} to {camera_end} with speed {speed:.2f}")

        for frame in range(scene.frame_start, scene.frame_end + 1):
            interp = (frame - scene.frame_start) / (scene.frame_end - scene.frame_start)
            scene.camera.position = (1 - interp) * camera_start + interp * camera_end
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    elif camera_mode == "panning":
        speed = rng.uniform(0., max_camera_speed)
        camera_start, camera_end = get_linear_camera_motion_start_end(movement_speed=speed)
        lookat_start, lookat_end = get_linear_lookat_motion_start_end()
        logging.info(f"Camera will move from {camera_start} to {camera_end} with speed {speed:.2f}")
        logging.info(f"Camera will pan from {lookat_start} to {lookat_end}")

        for frame in range(scene.frame_start, scene.frame_end + 1):
            interp = (frame - scene.frame_start) / (scene.frame_end - scene.frame_start)
            scene.camera.position = (1 - interp) * camera_start + interp * camera_end
            scene.camera.look_at((1 - interp) * lookat_start + interp * lookat_end)
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)
    
    # --- PHASE 1: Population and Settling of Static Objects ---
    print("INFO: Phase 1 - Population and settling of static objects...")
    
    dynamic_objects_data = []
    static_asset_references = []

    for obj_data in layout_data:
        if obj_data["static"]:
            obj = ASSET_SOURCE.create(asset_id=obj_data["asset_id"])
            obj.segmentation_id = obj_data["segmentation_id"]
            obj.position = obj_data["position"]
            obj.quaternion = obj_data["quaternion"]
            obj.scale = obj_data["scale"]
            scene += obj
            static_asset_references.append(obj)
        else:
            dynamic_objects_data.append(obj_data)

    if static_asset_references:
        print(f"INFO: Executing settling simulation for {len(static_asset_references)} static objects...")
        simulator.run(frame_start=-100, frame_end=0)

        for obj in static_asset_references:
            obj.velocity = (0., 0., 0.)
            obj.angular_velocity = (0., 0., 0.)

    # --- PHASE 2: Population of Dynamic Objects ---
    print(f"INFO: Phase 2 - Population of {len(dynamic_objects_data)} dynamic objects...")
    for obj_data in dynamic_objects_data:
        obj = ASSET_SOURCE.create(asset_id=obj_data["asset_id"])
        obj.segmentation_id = obj_data["segmentation_id"]
        obj.position = obj_data["position"]
        obj.quaternion = obj_data["quaternion"]
        obj.scale = obj_data["scale"]
        obj.velocity = obj_data["velocity"]
        obj.angular_velocity = obj_data["angular_velocity"]
        scene += obj

    # --- PHASE 3: Final Simulation and Rendering ---
    print("🎬 Final simulation...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    print("🎥 Rendering...")
    frames_dict = renderer.render()

    # 5. POST-PROCESSING AND SAVING (Manual Calculation Version)
    kb.compute_visibility(frames_dict["segmentation"], scene.assets)


    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]

    print("INFO: Adjusting contiguous segmentation IDs...")
    frames_dict["segmentation"] = kb.adjust_segmentation_idxs(
        frames_dict["segmentation"],
        scene.assets,
        visible_foreground_assets
    ).astype(np.uint8)
    print("INFO: Manual Bounding Box calculation...")

    segmentation_map = frames_dict["segmentation"]
    
    for asset in visible_foreground_assets:
        bboxes_array = compute_bboxes_manually(segmentation_map, asset)
        
        asset.metadata["bboxes"] = bboxes_array

    for asset in visible_foreground_assets:
        if "bbox_frames" in asset.metadata:
            del asset.metadata["bbox_frames"]
        

    # Saving frame
    seq_name = get_sequence_name()
    print(f"💾 Saving frame for {seq_name}...")
    for key in tqdm(frames_dict.keys(), desc=f"Writing Frame {seq_name}", unit="type"):
        value = frames_dict[key]
        base_dir = output_root / key / seq_name
        imgs_dir = base_dir / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)

        if key == "rgba":
            writer_map["rgba"](value, imgs_dir)
            rgb = value[..., :3]
            rgb_base_dir = output_root / "rgb" / seq_name
            rgb_imgs_dir = rgb_base_dir / "imgs"
            rgb_imgs_dir.mkdir(parents=True, exist_ok=True)
            writer_map["rgb"](rgb, rgb_imgs_dir)
            with open(rgb_base_dir / "fps.txt", "w") as f: f.write(str(scene.frame_rate))
        elif key in writer_map:
            writer_map[key](value, imgs_dir)
            with open(base_dir / "fps.txt", "w") as f: f.write(str(scene.frame_rate))
                
    # Metadata
    data = {
            "scene_metadata": kb.get_scene_metadata(scene), 
            "camera": kb.get_camera_info(scene.camera), 
            "object": kb.get_instance_info(scene, visible_foreground_assets) 
        }    
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = annotations_dir / f"{seq_name}_metadata.json"
    kb.file_io.write_json(filename=metadata_path, data=data)
    
    del scene, renderer, simulator
    gc.collect()
    return data, seq_name

def clean_blender_scene():
    """
    Forces complete cleanup of the Blender scene, removing all data.
    This is the manual and robust alternative to kb.utils.reset_blend_file().
    """
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for _ in range(4):
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

    print("INFO: Blender scene manually cleaned.")

# ============================================================
# --- RUN MODES ---
# ============================================================

def standard_run_mode(num_sequences, output_root, args, coco_data, annotation_id_counter, image_id_counter):
    
    for seq_batch in range(num_sequences):
        seed = get_seed()
        args.seed = seed
        print(f"\n🌟 Starting sequence batch {seq_batch + 1}/{num_sequences} with SEED {seed}")
        
        layout_data = generate_scene_layout(args)

        for intensity in args.light_levels:
                    for cam_mode in args.camera_modes:
                        if cam_mode == "fixed":
                            for cam_pos in [camera_positions_all[name] for name in args.camera_positions]:
                                print(f"\n🚀 Generating sequence | batch {seq_batch + 1} | seed {args.seed} | light={int(intensity*100)}% | camera_mode={cam_mode}")
                                kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=intensity, light_color=None, light_orientation=None, camera_position=cam_pos, camera_mode=cam_mode, FLAGS=args, output_root=output_root)
                                coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
                        else:
                            # For non-fixed camera modes, don't cycle through camera positions
                            print(f"\n🚀 Generating sequence | batch {seq_batch + 1} | seed {args.seed} | light={int(intensity*100)}% | camera_mode={cam_mode}")
                            kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=intensity, light_color=None, light_orientation=None, camera_position=None, camera_mode=cam_mode, FLAGS=args, output_root=output_root)
                            coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
  
    
    return coco_data, annotation_id_counter, image_id_counter

def total_random_run_mode(args, output_root, coco_data, annotation_id_counter, image_id_counter):
    print(f"Random run mode... Generating {args.number_of_random_sequences} sequences with random parameters.")
    for i in range(args.number_of_random_sequences):

        seed = get_seed()
        args.seed = seed
        layout_data = generate_scene_layout(args)
        light_orientation = Random.choice(list(light_orientations_all.values()))
        light_intensity = Random.choice(light_levels_all)
        camera_mode = Random.choice(CAMERA_TYPES)
        if camera_mode == "fixed": 
            camera_position = Random.choice(list(camera_positions_all.values()))
        else:
            camera_position = None
        light_color = Random.choice(list(light_colors_all.values()))
        print(f"\n🚀 Generating sequence | seed {args.seed} | light={int(light_intensity*100)}%| color={light_color} | light_orientation={light_orientation} | camera_mode={camera_mode}")
        kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=light_intensity, light_color=light_color, light_orientation=light_orientation, camera_position=camera_position, camera_mode=camera_mode, FLAGS=args, output_root=output_root)
        coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
    return coco_data, annotation_id_counter, image_id_counter

# ============================================================
# --- MAIN ---
# ============================================================

def main():
    args = parse_args()

    num_refresh = args.refresh_item_number
    output_root = args.output_root
    coco_data = initialize_coco_dict()
    annotation_id_counter = 1
    image_id_counter = 1
    print("INFO: Initializing Kubric environment (Bootstrap)...")
    kb.setup(args) 
    clean_blender_scene()
    if args.standard_mode:
        print("Standard mode selected.....")
        coco_data, annotation_id_counter, image_id_counter = standard_run_mode(num_refresh, output_root, args, coco_data, annotation_id_counter, image_id_counter)
    if args.rand_gen:
        print("Total random mode selected.....")
        coco_data, annotation_id_counter, image_id_counter= total_random_run_mode(args, output_root, coco_data, annotation_id_counter, image_id_counter)
    annotations_path = output_root / "annotations.json"
    kb.file_io.write_json(filename=annotations_path, data=coco_data)
    print(f"\n✅ COCO annotations saved to: '{annotations_path}'")
    print("\n✅ All sequences have been generated.")

    kb.done()


if __name__ == "__main__":
    main()