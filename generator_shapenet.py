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



# ============================================================
# --- CONFIGURAZIONE GLOBALE ---
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

# Parametri di base 
#TODO: TOGLIERLI DA QUI DATO CHE SONO NEL PARSER
RESOLUTION = (256, 256)
FRAME_END = 24
FRAME_RATE = 12
STEP_RATE = 240
MIN_STATIC, MAX_STATIC = 1, 2
MIN_DYNAMIC, MAX_DYNAMIC = 1, 2
SPAWN_REGION_STATIC = [[-5, -5, 0], [5, 5, 0]]
SPAWN_REGION_DYNAMIC = [[-5, -5, 1], [5, 5, 6]]
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
selector = HDRISelector(source=HDRI_SOURCE, json_path="hdri.json")


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
    "front": (0, -8, 0),            # 0° frontale no con luce 0
    "tilt_30": (4, -7, 3),          # 30° inclinata no con luce 0
    "tilt_60": (7, -4, 5),          # 60° obliqua si con luce 0
    "side_90": (8, 0, 0),           # 90° laterale puro no con luce 0
    "retro_120": (7, 4, 3),         # 120° retro-inclinata no con luce 0
    "back_180": (0, 8, 0),          # 180° dietro nope
    "top": (0, 0, 8),               # zenitale si vede l'oggetto troppo
    "bottom": (0, 0, -8),           # vista dal basso questo è ok 
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
# --- LIGHT DIRECTION SELECTION---
# ============================================================
def get_light_direction(luminosity: float, rng: np.random.RandomState, distance: float = 10.0) -> np.ndarray:
    """
    Calcola una posizione casuale per la luce su una semisfera,
    mantenendo l'altezza legata alla luminosità.

    Args:
        luminosity (float): Il livello di luminosità [0.0, 1.0], che controlla l'altezza.
        rng (np.random.RandomState): Il generatore di numeri casuali per la riproducibilità.
        distance (float): La distanza della luce dal centro della scena.

    Returns:
        np.ndarray: Un vettore di posizione 3D per la luce direzionale.
    """
    # Assicura che la luminosità sia nell'intervallo [0, 1]
    luminosity = np.clip(luminosity, 0.0, 1.0)
    
    # 1. Calcola l'altezza (angolo di elevazione) basata sulla luminosità
    #    1.0 -> 90 gradi (zenith), 0.0 -> 0 gradi (orizzonte)
    elevation_rad = luminosity * (math.pi / 2.0)
    
    # 2. Scegli una direzione casuale sull'orizzonte (angolo di azimuth)
    #    Scegliamo un punto casuale su un cerchio completo (0 a 360 gradi)
    azimuth_rad = rng.uniform(0, 2 * math.pi)
    
    # 3. Converti le coordinate sferiche (distanza, elevazione, azimuth) in cartesiane (x, y, z)
    #    Proiezione della distanza sul piano XY
    xy_projection = distance * math.cos(elevation_rad)
    
    x = xy_projection * math.cos(azimuth_rad)
    y = xy_projection * math.sin(azimuth_rad)
    z = distance * math.sin(elevation_rad)
    
    # Aggiungi un'altezza minima per evitare artefatti di rendering
    if z < 0.1:
        z = 0.1
        
    return np.array([x, y, z])

# ============================================================
# --- cAMERA MOVEMENT FUNCTIONS---
# ============================================================

def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 7.,
    outer_radius: float = 9.,
    z_offset: float = 0.1,
):
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

def get_linear_lookat_motion_start_end(
    inner_radius: float = 0.5,
    outer_radius: float = 2.0,
):
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
# --- PARSING DATI ---
# ============================================================


def parse_args():
    parser = kb.ArgumentParser()
    """     parser.set_defaults(
            resolution=RESOLUTION,
            frame_end=FRAME_END,
            frame_rate=FRAME_RATE,
            step_rate=STEP_RATE,
        ) """

    parser.add_argument("--sequences",type=int, default=5)
    parser.add_argument("--light_levels", nargs="+", type=float, default=[1.0])
    # pattern = nome seguito da 3 o 4 float
    parser.add_argument("--light_orientations", nargs="+", default=["side_45", "0.0", "0.0", "0.7854"])
    parser.add_argument("--camera_positions", nargs="+", default=["tilt_30", "4", "-7", "3"])
    parser.add_argument("--light_colors", nargs="+", default=["white", "1.0", "1.0", "1.0", "1.0"])
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--rand_gen", type=lambda x: x.lower() == 'true', default=False, help="Genera sequenze aggiuntive con parametri casuali e oggetti multipli")

    #other args by default
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--camera_mode", type=str, choices=["fixed", "linear_movement", "panning"], default="fixed")
    parser.add_argument("--max_camera_movement", type=float, default=4.0)

    return parser.parse_args()


# ============================================================
# --- COCO Style Annotations ---       
# ============================================================
# La funzione initialize_coco_dict() rimane la stessa
def initialize_coco_dict():
    """Crea la struttura di base, vuota, di un file di annotazioni COCO."""
    return {
        "info": { "year": datetime.now().year, "version": "1.0", "description": "Dataset simulato con Kubric", "contributor": "Matteo Tinacci" },
        "licenses": [], "categories": [], "images": [], "annotations": []
    }

def update_coco_from_metadata(coco_dict, kubric_metadata, seq_id, annotation_id, image_id):
    """
    Usa la struttura dati REALE (una lista di oggetti con chiave "bbox")
    per aggiornare il dizionario COCO.
    """
    # --- 1. Gestione Categorie ---
    # kubric_metadata["object"] è una LISTA.
    objects_metadata_list = kubric_metadata["object"]
    existing_categories = {cat['name']: cat['id'] for cat in coco_dict['categories']}

    # Iteriamo direttamente sulla LISTA.
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
    
    # --- 2. Gestione Immagini e Annotazioni ---
    scene_info = kubric_metadata["scene_metadata"]
    num_frames = scene_info["num_frames"]
    height, width = scene_info["resolution"]
    
    # Debug: print metadata structure info
    print(f"📊 Processing {len(objects_metadata_list)} objects for {num_frames} frames")
    for i, obj_info in enumerate(objects_metadata_list):
        vis_len = len(obj_info["visibility"]) if "visibility" in obj_info else 0
        bbox_len = len(obj_info["bboxes"]) if "bboxes" in obj_info else 0
        print(f"  Object {i}: visibility={vis_len}, bboxes={bbox_len} frames")
    
    for frame_idx in range(num_frames):
        current_image_id = image_id + frame_idx
        image_info = {
            "id": current_image_id,
            "file_name": f"seq{seq_id}/imgs/{frame_idx:05d}.png", 
            "width": width,
            "height": height
        }
        coco_dict['images'].append(image_info)
        
        # Iteriamo di nuovo sulla LISTA di oggetti.
        for obj_info in objects_metadata_list:
            # Check if object has any visibility data or if all visibility values are 0
            if ("visibility" not in obj_info or 
                len(obj_info["visibility"]) == 0 or
                all(vis == 0 for vis in obj_info["visibility"])):
                # Object never visible - create annotation with zero area bbox
                annotation_info = {
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": existing_categories[obj_info["asset_id"]],
                    "bbox": [0.0, 0.0, 0.0, 0.0],  # Zero-area bbox for invisible objects
                    "area": 0.0,
                    "iscrowd": 0,
                    "segmentation": [],
                    "visibility": "not_visible"  # Custom field to mark invisible objects
                }
                coco_dict['annotations'].append(annotation_info)
                annotation_id += 1
                continue
                
            # Check if frame_idx is within bounds and object is visible
            if (frame_idx < len(obj_info["visibility"]) and 
                obj_info["visibility"][frame_idx] > 0):
                
                # Check if bboxes data exists for this frame
                if ("bboxes" in obj_info and 
                    frame_idx < len(obj_info["bboxes"])):
                    # Object is visible and has bbox data
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
                    # Object is marked as visible but no bbox data - create zero bbox
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": existing_categories[obj_info["asset_id"]],
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "area": 0.0,
                        "iscrowd": 0,
                        "segmentation": [],
                        "visibility": "visible_no_bbox"  # Custom field for debugging
                    }
                    coco_dict['annotations'].append(annotation_info)
                    annotation_id += 1
            elif frame_idx < len(obj_info["visibility"]):
                # Object exists in this frame but is not visible
                annotation_info = {
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": existing_categories[obj_info["asset_id"]],
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "area": 0.0,
                    "iscrowd": 0,
                    "segmentation": [],
                    "visibility": "not_visible_frame"  # Custom field to mark frame-specific invisibility
                }
                coco_dict['annotations'].append(annotation_info)
                annotation_id += 1
            
    next_image_id = image_id + num_frames
    
    return coco_dict, annotation_id, next_image_id


# ============================================================
# --- FUNZIONE DI GENERAZIONE SEQUENZA ---
# ============================================================

def generate_sequence(seq_id: int, light_intensity: float, orientation: tuple, camera_position: tuple, light_color: tuple, FLAGS, output_root: Path = Path("output")):

    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)


    #--- Scene background HDRI ---
    hdri_id = selector.pick(light_intensity)
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = HDRI_SOURCE.create(asset_id=hdri_id)
    scene.metadata["background"] = hdri_id

    # --- Usa le tue variabili 'light_intensity' e 'light_color' ---
    logging.info(f"Using light intensity: {light_intensity:.2f}")
    logging.info(f"Using light color: {light_color}")

    # --- CALIBRAZIONE DELLA LUMINOSITÀ ---
    LIGHT_SOURCE_GAMMA = 1.0
    light_source_intensity = light_intensity ** LIGHT_SOURCE_GAMMA
    BACKGROUND_GAMMA = 1.0
    background_visual_intensity = light_intensity ** BACKGROUND_GAMMA
    AMBIENT_LIGHT_FACTOR = 0.8
    print(f"INFO: Intensità luce: {light_source_intensity:.4f} | Intensità sfondo: {background_visual_intensity:.4f} | Luce ambiente: {AMBIENT_LIGHT_FACTOR}")

    # --- Luce ambientale del Mondo di Blender ---
    world_background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_background_node:
        # Usiamo direttamente la variabile light_color, che dovrebbe essere un RGBA
        world_background_node.inputs["Color"].default_value = light_color
        world_background_node.inputs["Strength"].default_value = AMBIENT_LIGHT_FACTOR * light_intensity
        print("INFO: Luce ambientale del Mondo configurata con colore personalizzato.")

    # --- Aggiungiamo un Sole ---
    SUN_BASE_INTENSITY = 0.5
    sun = kb.DirectionalLight(
        name="sun",
        position=(-1, -1, 3.0),
        look_at=(0, 0, 0),
        # Forniamo solo i primi 3 componenti (RGB)
        color=light_color[:3],
        intensity=SUN_BASE_INTENSITY * light_source_intensity
    )
    scene.add(sun)
    print("INFO: Luce del Sole configurata con colore personalizzato.")

    # --- Dome dello sfondo ---
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

            # --- MODIFICA CHIAVE ---
            # Creiamo un nuovo vettore che combina il colore e l'intensità dello sfondo.
            # Moltiplichiamo ogni componente di light_color per l'intensità.
            bg_vec = [c * background_visual_intensity for c in light_color[:3]]
            # -------------------------

            color_dimmer_node.inputs[1].default_value = bg_vec

            node_tree.links.new(texture_node.outputs["Color"], color_dimmer_node.inputs[0])

            for dest_socket in original_destinations:
                node_tree.links.new(node_tree.nodes["Vector Math"].outputs["Vector"], dest_socket)

    
    
    # --- Camera ---
    camera_mode = FLAGS.camera_mode
    max_camera_speed = FLAGS.max_camera_movement
    logging.info(f"🎥 Setting up Camera in '{camera_mode}' mode...")
    print(f"🎥 Setting up Camera in '{FLAGS.camera_mode}' mode...")

    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)

    if camera_mode == "fixed":
        #scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7., outer_radius=9., offset=0.1)
        scene.camera.position = camera_position
        scene.camera.look_at((0, 0, 0))
        logging.info(f"Camera position fixed at {scene.camera.position}")

    elif camera_mode == "linear_movement":
        # Calcola una velocità casuale per questo movimento
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





    # === STATIC OBJECTS ===
    num_static = rng.randint(MIN_STATIC, MAX_STATIC + 1)
    print(f"📦 Generating {num_static} static objects...")
    for idx in range(num_static):
        random_class = rng.choice(classes_all)
        shape_ids = chooseClass(random_class)
        shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        obj.segmentation_id = idx + 1  # Assign segmentation ID

        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])  # Normalize scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_STATIC, rng=rng)
        print(f"📦 Static object {shape_id} at {obj.position} with velocity {obj.velocity}")

    print("Simulating to let objects settle...")
    _, _ = simulator.run(frame_start=-100, frame_end=0)

    print("Stopping any moving objects...")
    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.angular_velocity = (0., 0., 0.)
            obj.friction = 0.5
            obj.restitution = 0.5

    # === DYNAMIC OBJECTS ===
    num_dynamic = rng.randint(MIN_DYNAMIC, MAX_DYNAMIC + 1)
    print(f"🚀 Generating {num_dynamic} dynamic objects...")
    for idx in range(num_dynamic):
        random_class = rng.choice(classes_all)
        shape_ids = chooseClass(random_class)
        shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        obj.segmentation_id = num_static + idx + 1  # Unique ID
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_DYNAMIC, rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
        print(f"🚀 Dynamic object {shape_id} with velocity {obj.velocity} at position {obj.position}")

    # === Simulation ===
    print("🎬 Simulazione...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)


    # === Rendering ===
    print("Saving state...")
    renderer.save_state(output_root / f"states/seq{seq_id}.blend")
    print("🎥 Rendering...")
    frames_dict = renderer.render()


    # --- Calcola visibilità e aggiusta segmentation ---
    kb.compute_visibility(frames_dict["segmentation"], scene.assets)
    frames_dict["segmentation"] = kb.adjust_segmentation_idxs(
        frames_dict["segmentation"], scene.assets, [obj]).astype(np.uint8)

    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    kb.post_processing.compute_bboxes(frames_dict["segmentation"],
                                    visible_foreground_assets)

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
    exclude_names = {"floor", "camera", "sun", "dome"}
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
    return data  # Return metadata for COCO update


# ============================================================
# --- CHOOSE IDS ---
# ============================================================
def chooseClass(class_name):
    return [name for name, spec in ASSET_SOURCE._assets.items() if spec["metadata"]["category"] == class_name]

# ============================================================
# --- SET RANDOM SEED ---
# ============================================================

def get_seed():
    """Return a random seed for generation."""
    return int(time.time() * 1000) % 2**32



# ============================================================
# --- TEST SEED ---
# ============================================================


def generate_scene_layout(seed: int, FLAGS):
    """
    Usa un seed per generare una lista di oggetti con le loro proprietà 
    di "spawn" (posizione iniziale, scala, ecc.). 
    NON esegue la simulazione di assestamento.
    """
    print(f"🔩 Generazione della lista di spawn con seed {seed}...")
    rng = np.random.RandomState(seed)

    FLAGS.seed = seed
    temp_scene, _, _, _ = kb.setup(FLAGS)
    simulator = KubricSimulator(temp_scene)
    
    dome = KUBASIC_SOURCE.create(asset_id="dome", name="dome", static=True, background=True)
    temp_scene += dome
    
    layout_data = []
    # === 1. Posizionamento Iniziale Oggetti Statici ===
    num_static = rng.randint(MIN_STATIC, MAX_STATIC + 1)
    print(f"  📦 Generating {num_static} static objects...")
    for idx in range(num_static):
        random_class = rng.choice(classes_all)
        shape_ids = chooseClass(random_class)
        shape_id = rng.choice(shape_ids)
        
        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        
        temp_scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_STATIC, rng=rng)
        
        layout_data.append({
            "asset_id": obj.asset_id, "segmentation_id": idx + 1,
            "position": tuple(obj.position), "quaternion": tuple(obj.quaternion),
            "scale": tuple(obj.scale), "static": True
        })

    # === 4. Posizionamento Oggetti Dinamici ===
    num_dynamic = rng.randint(MIN_DYNAMIC, MAX_DYNAMIC + 1)
    print(f"  📦 Generating {num_dynamic} dynamic objects...")
    for idx in range(num_dynamic):
        random_class = rng.choice(classes_all)
        shape_ids = chooseClass(random_class)
        shape_id = rng.choice(shape_ids)

        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])

        temp_scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_DYNAMIC, rng=rng)
        velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
        
        layout_data.append({
            "asset_id": obj.asset_id, "segmentation_id": num_static + idx + 1,
            "position": tuple(obj.position), "quaternion": tuple(obj.quaternion),
            "scale": tuple(obj.scale), "velocity": tuple(velocity),
            "angular_velocity": (0., 0., 0.), "static": False
        })
        
    print(f"  -> Lista di spawn per {len(layout_data)} oggetti creata.")
    return layout_data


def render_variation(seq_id: int, layout_data: list, light_intensity: float, orientation: tuple, camera_position: tuple, light_color: tuple, FLAGS, output_root: Path = Path("output")):
    """Crea una scena, la popola, imposta i parametri della variazione (camera, luci) e include la logica completa per dome, HDRI, rendering e salvataggio."""
    # 1. SETUP INIZIALE
    # Eseguiamo la pulizia solo qui, per garantire che ogni variazione sia indipendente
    clean_blender_scene()

    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)


    # 2. SETUP DELLA SCENA (LUCI, SFONDO, CAMERA)
    hdri_id = selector.pick(light_intensity)
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = HDRI_SOURCE.create(asset_id=hdri_id)
    scene.metadata["background"] = hdri_id

    # --- Usa le tue variabili 'light_intensity' e 'light_color' ---
    logging.info(f"Using light intensity: {light_intensity:.2f}")
    logging.info(f"Using light color: {light_color}")

    # --- CALIBRAZIONE DELLA LUMINOSITÀ ---
    LIGHT_SOURCE_GAMMA = 2.2
    light_source_intensity = light_intensity ** LIGHT_SOURCE_GAMMA
    BACKGROUND_GAMMA = 2.2
    background_visual_intensity = light_intensity ** BACKGROUND_GAMMA
    AMBIENT_LIGHT_FACTOR = 0.1
    print(f"INFO: Intensità luce: {light_source_intensity:.4f} | Intensità sfondo: {background_visual_intensity:.4f} | Luce ambiente: {AMBIENT_LIGHT_FACTOR}")

    # --- Luce ambientale del Mondo di Blender ---
    world_background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_background_node:
        # Usiamo direttamente la variabile light_color, che dovrebbe essere un RGBA
        world_background_node.inputs["Color"].default_value = light_color
        world_background_node.inputs["Strength"].default_value = AMBIENT_LIGHT_FACTOR * light_intensity
        print("INFO: Luce ambientale del Mondo configurata con colore personalizzato.")

    # --- Aggiungiamo un Sole ---
    SUN_BASE_INTENSITY = 0.25
    sun = kb.DirectionalLight(
        name="sun",
        position=(-1, -1, 3.0),
        look_at=(0, 0, 0),
        # Forniamo solo i primi 3 componenti (RGB)
        color=light_color[:3],
        intensity=SUN_BASE_INTENSITY * light_source_intensity
    )
    scene.add(sun)
    print("INFO: Luce del Sole configurata con colore personalizzato.")

    # --- Dome dello sfondo ---
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

            # --- MODIFICA CHIAVE ---
            # Creiamo un nuovo vettore che combina il colore e l'intensità dello sfondo.
            # Moltiplichiamo ogni componente di light_color per l'intensità.
            bg_vec = [c * background_visual_intensity for c in light_color[:3]]
            # -------------------------

            color_dimmer_node.inputs[1].default_value = bg_vec

            node_tree.links.new(texture_node.outputs["Color"], color_dimmer_node.inputs[0])

            for dest_socket in original_destinations:
                node_tree.links.new(node_tree.nodes["Vector Math"].outputs["Vector"], dest_socket)

    # --- Camera ---
    camera_mode = FLAGS.camera_mode
    max_camera_speed = FLAGS.max_camera_movement
    logging.info(f"🎥 Setting up Camera in '{camera_mode}' mode...")
    print(f"🎥 Setting up Camera in '{FLAGS.camera_mode}' mode...")

    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)

    if camera_mode == "fixed":
        #scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7., outer_radius=9., offset=0.1)
        scene.camera.position = camera_position
        scene.camera.look_at((0, 0, 0))
        logging.info(f"Camera position fixed at {scene.camera.position}")

    elif camera_mode == "linear_movement":
        # Calcola una velocità casuale per questo movimento
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
# --- FASE 1: Popolamento e Assestamento degli Oggetti Statici ---
    print("INFO: Fase 1 - Popolamento e assestamento oggetti statici...")
    
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
            # Mettiamo da parte i dati degli oggetti dinamici per dopo
            dynamic_objects_data.append(obj_data)

    # Eseguiamo una simulazione sufficientemente lunga per l'assestamento
    # dei soli oggetti statici.
    if static_asset_references:
        print(f"INFO: Esecuzione simulazione di assestamento per {len(static_asset_references)} oggetti statici...")
        simulator.run(frame_start=-100, frame_end=0)

        # Fermiamo completamente gli oggetti statici dopo l'assestamento
        for obj in static_asset_references:
            obj.velocity = (0., 0., 0.)
            obj.angular_velocity = (0., 0., 0.)

    # --- FASE 2: Popolamento degli Oggetti Dinamici ---
    print(f"INFO: Fase 2 - Popolamento di {len(dynamic_objects_data)} oggetti dinamici...")
    for obj_data in dynamic_objects_data:
        obj = ASSET_SOURCE.create(asset_id=obj_data["asset_id"])
        obj.segmentation_id = obj_data["segmentation_id"]
        obj.position = obj_data["position"]
        obj.quaternion = obj_data["quaternion"]
        obj.scale = obj_data["scale"]
        obj.velocity = obj_data["velocity"]
        obj.angular_velocity = obj_data["angular_velocity"]
        scene += obj

    # --- FASE 3: Simulazione Finale e Rendering ---
    print("🎬 Simulazione finale...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    print("Saving state...")
    renderer.save_state(output_root / f"states/seq{seq_id}.blend")
    print("🎥 Rendering...")
    frames_dict = renderer.render()

    # 5. POST-PROCESSING E SALVATAGGIO
    kb.compute_visibility(frames_dict["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets if np.max(asset.metadata["visibility"]) > 0]
    kb.post_processing.compute_bboxes(frames_dict["segmentation"], visible_foreground_assets)

    # Salvataggio frame
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
            with open(rgb_base_dir / "fps.txt", "w") as f: f.write(str(scene.frame_rate))
        elif key in writer_map:
            writer_map[key](value, imgs_dir)
            with open(base_dir / "fps.txt", "w") as f: f.write(str(scene.frame_rate))
                
    # Metadata
    exclude_names = {"floor", "camera", "sun", "dome"}
    scene_objects = [obj for obj in scene.assets if obj.name not in exclude_names]
    data = {"scene_metadata": kb.get_scene_metadata(scene), "camera": kb.get_camera_info(scene.camera), "object": kb.get_instance_info(scene, scene_objects)}
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = annotations_dir / f"seq{seq_id}_metadata.json"
    kb.file_io.write_json(filename=metadata_path, data=data)
    
    gc.collect()
    return data


def clean_blender_scene():
    """
    Forza la pulizia completa della scena di Blender, rimuovendo tutti i dati.
    Questa è l'alternativa manuale e robusta a kb.utils.reset_blend_file().
    """
    # Assicurati che non ci sia una scena attiva in modalità modifica
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Seleziona tutti gli oggetti e cancellali
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Cancella i dati orfani (mesh, materiali, ecc. non più usati)
    # Eseguire più volte garantisce la pulizia delle dipendenze
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

    print("INFO: Scena di Blender pulita manualmente.")
# ============================================================
# --- MAIN ---
# ============================================================

def main():
    args = parse_args()
    print("🎛️  Configurazione in corso...")
    print("number of sequences:", args.sequences)
    print("Light levels:", args.light_levels)
    print("Orientations:", args.light_orientations)
    print("Cameras:", args.camera_positions)
    print("Colors:", args.light_colors)

    def _normalize_list_arg(lst):
        """Accetta ['a','b',...] oppure ['a,b;c d'] e restituisce ['a','b','c','d'].
        Se la lista contiene numeri (int/float), li restituisce così com'è.
        """
        if not lst:
            return []

        # Caso: lista di numeri → ritorna direttamente
        if all(isinstance(x, (int, float)) for x in lst):
            return lst

        # Caso: singola stringa da splittare
        if len(lst) == 1 and isinstance(lst[0], str):
            s = lst[0]
            return [t for t in re.split(r'[,\s;]+', s) if t]

        # Caso: lista di stringhe già separata
        return [str(x) for x in lst]


    # -- number of sequences
    num_sequences = args.sequences


    # -- light_levels
    raw_levels = _normalize_list_arg(args.light_levels)
    try:
        light_levels = [float(x) for x in raw_levels]
    except ValueError as e:
        raise ValueError(f"light_levels non validi: {raw_levels}") from e

    # -- light_orientations (gruppi da 4: name x y z)
    raw_orients = _normalize_list_arg(args.light_orientations)
    if len(raw_orients) % 4 != 0:
        raise ValueError(f"light_orientations: numero token non multiplo di 4: {raw_orients}")
    light_orientations = {}
    for i in range(0, len(raw_orients), 4):
        name = raw_orients[i]
        try:
            x, y, z = map(float, raw_orients[i+1:i+4])
        except ValueError as e:
            raise ValueError(f"Orientazione non valida per '{name}': {raw_orients[i+1:i+4]}") from e
        light_orientations[name] = (x, y, z)

    # -- camera_positions (gruppi da 4: name x y z)
    raw_cams = _normalize_list_arg(args.camera_positions)
    if len(raw_cams) % 4 != 0:
        raise ValueError(f"camera_positions: numero token non multiplo di 4: {raw_cams}")
    camera_positions = {}
    for i in range(0, len(raw_cams), 4):
        name = raw_cams[i]
        try:
            x, y, z = map(float, raw_cams[i+1:i+4])
        except ValueError as e:
            raise ValueError(f"Posizione camera non valida per '{name}': {raw_cams[i+1:i+4]}") from e
        camera_positions[name] = (x, y, z)

    # -- light_colors (gruppi da 5: name r g b a)
    raw_colors = _normalize_list_arg(args.light_colors)
    if len(raw_colors) % 5 != 0:
        raise ValueError(f"light_colors: numero token non multiplo di 5: {raw_colors}")
    light_colors = {}
    for i in range(0, len(raw_colors), 5):
        name = raw_colors[i]
        try:
            r, g, b, a = map(float, raw_colors[i+1:i+5])
        except ValueError as e:
            raise ValueError(f"Colore luce non valido per '{name}': {raw_colors[i+1:i+5]}") from e
        light_colors[name] = (r, g, b, a)


    print("✅ Config caricate:")
    print("number of sequences:", num_sequences)
    print("Light levels:", light_levels)
    print("Orientations:", light_orientations)
    print("Cameras:", camera_positions)
    print("Colors:", light_colors)


    # Use output_root from arguments
    output_root = args.output_root

    seq_id = 0
    coco_data = initialize_coco_dict()
    annotation_id_counter = 1
    image_id_counter = 1
    print("INFO: Inizializzazione dell'ambiente Kubric (Bootstrap)...")
    kb.setup(args) 
    clean_blender_scene()
    for seq_batch in range(num_sequences):
        # 1. Imposta il SEED una sola volta per l'intero batch
        seed = get_seed()
        args.seed = seed
        print(f"\n🌟 Inizio sequence batch {seq_batch + 1}/{num_sequences} con SEED {seed}")
        
        # 2. GENERA IL LAYOUT CASUALE UNA SOLA VOLTA USANDO IL SEED
        layout_data = generate_scene_layout(seed, args)

        # 3. Ora cicla sulle variazioni, che sono PURAMENTE DETERMINISTICHE
        for intensity in light_levels:
            for orient_name, orientation in light_orientations.items():
                for cam_name, cam_pos in camera_positions.items():
                    for color_name, color_value in light_colors.items():
                        print(f"\n🚀 Generazione sequenza {seq_id} | batch {seq_batch + 1} | seed {args.seed} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
                        kubric_metadata = render_variation(seq_id=seq_id, layout_data=layout_data, light_intensity=intensity, orientation=orientation, camera_position=cam_pos, light_color=color_value, FLAGS=args, output_root=output_root)
                        # 4. Aggiorna il dizionario COCO con i nuovi dati
                        coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_id, annotation_id_counter, image_id_counter)
                        seq_id += 1

    annotations_path = output_root / "annotations.json"
    kb.file_io.write_json(filename=annotations_path, data=coco_data)
    print(f"\n✅ Annotazioni COCO salvate in: '{annotations_path}'")
    print("\n✅ Tutte le sequenze sono state generate.")

    # # Generate additional sequences with multiple objects and random parameters if enabled
    # if args.rand_gen:
    #     print(f"\n🎲 Generating additional sequences with random multiple objects...")

    #     # Modified parameters for multiple objects
    #     global MIN_STATIC, MAX_STATIC, MIN_DYNAMIC, MAX_DYNAMIC
    #     MIN_STATIC, MAX_STATIC = 1, 2
    #     MIN_DYNAMIC, MAX_DYNAMIC = 1, 2

    #     # Generate 10 additional sequences with random parameters
    #     for i in range(1):
    #         # Set a seed for reproducibility
    #         seed = get_seed()
    #         args.seed = seed
            
    #         # Random shape selection
    #         random_class = Random.choice(classes_all)
    #         shape_ids = chooseClass(random_class)

    #         # Random light parameters
    #         intensity = Random.choice(light_levels_all)
    #         orient_name, orientation = Random.choice(list(light_orientations_all.items()))
    #         cam_name, cam_pos = Random.choice(list(camera_positions_all.items()))
    #         color_name, color_value = Random.choice(list(light_colors_all.items()))

    #         print(f"\n🎲 Random sequence {seq_id} | shape={random_class} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
    #         generate_sequence(seq_id, intensity, orientation, cam_pos, color_value, args, output_root)
    #         seq_id += 1

    kb.done()


if __name__ == "__main__":
    main()