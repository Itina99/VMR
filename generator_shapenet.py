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

CAMERA_TYPES = ["fixed", "linear_movement", "panning"]

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

light_levels_all = [0.25, 0.5, 0.75, 1.0]  # 0–100%

light_orientations_all = {
    "side_45": (0., 0., np.pi/4),
    "side_90": (0., 0., np.pi/2),
    "back_135": (0., 0., 3*np.pi/4),
    "top": (np.pi/2, 0., 0.),}

camera_positions_all = {
    "tilt_30": (4, -7, 3),          # 30° inclinata no con luce 0
    "tilt_60": (7, -4, 5),          # 60° obliqua si con luce 0
    "retro_120": (7, 4, 3),         # 120° retro-inclinata no con luce 0
    "top": (0, 0, 8),               # zenitale si vede l'oggetto troppo

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
# --- LIGHT DIRECTION AND COLOR SELECTION---
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

def get_light_color_by_intensity(light_intensity: float) -> tuple:
    """
    Choose a light color based on intensity to mimic time of day progression.
    
    Args:
        light_intensity (float): Light intensity value between 0.0 and 1.0
        
    Returns:
        tuple: RGBA color tuple representing the time of day
    """
    # Clamp intensity to valid range
    intensity = np.clip(light_intensity, 0.0, 1.0)
    
    # Define color transitions that mimic natural lighting throughout the day
    if intensity >= 0.9:  # Noon - bright white
        return (1.0, 1.0, 1.0, 1.0)
    elif intensity >= 0.7:  # Mid-morning/afternoon - slightly warm white
        return (1.0, 0.95, 0.85, 1.0)
    elif intensity >= 0.5:  # Early morning/late afternoon - warm yellow
        return (1.0, 0.9, 0.6, 1.0)
    elif intensity >= 0.25:  # Golden hour - orange
        return (1.0, 0.7, 0.3, 1.0)
    elif intensity >= 0.1:  # Sunset/sunrise - deep orange-red
        return (1.0, 0.5, 0.2, 1.0)
    else:  # Night/twilight - very dim blue
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
    parser.add_argument("--standard_mode", type=lambda x: x.lower() == 'true', default=True, help="Genera sequenze standard con un oggetto per scena e parametri fissi")
    parser.add_argument("--rand_gen", type=lambda x: x.lower() == 'true', default=False, help="Genera sequenze aggiuntive con parametri casuali e oggetti multipli")
    parser.add_argument("--number_of_random_sequences", type=int, default=5, help="Numero di sequenze casuali da generare se --rand_gen è attivo")
    #other args by default
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--max_camera_movement", type=float, default=4.0)
    parser.add_argument("--max_dynamic_objects", type=int, default=3)
    parser.add_argument("--max_static_objects", type=int, default=3)
    parser.add_argument("--min_dynamic_objects", type=int, default=1)
    parser.add_argument("--min_static_objects", type=int, default=1)
    parser.add_argument("--spawning_region_static", nargs=6, type=float, default=[-5, -5, 0, 5, 5, 0], help="Region to spawn static objects: x_min y_min z_min x_max y_max z_max")
    parser.add_argument("--spawning_region_dynamic", nargs=6, type=float, default=[-5, -5, 1, 5, 5, 6], help="Region to spawn dynamic objects: x_min y_min z_min x_max y_max z_max")
    parser.add_argument("--velocity_range", nargs=6, type=float, default=[-4.0, -4.0, 0.0, 4.0, 4.0, 0.0], help="Velocity range: x_min y_min z_min x_max y_max z_max")
    
    # Convert flat lists to nested format after parsing
    args = parser.parse_args()
    args.spawning_region_static = [args.spawning_region_static[0:3], args.spawning_region_static[3:6]]
    args.spawning_region_dynamic = [args.spawning_region_dynamic[0:3], args.spawning_region_dynamic[3:6]]
    args.velocity_range = [args.velocity_range[0:3], args.velocity_range[3:6]]

    return args

# ============================================================
# --- COCO STYLE ANNOTATIONS ---
# ============================================================

def initialize_coco_dict():
    """Crea la struttura di base, vuota, di un file di annotazioni COCO."""
    return {
        "info": { "year": datetime.now().year, "version": "1.0", "description": "Dataset simulato con Kubric", "contributor": "Matteo Tinacci" },
        "licenses": [], "categories": [], "images": [], "annotations": []
    }

def update_coco_from_metadata(coco_dict, kubric_metadata, seq_name, annotation_id, image_id):
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
            "file_name": f"{seq_name}/imgs/{frame_idx:05d}.png", 
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
# --- UTILS ---
# ============================================================

def chooseClass(class_name):
    return [name for name, spec in ASSET_SOURCE._assets.items() if spec["metadata"]["category"] == class_name]

def get_seed():
    """Return a random seed for generation."""
    return int(time.time() * 1000) % 2**32

def get_sequence_name():
    """Generate a unique sequence name using timestamp and random hash."""
    
    # Generate timestamp component (milliseconds since epoch)
    timestamp = int(time.time() * 1000)
    
    # Generate a random hash component
    random_bytes = os.urandom(4)  # Reduced since we have timestamp
    hash_component = hashlib.md5(random_bytes).hexdigest()[:6]
    
    return f"{timestamp}_{hash_component}"
# ============================================================
# --- SCENE GENERATION ---
# ============================================================

def compute_bboxes_manually(segmentation_map, asset):
    """
    Calcola manualmente i bounding box per un asset
    dalla mappa di segmentazione.
    
    Args:
        segmentation_map (np.ndarray): L'array di segmentazione (num_frames, H, W, 1).
        asset (kb.Asset): L'asset per cui calcolare i BBox.

    Returns:
        np.ndarray: Un array (num_frames, 4) di BBox normalizzati [ymin, xmin, ymax, xmax].
    """
    seg_id = asset.segmentation_id
    
    # --- FIX: Gestione della 4a Dimensione (Canale) ---
    # La mappa di Kubric ha forma (num_frames, H, W, 1)
    # Dobbiamo rimuovere l'ultima dimensione (canale)
    if segmentation_map.ndim == 4:
        # Se l'ultima dimensione è 1, la rimuoviamo
        if segmentation_map.shape[3] == 1:
            segmentation_map = segmentation_map.squeeze(axis=3)
        else:
            # Questo non dovrebbe accadere
            raise ValueError(f"La mappa di segmentazione ha una forma inaspettata: {segmentation_map.shape}")
            
    # Ora la forma è (num_frames, H, W), come previsto
    num_frames, height, width = segmentation_map.shape
    # --- FINE FIX ---
    
    all_bboxes = np.zeros((num_frames, 4), dtype=np.float32)

    for frame_idx in range(num_frames):
        # 1. Trova le coordinate (ora segmentation_map[frame_idx] è 2D)
        rows, cols = np.where(segmentation_map[frame_idx] == seg_id)
        
        # 2. Se l'oggetto non è nel frame, il BBox resta [0.0, 0.0, 0.0, 0.0]
        if rows.size > 0:
            # 3. Trova i bordi (min/max)
            ymin = np.min(rows)
            xmin = np.min(cols)
            ymax = np.max(rows) + 1 
            xmax = np.max(cols) + 1
            
            # 4. Normalizza le coordinate
            all_bboxes[frame_idx] = [
                float(ymin) / height,
                float(xmin) / width,
                float(ymax) / height,
                float(xmax) / width
            ]

    return all_bboxes

def generate_scene_layout(FLAGS):
    """
    Usa un seed per generare una lista di oggetti con le loro proprietà 
    di "spawn" (posizione iniziale, scala, ecc.). 
    NON esegue la simulazione di assestamento.
    """
    print(f"🔩 Generazione della lista di spawn con seed {FLAGS.seed}...")
    rng = np.random.RandomState(FLAGS.seed)

    temp_scene, _, _, _ = kb.setup(FLAGS)
    simulator = KubricSimulator(temp_scene)
    
    dome = KUBASIC_SOURCE.create(asset_id="dome", name="dome", static=True, background=True)
    temp_scene += dome
    
    layout_data = []
    current_segmentation_id = 1

    # === 1. Posizionamento Iniziale Oggetti Statici ===
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

                print(f"    AVVISO MASSA: Tentativo {attempt + 1}/{max_retries} fallito. L'oggetto {shape_id} ha una mesh problematica. Riprovo con un altro oggetto.")
                continue  
            scale = rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            
            temp_scene += obj
            
            try:
                kb.move_until_no_overlap(
                    obj, 
                    simulator, 
                    spawn_region=FLAGS.spawning_region_static, 
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
                print(f"    AVVISO POSIZIONAMENTO: Tentativo {attempt + 1}/{max_retries} fallito per l'oggetto {shape_id}.")
                temp_scene.remove(obj)
        
        if not oggetto_posizionato_con_successo:
            print(f"    AVVISO FINALE: Impossibile posizionare un oggetto statico nello slot #{idx+1} dopo {max_retries} tentativi.")

    # === 2. Posizionamento Oggetti Dinamici con Logica di Retry ===
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
                print(f"    AVVISO MASSA: Tentativo {attempt + 1}/{max_retries} fallito. L'oggetto {shape_id} ha una mesh problematica. Riprovo con un altro oggetto.")
                continue 
            scale = rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            
            temp_scene += obj
            try:
                kb.move_until_no_overlap(
                    obj,
                    simulator,
                    spawn_region=FLAGS.spawning_region_dynamic,
                    max_trials=200,
                    rng=rng
                )

                velocity = (rng.uniform(*FLAGS.velocity_range) - [obj.position[0], obj.position[1], 0])
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
                print(f"    AVVISO POSIZIONAMENTO: Tentativo {attempt + 1}/{max_retries} fallito per l'oggetto dinamico {shape_id}.")
                temp_scene.remove(obj)

        if not oggetto_posizionato_con_successo:
            print(f"    AVVISO FINALE: Impossibile posizionare un oggetto dinamico nello slot #{idx+1} dopo {max_retries} tentativi.")
            
    print(f"  -> Lista di spawn per {len(layout_data)} oggetti creata con successo.")
    del temp_scene, simulator
    gc.collect()
    return layout_data

def render_variation(layout_data: list, light_intensity: float, light_color: tuple, light_orientation: tuple, camera_position: tuple, camera_mode:str, FLAGS, output_root: Path = Path("output")):
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
    LIGHT_SOURCE_GAMMA = 1.8
    light_source_intensity = light_intensity ** LIGHT_SOURCE_GAMMA
    BACKGROUND_GAMMA = 1.8
    background_visual_intensity = light_intensity ** BACKGROUND_GAMMA
    AMBIENT_LIGHT_FACTOR = 0.1
    print(f"INFO: Intensità luce: {light_source_intensity:.4f} | Intensità sfondo: {background_visual_intensity:.4f} | Luce ambiente: {AMBIENT_LIGHT_FACTOR}")

    # --- Luce ambientale del Mondo di Blender ---
    light_color = get_light_color_by_intensity(light_intensity) if light_color is None else light_color
    world_background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_background_node:
        # Usiamo direttamente la variabile light_color, che dovrebbe essere un RGBA
        world_background_node.inputs["Color"].default_value = light_color
        world_background_node.inputs["Strength"].default_value = AMBIENT_LIGHT_FACTOR * light_intensity
        print("INFO: Luce ambientale del Mondo configurata con colore personalizzato.")

    # --- Aggiungiamo un Sole ---
    SUN_BASE_INTENSITY = 0.25
    sun_pos = get_light_direction(light_intensity, rng) if light_orientation is None else light_orientation

    sun = kb.DirectionalLight(
        name="sun",
        position=sun_pos,
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
    max_camera_speed = FLAGS.max_camera_movement
    print(f"🎥 Setting up Camera in '{camera_mode}' mode...")

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
    print("🎥 Rendering...")
    frames_dict = renderer.render()

    # 5. POST-PROCESSING E SALVATAGGIO (Versione con Calcolo Manuale)
    kb.compute_visibility(frames_dict["segmentation"], scene.assets)
    
    # 1. Filtra gli asset visibili (Corretto)
    visible_foreground_assets = [asset for asset in scene.foreground_assets 
                                 if np.max(asset.metadata["visibility"]) > 0]

    # 2. Aggiusta la mappa di segmentazione (FONDAMENTALE)
    #    Questo garantisce che gli ID sulla mappa (es. [1, 3, 4])
    #    corrispondano agli ID sugli asset (es. [1, 2, 3]).
    print("INFO: Riassegnamento ID di segmentazione contigui...")
    frames_dict["segmentation"] = kb.adjust_segmentation_idxs(
        frames_dict["segmentation"],
        scene.assets,
        visible_foreground_assets
    ).astype(np.uint8)

    # 3. CALCOLO MANUALE (Sostituisce compute_bboxes)
    #    Iteriamo sugli asset visibili e calcoliamo i loro BBox
    #    dalla mappa di segmentazione pulita.
    print("INFO: Calcolo manuale dei Bounding Box...")
    
    segmentation_map = frames_dict["segmentation"]
    
    for asset in visible_foreground_assets:
        # Chiama la nostra nuova funzione
        bboxes_array = compute_bboxes_manually(segmentation_map, asset)
        
        # Salva i BBox densi e corretti nei metadati
        asset.metadata["bboxes"] = bboxes_array

    # 4. Rimuovi i BBox "sparsi" che potrebbero essere stati
    #    lasciati da una chiamata precedente (per pulizia)
    for asset in visible_foreground_assets:
        if "bbox_frames" in asset.metadata:
            del asset.metadata["bbox_frames"]
        

    # Salvataggio frame
    seq_name = get_sequence_name()
    print(f"💾 Salvataggio frame per {seq_name}...")
    for key in tqdm(frames_dict.keys(), desc=f"Scrittura Frame {seq_name}", unit="tipo"):
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
            "object": kb.get_instance_info(scene, visible_foreground_assets) # <-- USA QUESTA LISTA
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
# --- RUN MODES ---
# ============================================================

def standard_run_mode(num_sequences, output_root, args, coco_data, annotation_id_counter, image_id_counter):
    
    for seq_batch in range(num_sequences):
        # 1. Imposta il SEED una sola volta per l'intero batch
        seed = get_seed()
        args.seed = seed
        print(f"\n🌟 Inizio sequence batch {seq_batch + 1}/{num_sequences} con SEED {seed}")
        
        # 2. GENERA IL LAYOUT CASUALE UNA SOLA VOLTA USANDO IL SEED
        layout_data = generate_scene_layout(args)

        # 3. Ora cicla sulle variazioni, che sono PURAMENTE DETERMINISTICHE
        for intensity in args.light_levels:
                # Camera position is only cycled if camera mode is fixed
                    for cam_mode in args.camera_modes:
                        if cam_mode == "fixed":
                            for cam_pos in [camera_positions_all[name] for name in args.camera_positions]:
                                print(f"\n🚀 Generazione sequenza | batch {seq_batch + 1} | seed {args.seed} | light={int(intensity*100)}% | camera_mode={cam_mode}")
                                kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=intensity, light_color=None, light_orientation=None, camera_position=cam_pos, camera_mode=cam_mode, FLAGS=args, output_root=output_root)
                                # 4. Aggiorna il dizionario COCO con i nuovi dati
                                coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
                        else:
                            # For non-fixed camera modes, don't cycle through camera positions
                            print(f"\n🚀 Generazione sequenza | batch {seq_batch + 1} | seed {args.seed} | light={int(intensity*100)}% | camera_mode={cam_mode}")
                            kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=intensity, light_color=None, light_orientation=None, camera_position=None, camera_mode=cam_mode, FLAGS=args, output_root=output_root)
                            # 4. Aggiorna il dizionario COCO con i nuovi dati
                            coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
  
    
    return coco_data, annotation_id_counter, image_id_counter

def total_random_run_mode(args, output_root, coco_data, annotation_id_counter, image_id_counter):
    #Fully random setup for each sequence
    print(f"Random run mode")
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
        print(f"\n🚀 Generazione sequenza | seed {args.seed} | light={int(light_intensity*100)}%| color={light_color} | camera_mode={camera_mode}")
        kubric_metadata, seq_name = render_variation(layout_data=layout_data, light_intensity=light_intensity, light_color=light_color, light_orientation=light_orientation, camera_position=camera_position, camera_mode=camera_mode, FLAGS=args, output_root=output_root)
        # 4. Aggiorna il dizionario COCO con i nuovi dati
        coco_data, annotation_id_counter, image_id_counter = update_coco_from_metadata(coco_data, kubric_metadata, seq_name, annotation_id_counter, image_id_counter)
    return coco_data, annotation_id_counter, image_id_counter

# ============================================================
# --- MAIN ---
# ============================================================

def main():
    args = parse_args()
    print("🎛️  Configurazione in corso...")
    print("number of sequences:", args.refresh_item_number)
    print("Light levels:", args.light_levels)
    print("Cameras:", args.camera_positions)
    print("Output root:", args.output_root)
    print("Camera modes:", args.camera_modes)
    print("Standard mode:", args.standard_mode)
    print("Random generation mode:", args.rand_gen)
    print("Resolution:", args.resolution)
    print("Spawn regions:", args.spawning_region_static, args.spawning_region_dynamic)
    print("Velocity range:", args.velocity_range)


    num_refresh = args.refresh_item_number

    # Use output_root from arguments
    output_root = args.output_root
    coco_data = initialize_coco_dict()
    annotation_id_counter = 1
    image_id_counter = 1
    print("INFO: Inizializzazione dell'ambiente Kubric (Bootstrap)...")
    kb.setup(args) 
    clean_blender_scene()
    if args.standard_mode:
        print("Standard mode selected.....")
        coco_data, annotation_id_counter, image_id_counter = standard_run_mode(num_refresh, output_root, args, coco_data, annotation_id_counter, image_id_counter)
    if args.rand_gen:
        print("Total random mode run.....")
        coco_data, annotation_id_counter, image_id_counter= total_random_run_mode(args, output_root, coco_data, annotation_id_counter, image_id_counter)
    annotations_path = output_root / "annotations.json"
    kb.file_io.write_json(filename=annotations_path, data=coco_data)
    print(f"\n✅ Annotazioni COCO salvate in: '{annotations_path}'")
    print("\n✅ Tutte le sequenze sono state generate.")

    kb.done()


if __name__ == "__main__":
    main()