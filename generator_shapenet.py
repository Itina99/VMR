import gc
import logging
import os
import re
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
import bmesh
from HDRISelector import HDRISelector
from mathutils import Vector




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
RESOLUTION = (256, 256)
FRAME_END = 24
FRAME_RATE = 12
STEP_RATE = 240
MIN_STATIC, MAX_STATIC = 1, 3
MIN_DYNAMIC, MAX_DYNAMIC = 1, 4
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

    parser.add_argument("--classes", nargs="+", default=["airplane", "display", "earphone", "faucet", "microphone"])
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

    return parser.parse_args()


# ============================================================
# --- FUNZIONE DI GENERAZIONE SEQUENZA ---
# ============================================================

def generate_sequence(seq_id: int, shape_ids: list, light_intensity: float, orientation: tuple, camera_position: tuple, light_color: tuple, FLAGS, output_root: Path = Path("output")):

    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

    renderer = KubricBlender(scene, use_denoising=True, samples_per_pixel=64)
    simulator = KubricSimulator(scene)    

    #--- Scene background HDRI ---
    for _ in range(3):  # Scegli 3 HDRI diversi
        hdri_id = selector.pick(light_intensity)
        print(f"🌅 Tentativo HDRI: {hdri_id}")
    #hdri_id = rng.choice(list(HDRI_SOURCE._assets.keys()))
    #hdri_id = selector.pick(light_intensity)
    hdri_id = "muddy_autumn_forest"
    print(f"🌅 Using HDRI: {hdri_id}")
    background_hdri = HDRI_SOURCE.create(asset_id=hdri_id)
    #assert isinstance(background_hdri, kb.Texture)


    """     # HDRI per l’illuminazione globale (luce)
        renderer._set_ambient_light_hdri(
            background_hdri.filename,
            # hdri_rotation=orientation,
            #strength=light_intensity
        )
        renderer._set_ambient_light_color(light_color)
        # --- Dome di Kubasic ---
        dome = KUBASIC_SOURCE.create(
            asset_id="dome",
            friction=FLAGS.friction,
            restitution=FLAGS.restitution,
            static=True,
            background=True
        )
        scene += dome
        

        # Oggetto Blender del dome
        dome_blender = dome.linked_objects[renderer]
        dome_blender = dome.linked_objects[renderer]
        texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
        texture_node.image = bpy.data.images.load(background_hdri.filename)  """

    scene.metadata["background"] = hdri_id

    # --- Usa la tua variabile 'light_intensity'
    logging.info(f"Using light intensity: {light_intensity:.2f}")

    # --- NUOVO: Definiamo due scale di intensità separate ---
    # 1. Per le fonti di luce (il sole). Usiamo una gamma alta per un calo drastico.
    LIGHT_SOURCE_GAMMA = 2.5 # Prova a giocare con questo valore (es. 2.2, 2.5, 3.0)
    light_source_intensity = light_intensity ** LIGHT_SOURCE_GAMMA

    # 2. Per la luminosità VISIVA dello sfondo. Usiamo una scala lineare o una gamma leggera.
    BACKGROUND_GAMMA = 1.0 # Prova con 1.0 (lineare) o un valore basso come 1.5
    background_visual_intensity = light_intensity ** BACKGROUND_GAMMA

    print(f"INFO: Intensità luce scalata: {light_source_intensity:.4f} | Intensità sfondo scalata: {background_visual_intensity:.4f}")

    # --- Azzeriamo la luce del "Mondo" di Blender
    world_background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_background_node:
        world_background_node.inputs["Color"].default_value = (0, 0, 0, 1)

    # --- Aggiungiamo un Sole, controllato dalla SCALA PER LE LUCI ---
    # Puoi giocare con l'intensità di base (es. 3.0, 4.0, 5.0) per rendere le ombre più o meno marcate.
    SUN_BASE_INTENSITY = 2.5
    sun = kb.DirectionalLight(name="sun", position=(-1, -1, 3.0), look_at=(0, 0, 0), intensity=SUN_BASE_INTENSITY)
    sun.intensity *= light_source_intensity
    scene.add(sun)

    # --- Dome (usando KUBASIC_SOURCE)
    dome = KUBASIC_SOURCE.create(asset_id="dome", name="dome", static=True, background=True)
    scene += dome
    dome_blender = dome.linked_objects[renderer]

    # Applica la texture al dome come sempre
    texture_node_ref = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node_ref.image = bpy.data.images.load(background_hdri.filename)

    # --- Manipolazione del materiale del dome, controllata dalla SCALA PER LO SFONDO ---
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
            # Usiamo l'intensità specifica per la visualizzazione dello sfondo
            bg_vec = (background_visual_intensity, background_visual_intensity, background_visual_intensity)
            color_dimmer_node.inputs[1].default_value = bg_vec

            node_tree.links.new(texture_node.outputs["Color"], color_dimmer_node.inputs[0])

            for dest_socket in original_destinations:
                node_tree.links.new(color_dimmer_node.outputs["Vector"], dest_socket)




# --- Camera ---
    scene.camera = kb.PerspectiveCamera(name="camera", focal_length=35., sensor_width=32)
    scene.camera.position = camera_position
    scene.camera.look_at((0, 0, 0))


    

    # === STATIC OBJECTS ===
    num_static = rng.randint(MIN_STATIC, MAX_STATIC + 1)
    print(f"📦 Generating {num_static} static objects...")
    for _ in range(num_static):
        shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])  # Normalize scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_STATIC, rng=rng)
        print(f"📦 Static object {shape_id} at {obj.position}")

    print("Simulating to let objects settle...")
    _, _ = simulator.run(frame_start=-100, frame_end=0)

    print("Stopping any moving objects...")
    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.friction = 0.5
            obj.restitution = 0.5

    # === DYNAMIC OBJECTS ===
    num_dynamic = rng.randint(MIN_DYNAMIC, MAX_DYNAMIC + 1)
    print(f"🚀 Generating {num_dynamic} dynamic objects...")
    for _ in range(num_dynamic):
        shape_id = rng.choice(shape_ids)
        obj = ASSET_SOURCE.create(shape_id)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION_DYNAMIC, rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
        print(f"🚀 Dynamic object {shape_id} with velocity {obj.velocity}")

    # === Simulation ===
    print("🎬 Simulazione...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)


    # === Rendering ===
    print("Saving state...")
    renderer.save_state(output_root / f"states/seq{seq_id}.blend")
    print("🎥 Rendering...")
    frames_dict = renderer.render()
    #frames_dict = renderer.render_still() # just one frame for testing

    # === Post-processing ===

    print("🎞️ Post-processing...")

    # --- Calcola visibilità e aggiusta segmentation ---
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
    print("🎛️  Configurazione in corso...")
    print("Classes:", args.classes)
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


    # -- classes
    raw_classes = _normalize_list_arg(args.classes)
    classes = raw_classes  # già lista di stringhe singole

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
    print("Classes:", classes)
    print("Light levels:", light_levels)
    print("Orientations:", light_orientations)
    print("Cameras:", camera_positions)
    print("Colors:", light_colors)

    
    # Use output_root from arguments
    output_root = args.output_root
    
    seq_id = 0
    
    for shape_class in classes:
        shape_ids = chooseClass(shape_class)
        
        for intensity in light_levels:
            for orient_name, orientation in light_orientations.items():
                for cam_name, cam_pos in camera_positions.items():
                    for color_name, color_value in light_colors.items():
                        print(f"\n🚀 Generazione sequenza {seq_id} | shape={shape_class} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
                        generate_sequence(seq_id, shape_ids, intensity, orientation, cam_pos, color_value, args, output_root)
                        seq_id += 1
    print("\n✅ Tutte le sequenze sono state generate.")

    # Generate additional sequences with multiple objects and random parameters if enabled
    if args.rand_gen:
        print(f"\n🎲 Generating additional sequences with random multiple objects...")

        # Modified parameters for multiple objects
        global MIN_STATIC, MAX_STATIC, MIN_DYNAMIC, MAX_DYNAMIC
        MIN_STATIC, MAX_STATIC = 1, 2
        MIN_DYNAMIC, MAX_DYNAMIC = 1, 2
        
        # Generate 10 additional sequences with random parameters
        for i in range(1):
            # Random shape selection
            random_class = Random.choice(classes_all)
            shape_ids = chooseClass(random_class)
            
            # Random light parameters
            intensity = Random.choice(light_levels_all)
            orient_name, orientation = Random.choice(list(light_orientations_all.items()))
            cam_name, cam_pos = Random.choice(list(camera_positions_all.items()))
            color_name, color_value = Random.choice(list(light_colors_all.items()))

            print(f"\n🎲 Random sequence {seq_id} | shape={random_class} | light={int(intensity*100)}% | orient={orient_name} | cam={cam_name} | color={color_name}")
            generate_sequence(seq_id, shape_ids, intensity, orientation, cam_pos, color_value, args, output_root)
            seq_id += 1

    kb.done()


if __name__ == "__main__":
    main()
