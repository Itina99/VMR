import json
import os
from pathlib import Path

import kubric as kb
from kubric.assets.asset_source import AssetSource
from kubric.simulator.pybullet import PyBullet as p
import numpy as np

# === CONFIG ===
GSO_MANIFEST_URL = "gs://kubric-public/assets/GSO/GSO.json"
OUTPUT_DIR = Path("GSO_preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CARICA ASSET SOURCE ORIGINALE ===
asset_source = AssetSource.from_manifest(GSO_MANIFEST_URL)
print(f"Trovati {len(asset_source._assets)} asset GSO originali")

new_manifest = {}

# === INIZIALIZZA PYBULLET PER CALCOLARE COLLIDER ===
p.connect(p.DIRECT)

for i, shape_id in enumerate(asset_source._assets.keys()):
    print(f"[{i+1}/{len(asset_source._assets)}] Processing {shape_id}...")

    # 1. Crea oggetto Kubric
    obj = asset_source.create(shape_id, scale=1.0)

    # 2. Scarica localmente il mesh (se non è già in locale)
    mesh_path = Path(asset_source._assets[shape_id]["uri"])
    if not mesh_path.exists():
        mesh_path = kb.download(asset_source._assets[shape_id]["uri"], OUTPUT_DIR / f"{shape_id}.glb")

    # 3. Calcola bounding box / collider in PyBullet
    collision_shape_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=str(mesh_path),
        meshScale=[1, 1, 1]
    )
    aabb_min, aabb_max = p.getAABB(collision_shape_id)
    bbox_size = np.array(aabb_max) - np.array(aabb_min)

    # 4. Assegna proprietà fisiche
    mass = float(np.prod(bbox_size) * 0.25)  # massa proporzionale al volume
    friction = 0.5
    restitution = 0.5

    # 5. Salva entry nel nuovo manifest
    new_manifest[shape_id] = {
        "uri": str(mesh_path.resolve()),
        "scale": [1.0, 1.0, 1.0],
        "mass": mass,
        "friction": friction,
        "restitution": restitution,
        "bbox": {
            "min": list(aabb_min),
            "max": list(aabb_max)
        }
    }

# === SALVA MANIFEST ===
manifest_path = OUTPUT_DIR / "GSO_preprocessed.json"
with open(manifest_path, "w") as f:
    json.dump(new_manifest, f, indent=2)

print(f"Manifest preprocessato salvato in: {manifest_path}")
p.disconnect()
