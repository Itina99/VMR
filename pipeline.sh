#!/bin/bash

#!/bin/bash

# ========== CONFIGURAZIONE ==========
SIMULATION_TYPE="shapenet"    # oppure "gso"
OUTPUT_DIR="output"
UPSAMPLED_DIR="$OUTPUT_DIR/upsampled_rgb"
EVENTS_DIR="$OUTPUT_DIR/events"

# Carica configurazioni da file JSON
CONFIG_FILE="${1:-config.json}"
if [ -f "$CONFIG_FILE" ]; then
    SHAPENET_CLASSES=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(config.get('shapenet_classes', [])))")
    SIMULATION_TYPE=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config.get('simulation_type', 'shapenet'))")
    OUTPUT_DIR=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config.get('output_dir', 'output'))")
    LIGHT_LEVELS=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(map(str, config.get('light_levels', []))))")
    LIGHT_COLORS=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(config.get('light_colors', [])))")
    CAMERA_POSITIONS=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(config.get('camera_positions', [])))")
    LIGHT_ORIENTATIONS=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(config.get('light_orientations', [])))")
    echo "📋 Configurazioni caricate da $CONFIG_FILE"
else
    echo "⚠️  File di configurazione $CONFIG_FILE non trovato, uso valori di default"
    SHAPENET_CLASSES="chair table lamp"
    LIGHT_LEVELS="0.5 1.0 1.5"
    LIGHT_COLORS="white warm cool"
    CAMERA_POSITIONS="front side top"
    LIGHT_ORIENTATIONS="top side diagonal"
fi


# Utente e gruppo corrente (per Docker)
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CURRENT_DIR=$(pwd)

# ========== 1. SIMULAZIONE ==========
echo "🚀 Avvio simulazione Kubric ($SIMULATION_TYPE)..."
if [ "$SIMULATION_TYPE" = "shapenet" ]; then
    docker run --rm -it \
        --user ${USER_ID}:${GROUP_ID} \
        --volume ${CURRENT_DIR}:/kubric \
        kubricdockerhub/kubruntu \
        /usr/bin/python3 generator_shapenet.py \
        --shapenet_classes "$SHAPENET_CLASSES" \
        --output_dir "$OUTPUT_DIR" \
        --light_levels "$LIGHT_LEVELS" \
        --light_colors "$LIGHT_COLORS" \
        --camera_positions "$CAMERA_POSITIONS" \
        --light_orientations "$LIGHT_ORIENTATIONS"
fi

# ========== 2. CLEANUP OUTPUT ==========
if [ -d "$UPSAMPLED_DIR" ]; then
  echo "🧹 Pulizia directory esistente: $UPSAMPLED_DIR"
  rm -rf "$UPSAMPLED_DIR"
fi

# ========== 3. UPSAMPLING AND EVENT GENERATION ==========

echo "⚡ Generazione eventi e upsampling..."
python3 event_generation.py --input_dir "$UPSAMPLED_DIR" --output_dir "$EVENTS_DIR"

echo "✅ Pipeline completata!"
