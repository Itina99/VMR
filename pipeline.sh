#!/bin/bash
set -e

# ========== CONFIGURAZIONE ==========
SIMULATION_TYPE="shapenet"

CONFIG_FILE="${1:-config.json}"
OUTPUT_DIR="${2:-output_default}" 

RGB_DIR="$OUTPUT_DIR/rgb"
UPSAMPLED_DIR="$OUTPUT_DIR/upsampled_rgb"
EVENTS_DIR="$OUTPUT_DIR/events"
ANNOTATIONS_DIR="$OUTPUT_DIR/annotations"
GIF_DIR="$OUTPUT_DIR/gif"
COCO_FILE_NAME="annotations.json"
# -----------------------------
if [ -f "$CONFIG_FILE" ]; then
    REFRESH_ITEM_NUMBER=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('refresh_item_number', 1))")
    STANDARD_GEN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(str(c.get('standard_generation', True)).lower())")
    RAND_GEN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(str(c.get('random_generation', False)).lower())")    
    NUMBER_OF_RANDOM_SEQUENCES=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('number_of_random_sequences', 5))")    
    LIGHT_LEVELS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('light_levels', [1.0]); print(' '.join(map(str, v)))")
    CAMERA_POSITIONS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('camera_positions', ['tilt_30']); print(' '.join(v))")   
    RESOLUTION=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('resolution', '256x256'))")
    FRAME_END=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('frame_end', 100))")
    FRAME_RATE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('frame_rate', 24))")
    STEP_RATE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('step_rate', 1))")    
    CAMERA_MODE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('camera_mode', ['fixed']); print(' '.join(v))")
    MAX_CAMERA_MOVEMENT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_camera_movement', 4.0))")
    MAX_DYNAMIC_OBJECTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_dynamic_objects', 3))")
    MAX_STATIC_OBJECTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_static_objects', 3))")
    MIN_DYNAMIC_OBJECTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('min_dynamic_objects', 1))")
    MIN_STATIC_OBJECTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('min_static_objects', 1))")
    SPAWNING_REGION_STATIC=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('spawning_region_static', [[-5, -5, 0], [5, 5, 0]]); print(' '.join([str(x) for sublist in v for x in sublist]))")
    SPAWNING_REGION_DYNAMIC=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('spawning_region_dynamic', [[-5, -5, 1], [5, 5, 6]]); print(' '.join([str(x) for sublist in v for x in sublist]))")
    VELOCITY_RANGE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('velocity_range', [[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0]]); print(' '.join([str(x) for sublist in v for x in sublist]))")
echo "📋 Configurazioni caricate da $CONFIG_FILE"
else
    echo "⚠️  File di configurazione $CONFIG_FILE non trovato, uso valori di default"
    REFRESH_ITEM_NUMBER="1"
    LIGHT_LEVELS="1.0"
    CAMERA_POSITIONS="tilt_30"
    RAND_GEN="false"
    NUMBER_OF_RANDOM_SEQUENCES="5"
    STANDARD_GEN="true"
    CAMERA_MODE="fixed"
    MAX_CAMERA_MOVEMENT="4.0"
    FRAME_END="24"
    FRAME_RATE="12"
    STEP_RATE="240"
    RESOLUTION="256x256"
    MAX_DYNAMIC_OBJECTS="3"
    MAX_STATIC_OBJECTS="3"
    MIN_DYNAMIC_OBJECTS="1"
    MIN_STATIC_OBJECTS="1"
    SPAWNING_REGION_STATIC="[[-5 -5 0] [5 5 0]]"
    SPAWNING_REGION_DYNAMIC="[[-5 -5 1] [5 5 6]]"
    VELOCITY_RANGE="[[-4.0 -4.0 0.0] [4.0 4.0 0.0]]"
fi

# Utente e gruppo corrente (per Docker)
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CURRENT_DIR=$(pwd)

# ========== ATTIVAZIONE AMBIENTE CONDA ==========
echo "🔧 Attivazione ambiente conda 'vid2e'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vid2e
echo "✅ Ambiente conda 'vid2e' attivato"
echo ""

# ========== 1. SIMULAZIONE ==========
echo "🚀 Avvio simulazione Kubric ($SIMULATION_TYPE) per $OUTPUT_DIR..."

if [ "$SIMULATION_TYPE" = "shapenet" ]; then
    docker run --rm -it \
        --user ${USER_ID}:${GROUP_ID} \
        --volume ${CURRENT_DIR}:/kubric \
        -e PYTHONPATH=/kubric \
        kubricdockerhub/kubruntu \
        /usr/bin/python3 /kubric/generator_shapenet.py \
            --output_root "$OUTPUT_DIR" \
            --refresh_item_number $REFRESH_ITEM_NUMBER \
            --light_levels $LIGHT_LEVELS \
            --camera_positions $CAMERA_POSITIONS \
            --standard_mode $STANDARD_GEN \
            --rand_gen $RAND_GEN \
            --number_of_random_sequences $NUMBER_OF_RANDOM_SEQUENCES \
            --camera_mode $CAMERA_MODE \
            --max_camera_movement $MAX_CAMERA_MOVEMENT \
            --resolution $RESOLUTION \
            --frame_end $FRAME_END \
            --frame_rate $FRAME_RATE \
            --step_rate $STEP_RATE \
            --max_dynamic_objects $MAX_DYNAMIC_OBJECTS \
            --max_static_objects $MAX_STATIC_OBJECTS \
            --min_dynamic_objects $MIN_DYNAMIC_OBJECTS \
            --min_static_objects $MIN_STATIC_OBJECTS\
            --spawning_region_static $SPAWNING_REGION_STATIC \
            --spawning_region_dynamic $SPAWNING_REGION_DYNAMIC \
            --velocity_range $VELOCITY_RANGE
fi

# # ========== 2. CLEANUP OUTPUT ==========
# # Remove existing upsampled directory if it exists to ensure clean output
# if [ -d "$UPSAMPLED_DIR" ]; then
#     echo "🧹 Pulizia directory esistente: $UPSAMPLED_DIR"
#     rm -rf "$UPSAMPLED_DIR"
# fi

# # ========== 3. UPSAMPLING AND EVENT GENERATION ==========
# # Generate upsampled frames from simulation output to increase temporal resolution
# echo "⚡ Generazione eventi e upsampling..."
# python3 upsample_frames.py --input_dir "$RGB_DIR" --output_dir "$UPSAMPLED_DIR"

# conda activate vid2e
# # Generate event data from upsampled RGB frames
# echo "✅ Pulizia completata. Generazione eventi..."
# python3 event_generation.py --input_dir "$UPSAMPLED_DIR" --output_dir "$EVENTS_DIR"

# # Create GIF animations from NPZ event data for visualization
# echo "🎬 Generazione gif eventi..."
# python3 npz_to_gif.py --input_dir "$EVENTS_DIR" --output_dir "$GIF_DIR" --shape $(echo $RESOLUTION | tr 'x' ' ') --frames 120 --fps 60 --window_size 0.1 --use_accumulation

# # Pipeline execution completed successfully
# echo "✅ Pipeline per il Batch $BATCH_NUMBER completata!"
