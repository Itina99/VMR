# Carica configurazioni da file JSON
#!/bin/bash
set -e

# ========== CONFIGURAZIONE ==========
SIMULATION_TYPE="shapenet"    # oppure "gso"
OUTPUT_DIR="output"
UPSAMPLED_DIR="$OUTPUT_DIR/upsampled_rgb"
EVENTS_DIR="$OUTPUT_DIR/events"



CONFIG_FILE="${1:-config.json}"
if [ -f "$CONFIG_FILE" ]; then
    CLASSES=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('classes', []); print(' '.join(v if isinstance(v, list) else str(v).replace(',', ' ').split()))")
    LIGHT_LEVELS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('light_levels', []); v = v if isinstance(v, list) else [x for x in str(v).replace(',', ' ').split()]; print(' '.join(map(str, v)))")
    LIGHT_ORIENTATIONS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); d=c.get('light_orientations', {}); print(' '.join([f\"{k} {v[0]} {v[1]} {v[2]}\" for k,v in d.items()]))")
    CAMERA_POSITIONS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); d=c.get('camera_positions', {}); print(' '.join([f\"{k} {v[0]} {v[1]} {v[2]}\" for k,v in d.items()]))")
    LIGHT_COLORS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); d=c.get('light_colors', {}); print(' '.join([f\"{k} {v[0]} {v[1]} {v[2]} {v[3]}\" for k,v in d.items()]))")
echo "📋 Configurazioni caricate da $CONFIG_FILE"
else
    echo "⚠️  File di configurazione $CONFIG_FILE non trovato, uso valori di default"
    CLASSES="chair table lamp"
    LIGHT_LEVELS="0.5 1.0 1.5"
    LIGHT_COLORS="white 1.0 1.0 1.0 1.0 red 1.0 0.0 0.0 1.0"
    CAMERA_POSITIONS="front 0 0 5 side 5 0 0 top 0 5 5"
    LIGHT_ORIENTATIONS="top 0 0 1 side 1 0 0"
fi



# Stampa configurazioni caricate
echo "📋 Configurazioni attive:"
echo "  - Tipo simulazione: $SIMULATION_TYPE"
echo "  - Directory output: $OUTPUT_DIR"
echo "  - Classi ShapeNet: $CLASSES"
echo "  - Livelli di luce: $LIGHT_LEVELS"
echo "  - Colori luce: $LIGHT_COLORS"
echo "  - Posizioni camera: $CAMERA_POSITIONS"
echo "  - Orientamenti luce: $LIGHT_ORIENTATIONS"
echo ""

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
            --output_root "$OUTPUT_DIR" \
            --classes $CLASSES \
            --light_levels $LIGHT_LEVELS \
            --light_orientations $LIGHT_ORIENTATIONS \
            --camera_positions $CAMERA_POSITIONS \
            --light_colors $LIGHT_COLORS

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
