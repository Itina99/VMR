#!/bin/bash

#!/bin/bash

# ========== CONFIGURAZIONE ==========
SIMULATION_TYPE="shapenet"    # oppure "gso"
OUTPUT_DIR="output"
UPSAMPLED_DIR="$OUTPUT_DIR/upsampled_rgb"
EVENTS_DIR="$OUTPUT_DIR/events"

# elenco delle classi ShapeNet
CLASSES=("airplane" "ashcan" "bag" "basket" "bathtub" "bed" "bench" "birdhouse" "bookshelf" "bottle" "bowl" "bus" "cabinet" "camera" "can" "cap" "car" "cellular telephone" "chair" "clock" "computer keyboard" "dishwasher" "display" "earphone" "faucet" "file" "guitar" "helmet" "jar" "knife" "lamp" "laptop" "loudspeaker" "mailbox" "microphone" "microwave" "motorcycle" "mug" "piano" "pillow" "pistol" "pot" "printer" "remote control" "rifle" "rocket" "skateboard" "sofa" "stove" "table" "telephone" "tower" "train" "vessel" "washer")

# livelli di intensità luce
LIGHT_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# orientamenti luce (nome:x,y,z in radianti)
LIGHT_ORIENTATIONS=(
    "front:0.0,0.0,0.0"
    "side_45:0.0,0.0,0.785398"
    "side_90:0.0,0.0,1.570796"
    "back_135:0.0,0.0,2.356194"
    "top:1.570796,0.0,0.0"
    "bottom:-1.570796,0.0,0.0"
)

# colori luce (RGBA)
LIGHT_COLORS=(
  "white:1.0,1.0,1.0,1.0"
  "red:1.0,0.0,0.0,1.0"
  "green:0.0,1.0,0.0,1.0"
  "blue:0.0,0.0,1.0,1.0"
  "yellow:1.0,1.0,0.0,1.0"
  "purple:0.5,0.0,0.5,1.0"
  "cyan:0.0,1.0,1.0,1.0"
  "orange:1.0,0.5,0.0,1.0"
)

# posizioni camera (nome:x,y,z)
CAMERA_POSITIONS=(
  "front:0,-8,0"
  "tilt_30:4,-7,3"
  "tilt_60:7,-4,5"
  "side_90:8,0,0"
  "retro_120:7,4,3"
  "back_180:0,8,0"
  "top:0,0,8"
  "bottom:0,0,-8"
)


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
    /usr/bin/python3 generator_shapenet.py 
fi

# ========== 2. CLEANUP OUTPUT ==========
if [ -d "$UPSAMPLED_DIR" ]; then
  echo "🧹 Pulizia directory esistente: $UPSAMPLED_DIR"
  rm -rf "$UPSAMPLED_DIR"
fi

# ========== 3. UPSAMPLING ==========
echo "📈 Avvio upsampling..."
python3 upsampler.py --input_dir "$OUTPUT_DIR/rgb" --output_dir "$UPSAMPLED_DIR"

# ========== 4. EVENT GENERATION ==========
echo "⚡ Generazione eventi..."
python3 event_generator.py --input_dir "$UPSAMPLED_DIR" --output_dir "$EVENTS_DIR"

echo "✅ Pipeline completata!"
