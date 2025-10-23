#!/bin/bash
set -e

# --- Configuration ---
CONFIG_FILE="${1:-config.json}"       
TOTAL_BATCHES="${2:-5}" 
PIPELINE_SCRIPT="./pipeline.sh"  
BASE_OUTPUT_DIR="output"         

echo "🚀 Starting total dataset generation in $TOTAL_BATCHES batches..."
echo "   Using configuration file: $CONFIG_FILE"
echo "===================================================="


for (( i=1; i<=$TOTAL_BATCHES; i++ ))
do

    BATCH_OUTPUT_DIR="${BASE_OUTPUT_DIR}_batch_${i}"
    # -----------------------

    echo "🔥 Starting Batch $i / $TOTAL_BATCHES..."
    echo "   Output files will be saved in: $BATCH_OUTPUT_DIR"
    

    mkdir -p "$BATCH_OUTPUT_DIR"

    $PIPELINE_SCRIPT "$CONFIG_FILE" "$BATCH_OUTPUT_DIR"
    
   
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Batch $i failed. Stopping script."
        exit 1 
    fi

    echo "✅ Batch $i completed successfully. Data saved in $BATCH_OUTPUT_DIR"
    echo "----------------------------------------------------"
done

echo "🎉 All $TOTAL_BATCHES batches have been generated."