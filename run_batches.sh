#!/bin/bash
set -e

# --- Configurazione ---
CONFIG_FILE="${1:-config.json}"       
TOTAL_BATCHES="${2:-5}" 
PIPELINE_SCRIPT="./pipeline.sh"  
BASE_OUTPUT_DIR="output"         

echo "🚀 Inizio generazione totale del dataset in $TOTAL_BATCHES batch..."
echo "   Usando il file di configurazione: $CONFIG_FILE"
echo "===================================================="

# Esegui un ciclo 'for' da 1 fino a TOTAL_BATCHES
for (( i=1; i<=$TOTAL_BATCHES; i++ ))
do
    # --- MODIFICA CHIAVE ---
    # Definisce un nome di cartella unico per questo batch
    BATCH_OUTPUT_DIR="${BASE_OUTPUT_DIR}_batch_${i}"
    # -----------------------

    echo "🔥 Avvio Batch $i / $TOTAL_BATCHES..."
    echo "   I file di output saranno salvati in: $BATCH_OUTPUT_DIR"
    
    # Crea la directory di output specifica per il batch (non dà errore se esiste già)
    mkdir -p "$BATCH_OUTPUT_DIR"

    # Esegui la pipeline, passando il config file e il NUOVO percorso di output
    $PIPELINE_SCRIPT "$CONFIG_FILE" "$BATCH_OUTPUT_DIR"
    
    # Controlla se lo script precedente è fallito
    if [ $? -ne 0 ]; then
        echo "❌ ERRORE: Il Batch $i è fallito. Interruzione dello script."
        exit 1 # Esce dallo script con un codice di errore
    fi
    
    echo "✅ Batch $i completato con successo. Dati salvati in $BATCH_OUTPUT_DIR"
    echo "----------------------------------------------------"
done

echo "🎉 Tutti i $TOTAL_BATCHES batch sono stati generati."