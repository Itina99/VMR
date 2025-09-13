import cv2
import numpy as np
import glob
import os

base_folder = "output/upsampled_rgb"  # cartella principale con seq0, seq1, seq2, ...

# --- Parametri più restrittivi ---
MAD_FACTOR = 1.0  # <--- abbassa a 0.5 per essere ancora più severo

for seq_folder in sorted(os.listdir(base_folder)):
    seq_path = os.path.join(base_folder, seq_folder)
    if not os.path.isdir(seq_path):
        continue

    imgs_folder = os.path.join(seq_path, "imgs")
    timestamps_file = os.path.join(seq_path, "timestamps.txt")

    if not os.path.exists(imgs_folder) or not os.path.exists(timestamps_file):
        continue

    print(f"\n--- Elaborazione {seq_folder} ---")

    # Carica immagini e timestamp
    image_files = sorted(glob.glob(os.path.join(imgs_folder, "*.png")))
    if not image_files:
        print("  Nessuna immagine trovata, salto.")
        continue

    with open(timestamps_file, "r") as f:
        timestamps = [float(line.strip()) for line in f if line.strip()]

    if len(timestamps) != len(image_files):
        print("  ⚠️ Numero timestamp diverso dal numero di frame, salto questa sequenza.")
        continue

    # Calcolo luminosità media
    mean_intensities = []
    for f in image_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mean_intensity = np.mean(img)
        mean_intensities.append(mean_intensity)

    mean_intensities = np.array(mean_intensities)
    diffs = np.diff(mean_intensities)

    # --- Soglia basata su mediana + MAD ---
    baseline = np.median(diffs)
    mad = np.median(np.abs(diffs - baseline))
    threshold = baseline + MAD_FACTOR * mad

    print(f"  baseline={baseline:.4f}, mad={mad:.4f}, soglia={threshold:.4f}")

    # Determina quali frame tenere
    keep_mask = [True]  # primo frame sempre tenuto
    for d in diffs:
        keep_mask.append(False if d > threshold else True)

    # Elimina i frame fuori soglia
    filtered_timestamps = []
    removed_count = 0
    for f, ts, keep in zip(image_files, timestamps, keep_mask):
        if keep:
            filtered_timestamps.append(ts)
        else:
            os.remove(f)
            removed_count += 1

    # Aggiorna file dei timestamp
    with open(timestamps_file, "w") as f:
        for ts in filtered_timestamps:
            f.write(f"{ts}\n")

    print(f"  Rimossi {removed_count} frame su {len(image_files)} totali.")

print("\n✅ Tutte le sequenze sono state filtrate con soglia restrittiva.")
