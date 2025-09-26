import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import shutil

def replace_background(rgb_frame, segmentation_mask, color=[128, 128, 128]):
    """
    Sostituisce lo sfondo di un'immagine con un colore solido.

    Args:
        rgb_frame (np.array): L'immagine a colori (H, W, 3).
        segmentation_mask (np.array): La maschera di segmentazione (H, W).
        color (list): La lista di valori [B, G, R] per il nuovo sfondo.

    Returns:
        np.array: L'immagine con lo sfondo sostituito.
    """
    # Creiamo una maschera binaria: True dove c'è il foreground.
    # Assumiamo che il foreground abbia ID > 0.
    foreground_mask = (segmentation_mask > 0)

    # Creiamo un'immagine di sfondo completamente del colore scelto
    background_frame = np.full(rgb_frame.shape, color, dtype=np.uint8)
    
    # Usiamo la maschera per "incollare" il foreground originale
    # sopra il nostro nuovo sfondo a colore solido.
    # np.where(condizione, valore_se_vero, valore_se_falso)
    final_frame = np.where(np.stack([foreground_mask]*3, axis=-1), 
                           rgb_frame, 
                           background_frame)
    
    return final_frame

def process_sequences(rgb_root, seg_root, output_root, bg_color):
    """
    Scansiona tutte le sequenze, sostituisce lo sfondo e le salva.
    """
    print(f"📁 Scansione delle sequenze in: '{rgb_root}'")
    try:
        sequence_dirs = sorted([d for d in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, d))])
    except FileNotFoundError:
        print(f"❌ ERRORE: La cartella di input '{rgb_root}' non è stata trovata.")
        return

    if not sequence_dirs:
        print("⚠️ Nessuna sottocartella di sequenze trovata.")
        return

    print(f"✅ Trovate {len(sequence_dirs)} sequenze. Avvio del processo...")

    for seq_name in tqdm(sequence_dirs, desc="Sequenze Processate"):
        input_rgb_dir = os.path.join(rgb_root, seq_name, "imgs")
        input_seg_dir = os.path.join(seg_root, seq_name, "imgs")
        output_seq_dir = os.path.join(output_root, seq_name, "imgs")
        os.makedirs(output_seq_dir, exist_ok=True)
        
        source_fps_path = os.path.join(rgb_root, seq_name, "fps.txt")
        dest_seq_folder = os.path.join(output_root, seq_name)
        if os.path.exists(source_fps_path):
            shutil.copy2(source_fps_path, dest_seq_folder)

        if not os.path.isdir(input_seg_dir):
            print(f"\n⚠️ Attenzione: Cartella segmentazione non trovata per '{seq_name}'. Sequenza saltata.")
            continue

        rgb_files = sorted(glob.glob(os.path.join(input_rgb_dir, "*.png")))

        for rgb_path in tqdm(rgb_files, desc=f"Processing {seq_name}", leave=False):
            frame_filename = os.path.basename(rgb_path)
            seg_filename = frame_filename.replace("rgb_", "segmentation_")
            seg_path = os.path.join(input_seg_dir, seg_filename)
            output_path = os.path.join(output_seq_dir, frame_filename)

            if not os.path.exists(seg_path):
                seg_filename_alt = f"segmentation_{frame_filename}"
                seg_path_alt = os.path.join(input_seg_dir, seg_filename_alt)
                if not os.path.exists(seg_path_alt):
                    continue
                else:
                    seg_path = seg_path_alt

            rgb_image = cv2.imread(rgb_path)
            seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            replaced_frame = replace_background(rgb_image, seg_mask, color=bg_color)
            cv2.imwrite(output_path, replaced_frame)

    print("\n🎉 Processo completato!")
    print(f"Le sequenze con sfondo sostituito sono state salvate in: '{output_root}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sostituisce lo sfondo di sequenze di immagini con un colore solido, usando le maschere di segmentazione.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Cartella principale delle sequenze RGB.")
    parser.add_argument("--seg_dir", type=str, required=True, help="Cartella principale delle sequenze di segmentazione.")
    parser.add_argument("--output_dir", type=str, required=True, help="Cartella dove salvare le nuove sequenze.")
    parser.add_argument("--color", type=int, nargs=3, default=[128, 128, 128], help="Colore di sfondo in B G R (es. --color 0 0 0 per nero). Default: grigio 128.")
    
    args = parser.parse_args()

    process_sequences(args.rgb_dir, args.seg_dir, args.output_dir, args.color)