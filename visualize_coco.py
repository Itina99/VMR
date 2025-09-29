import cv2
import json
import os
import argparse
from tqdm import tqdm
import random

def main(args):
    # --- 1. Caricamento e indicizzazione dei dati COCO ---
    print(f"🔎 Caricamento del file di annotazioni: '{args.annotations_file}'")
    try:
        with open(args.annotations_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE: File non trovato. Assicurati che il percorso sia corretto.")
        return

    print("🔄 Indicizzazione delle annotazioni...")
    
    images_info = {img['id']: img for img in coco_data['images']}
    category_info = {cat['id']: cat for cat in coco_data['categories']}
    
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # --- LA CORREZIONE È QUI ---
    # Usiamo la variabile corretta 'cat_id' che viene dal ciclo.
    category_colors = {cat_id: [random.randint(100, 255), random.randint(100, 255), random.randint(50, 200)] for cat_id in category_info}
    # -------------------------

    # --- 2. Preparazione dei file della sequenza ---
    sequence_images = [img for img in coco_data['images'] if img['file_name'].startswith(f"{args.sequence_name}/")]
    
    if not sequence_images:
        print(f"⚠️ Attenzione: Nessuna immagine trovata per la sequenza '{args.sequence_name}' nel file JSON.")
        return

    sequence_images = sorted(sequence_images, key=lambda i: i['file_name'])
    output_dir_path = os.path.join(args.output_dir, args.sequence_name)
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"🎨 Le immagini con le annotazioni verranno salvate in: '{output_dir_path}'")

    # --- 3. Disegno dei Bounding Box ---
    for img_data in tqdm(sequence_images, desc=f"Disegnando BBox per {args.sequence_name}"):
        directory_part, base_filename = os.path.split(img_data['file_name'])
        correct_filename = f"rgb_{base_filename}"
        image_path = os.path.join(args.image_dir, directory_part, correct_filename)

        if not os.path.exists(image_path):
            continue
            
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape
        
        img_id = img_data['id']
        annotations = image_id_to_annotations.get(img_id, [])
        
        for ann in annotations:
            bbox = ann['bbox']
            cat_id = ann['category_id']
            
            x = int(bbox[0] * img_width)
            y = int(bbox[1] * img_height)
            w = int(bbox[2] * img_width)
            h = int(bbox[3] * img_height)
            
            color = category_colors[cat_id]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            cat_name = category_info[cat_id]['name']
            label = f"{cat_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            label_y = max(y, label_h + 10)
            cv2.rectangle(image, (x, label_y - label_h - 10), (x + label_w, label_y), color, -1)
            cv2.putText(image, label, (x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        output_image_path = os.path.join(output_dir_path, correct_filename)
        cv2.imwrite(output_image_path, image)

    print("\n🎉 Processo completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disegna i bounding box delle annotazioni COCO su una sequenza di immagini.")
    parser.add_argument("--annotations_file", type=str, required=True, help="Percorso del file annotations.json.")
    parser.add_argument("--image_dir", type=str, required=True, help="Cartella radice delle immagini (es. 'output/rgb').")
    parser.add_argument("--sequence_name", type=str, required=True, help="Nome della sequenza da processare (es. 'seq0').")
    parser.add_argument("--output_dir", type=str, default="output/annotated_output", help="Cartella dove salvare le immagini annotate.")
    
    args = parser.parse_args()
    main(args)