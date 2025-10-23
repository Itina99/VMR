import cv2
import json
import os
import argparse
from tqdm import tqdm
import random

def main(args):
    print(f"🔎 Loading annotations file: '{args.annotations_file}'")
    try:
        with open(args.annotations_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found. Please ensure the path is correct.")
        return

    print("🔄 Indexing annotations...")
    
    images_info = {img['id']: img for img in coco_data['images']}
    category_info = {cat['id']: cat for cat in coco_data['categories']}
    
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    category_colors = {cat_id: [random.randint(100, 255), random.randint(100, 255), random.randint(50, 200)] for cat_id in category_info}
    
    sequence_images = [img for img in coco_data['images'] if img['file_name'].startswith(f"{args.sequence_name}/")]
    
    if not sequence_images:
        print(f"⚠️ Warning: No images found for sequence '{args.sequence_name}' in JSON file.")
        return

    sequence_images = sorted(sequence_images, key=lambda i: i['file_name'])
    output_dir_path = os.path.join(args.output_dir, args.sequence_name)
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"🎨 Annotated images will be saved to: '{output_dir_path}'")

    for img_data in tqdm(sequence_images, desc=f"Drawing BBoxes for {args.sequence_name}"):
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

            if w <= 0 or h <= 0:
                continue
            # ------------------------------------
            
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

    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes of COCO annotations on a sequence of images.")
    parser.add_argument("--annotations_file", type=str, default="output_batch_1/annotations.json", help="Path to the annotations.json file.")
    parser.add_argument("--image_dir", type=str, default="output_batch_2/rgb", help="Root folder of the images (e.g. 'output/rgb').")
    parser.add_argument("--sequence_name", type=str, default="1761163564053_deb400", help="Name of the sequence to process (e.g. 'seq0').")
    parser.add_argument("--output_dir", type=str, default="output_batch_1/annotated_output", help="Folder to save the annotated images.")

    args = parser.parse_args()
    main(args)