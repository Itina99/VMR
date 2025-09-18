import cv2
import numpy as np
import os
import glob

def simulate_light(img, factor):
    img = img.astype(np.float32) / 255.0
    img_linear = np.power(img, 2.2)
    img_linear *= factor
    img_out = np.power(np.clip(img_linear, 0, 1), 1/2.2)
    return (img_out * 255).astype(np.uint8)

def process_all_sequences(rgb_root, factors):
    for seq_dir in sorted(os.listdir(rgb_root)):
        seq_path = os.path.join(rgb_root, seq_dir)
        imgs_dir = os.path.join(seq_path, "imgs")
        if not os.path.isdir(imgs_dir):
            continue  # ignora cartelle che non contengono imgs

        # elabora per ogni fattore
        for factor in factors:
            intensity_str = str(int(factor * 100))
            new_seq_dir = f"{seq_dir}-{intensity_str}"
            new_imgs_dir = os.path.join(rgb_root, new_seq_dir, "imgs")
            os.makedirs(new_imgs_dir, exist_ok=True)

            for path in glob.glob(os.path.join(imgs_dir, "*.png")):
                img = cv2.imread(path)
                dark = simulate_light(img, factor)
                filename = os.path.basename(path)
                cv2.imwrite(os.path.join(new_imgs_dir, filename), dark)

            print(f"✔ Sequenza {seq_dir} → {new_seq_dir}")

if __name__ == "__main__":
    RGB_ROOT = "output/rgb"
    FACTORS = [0.5, 0.25, 0.1]  # lista di intensità desiderate
    process_all_sequences(RGB_ROOT, FACTORS)
