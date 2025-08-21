import argparse
import glob
import os
import natsort
import numpy as np
from PIL import Image

def render(x, y, p, shape):
    """Rende un frame RGB a partire da eventi (x,y) con polarità."""
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")  # sfondo bianco
    
    if len(x) == 0:  # Nessun evento
        return img
    
    # Eventi positivi → rosso
    mask_pos = p == True
    if np.any(mask_pos):
        img[y[mask_pos], x[mask_pos], :] = 0
        img[y[mask_pos], x[mask_pos], 0] = 255  # canale R
    
    # Eventi negativi → blu
    mask_neg = p == False
    if np.any(mask_neg):
        img[y[mask_neg], x[mask_neg], :] = 0
        img[y[mask_neg], x[mask_neg], 2] = 255  # canale B
    
    return img

def render_accumulated(x, y, p, t, shape, current_time, decay_factor=0.7):
    """Render con decay temporale per migliore visualizzazione."""
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    
    if len(x) == 0:
        return img
    
    # Calcola intensità basata sul tempo (più recente = più intenso)
    time_diff = current_time - t
    intensity = np.exp(-time_diff * decay_factor)
    intensity = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    
    # Eventi positivi → rosso con intensità variabile
    mask_pos = p == True
    if np.any(mask_pos):
        img[y[mask_pos], x[mask_pos], 1:] = 255 - intensity[mask_pos, np.newaxis]  # Riduci G e B
        img[y[mask_pos], x[mask_pos], 0] = 255  # Mantieni R al massimo
    
    # Eventi negativi → blu con intensità variabile
    mask_neg = p == False
    if np.any(mask_neg):
        img[y[mask_neg], x[mask_neg], :2] = 255 - intensity[mask_neg, np.newaxis]  # Riduci R e G
        img[y[mask_neg], x[mask_neg], 2] = 255  # Mantieni B al massimo
    
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate GIF animations from event streams")
    parser.add_argument("--input_dir", default="output/events")
    parser.add_argument("--shape", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--frames", type=int, default=24, help="Numero di frame per GIF")
    parser.add_argument("--fps", type=int, default=10, help="Frame per second nella GIF")
    parser.add_argument("--window_size", type=float, default=0.1, help="Finestra temporale per accumulo eventi")
    parser.add_argument("--use_accumulation", action="store_true", help="Usa rendering con accumulo")
    args = parser.parse_args()

    event_files = natsort.natsorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    print(f"Trovati {len(event_files)} file di eventi in {args.input_dir}")

    for f in event_files:
        events = np.load(f)
        x, y, t, p = events["x"], events["y"], events["t"], events["p"].astype(bool)

        # Normalizza il tempo tra 0 e 1
        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-9)

        images = []
        for i in range(args.frames):
            current_time = (i + 1) / args.frames
            
            if args.use_accumulation:
                # Usa finestra temporale mobile con decay
                window_start = max(0, current_time - args.window_size)
                mask = (t_normalized >= window_start) & (t_normalized <= current_time)
                if np.any(mask):
                    img = render_accumulated(x[mask], y[mask], p[mask], t_normalized[mask], 
                                           args.shape, current_time)
                else:
                    img = render([], [], [], args.shape)
            else:
                # Metodo originale migliorato
                mask = (t_normalized >= i/args.frames) & (t_normalized < (i+1)/args.frames)
                if np.any(mask):
                    img = render(x[mask], y[mask], p[mask], shape=args.shape)
                else:
                    img = render([], [], [], args.shape)
            
            images.append(Image.fromarray(img))  # Rimussa cv2.cvtColor

        if images:
            filename = os.path.splitext(os.path.basename(f))[0]
            output_path = f"{filename}.gif"
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000/args.fps),
                loop=0
            )
            print(f"GIF salvata come {output_path} ({len(images)} frame)")
        else:
            print(f"Nessun frame valido per {f}")
