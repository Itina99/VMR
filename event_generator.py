import torch
import numpy as np
import cv2
import os
import glob
import logging
import esim_torch


class EventGenerator:
    def __init__(self, image_dir, timestamp_file, output_file, device="cuda:0"):
        self.image_dir = image_dir
        self.timestamp_file = timestamp_file
        self.output_file = output_file
        self.device = device
        self.esim = esim_torch.ESIM(
            contrast_threshold_neg=0.2,
            contrast_threshold_pos=0.2,
            refractory_period_ns=0
        )

    def _load_images(self):
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        if not image_files:
            raise FileNotFoundError(f"Nessuna immagine trovata in {self.image_dir}")
        logging.info(f"Caricate {len(image_files)} immagini")
        images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
        return images

    def _load_timestamps(self):
        if not os.path.exists(self.timestamp_file):
            raise FileNotFoundError(f"File timestamps non trovato: {self.timestamp_file}")
        timestamps_s = np.genfromtxt(self.timestamp_file)
        timestamps_ns = (timestamps_s * 1e9).astype("int64")
        return timestamps_ns

    def generate(self):
        logging.info("Inizio generazione eventi...")

        images = self._load_images()
        timestamps_ns = self._load_timestamps()

        log_images = np.log(images.astype("float32") / 255 + 1e-4)

        log_images = torch.from_numpy(log_images).to(self.device)
        timestamps_ns = torch.from_numpy(timestamps_ns).to(self.device)

        events = self.esim.forward(log_images, timestamps_ns)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        np.savez_compressed(self.output_file,
                            x=events['x'].cpu().numpy(),
                            y=events['y'].cpu().numpy(),
                            t=events['t'].cpu().numpy(),
                            p=events['p'].cpu().numpy())

        logging.info(f"✅ Eventi salvati in {self.output_file}")
