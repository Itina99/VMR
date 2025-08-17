import torch
import numpy as np
import cv2
import os
import glob
import logging
from pathlib import Path
import esim_torch


class EventGenerator:
    def __init__(self, base_dir="output/upsampled_rgb", output_base_dir="output/events", device="cuda:0"):
        self.base_dir = Path(base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.device = device
        self.esim = esim_torch.ESIM(
            contrast_threshold_neg=0.2,
            contrast_threshold_pos=0.2,
            refractory_period_ns=0
        )

    def _load_images(self, image_dir):
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not image_files:
            raise FileNotFoundError(f"Nessuna immagine trovata in {image_dir}")
        logging.info(f"Caricate {len(image_files)} immagini da {image_dir}")
        images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
        return images

    def _load_timestamps(self, timestamp_file):
        if not os.path.exists(timestamp_file):
            raise FileNotFoundError(f"File timestamps non trovato: {timestamp_file}")
        timestamps_s = np.genfromtxt(timestamp_file)
        timestamps_ns = (timestamps_s * 1e9).astype("int64")
        return timestamps_ns

    def _process_sequence(self, seq_dir):
        """Process a single sequence directory"""
        seq_name = seq_dir.name
        logging.info(f"🎬 Elaborando sequenza: {seq_name}")
        
        # Paths for this sequence
        image_dir = seq_dir / "imgs"
        timestamp_file = seq_dir / "timestamps.txt"
        output_file = self.output_base_dir / f"{seq_name}.npz"
        
        # Check if required files exist
        if not image_dir.exists():
            logging.warning(f"⚠️ Directory immagini non trovata per {seq_name}: {image_dir}")
            return False
            
        if not timestamp_file.exists():
            logging.warning(f"⚠️ File timestamps non trovato per {seq_name}: {timestamp_file}")
            return False

        try:
            # Load images and timestamps
            images = self._load_images(image_dir)
            timestamps_ns = self._load_timestamps(timestamp_file)
            
            # Validate dimensions
            if len(images) != len(timestamps_ns):
                logging.error(f"❌ Mismatch: {len(images)} immagini vs {len(timestamps_ns)} timestamps per {seq_name}")
                return False

            # Convert to log space
            log_images = np.log(images.astype("float32") / 255 + 1e-4)

            # Convert to tensors
            log_images = torch.from_numpy(log_images).to(self.device)
            timestamps_ns = torch.from_numpy(timestamps_ns).to(self.device)

            # Generate events
            events = self.esim.forward(log_images, timestamps_ns)

            # Save events
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file,
                                x=events['x'].cpu().numpy(),
                                y=events['y'].cpu().numpy(),
                                t=events['t'].cpu().numpy(),
                                p=events['p'].cpu().numpy())

            logging.info(f"✅ Eventi salvati per {seq_name} in {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Errore durante l'elaborazione di {seq_name}: {e}")
            return False

    def generate_all(self):
        """Generate events for all sequences in the base directory"""
        logging.info(f"🚀 Inizio generazione eventi per tutte le sequenze in {self.base_dir}")
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Directory base non trovata: {self.base_dir}")
        
        # Find all sequence directories (should be named like seq0, seq1, etc.)
        seq_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('seq')]
        
        if not seq_dirs:
            raise FileNotFoundError(f"Nessuna directory di sequenza trovata in {self.base_dir}")
        
        logging.info(f"📁 Trovate {len(seq_dirs)} sequenze da elaborare")
        
        # Process each sequence
        successful = 0
        failed = 0
        
        for seq_dir in sorted(seq_dirs):
            if self._process_sequence(seq_dir):
                successful += 1
            else:
                failed += 1
        
        logging.info(f"🎯 Completato: {successful} successi, {failed} fallimenti")
        return successful, failed

    def generate_single(self, seq_name):
        """Generate events for a specific sequence"""
        seq_dir = self.base_dir / seq_name
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequenza non trovata: {seq_dir}")
        
        return self._process_sequence(seq_dir)

    # Keep the old interface for backward compatibility
    def generate(self, image_dir=None, timestamp_file=None, output_file=None):
        """Legacy method for single sequence processing"""
        if image_dir and timestamp_file and output_file:
            # Old style usage
            logging.warning("⚠️ Uso del metodo legacy generate(). Considera di usare generate_single() o generate_all()")
            
            images = self._load_images(image_dir)
            timestamps_ns = self._load_timestamps(timestamp_file)

            log_images = np.log(images.astype("float32") / 255 + 1e-4)
            log_images = torch.from_numpy(log_images).to(self.device)
            timestamps_ns = torch.from_numpy(timestamps_ns).to(self.device)

            events = self.esim.forward(log_images, timestamps_ns)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file,
                                x=events['x'].cpu().numpy(),
                                y=events['y'].cpu().numpy(),
                                t=events['t'].cpu().numpy(),
                                p=events['p'].cpu().numpy())

            logging.info(f"✅ Eventi salvati in {output_file}")
        else:
            # New style - process all sequences
            return self.generate_all()
