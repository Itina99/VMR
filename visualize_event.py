import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os

def visualize_events(npz_path, resolution=(256, 256), dt_ns=1e6):
    """
    Visualizza gli eventi da un file .npz salvato con ESIM Torch.

    Args:
        npz_path (str): path al file .npz contenente gli eventi.
        resolution (tuple): dimensione dell'immagine di output (larghezza, altezza).
        dt_ns (float): intervallo temporale in nanosecondi tra i frame visualizzati.
    """
    data = np.load(npz_path)
    x = data["x"]
    y = data["y"]
    t = data["t"]
    p = data["p"].astype(bool)  # polarità: True (positivo), False (negativo)

    print(f"Event duration: {(t[-1] - t[0]) / 1e9:.3f}s")

    # Preparazione per visualizzazione
    start_time = t[0]
    end_time = t[-1]
    current_time = start_time
    idx = 0
    num_events = len(x)

    while current_time < end_time:
        # Trova tutti gli eventi nel range [current_time, current_time + dt_ns]
        mask = (t >= current_time) & (t < current_time + dt_ns)

        x_bin = x[mask]
        y_bin = y[mask]
        p_bin = p[mask]

        # Crea canvas nero RGB
        canvas = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Disegna eventi: verde = positivo, rosso = negativo
        canvas[y_bin[p_bin], x_bin[p_bin], 1] = 255  # canale G
        canvas[y_bin[~p_bin], x_bin[~p_bin], 0] = 255  # canale R

        # Mostra la finestra
        cv2.imshow("Event Visualization", canvas)
        key = cv2.waitKey(1)
        if key == 27:  # ESC per uscire
            break

        current_time += dt_ns

    cv2.destroyAllWindows()


if __name__ == "__main__":
    npz_file = "output/events/seq0.npz"
    if not os.path.exists(npz_file):
        print(f"File {npz_file} non trovato.")
    else:
        visualize_events(npz_file, resolution=(256, 256), dt_ns=1e6)