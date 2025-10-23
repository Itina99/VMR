import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os

def visualize_events(npz_path, resolution=(256, 256), dt_ns=1e6):
    """
    Visualizes events from a .npz file saved with ESIM Torch.

    Args:
        npz_path (str): path to the .npz file containing the events.
        resolution (tuple): output image dimensions (width, height).
        dt_ns (float): time interval in nanoseconds between displayed frames.
    """
    data = np.load(npz_path)
    x = data["x"]
    y = data["y"]
    t = data["t"]
    p = data["p"].astype(bool) 

    print(f"Event duration: {(t[-1] - t[0]) / 1e9:.3f}s")

    start_time = t[0]
    end_time = t[-1]
    current_time = start_time
    idx = 0
    num_events = len(x)

    while current_time < end_time:
        mask = (t >= current_time) & (t < current_time + dt_ns)

        x_bin = x[mask]
        y_bin = y[mask]
        p_bin = p[mask]

        canvas = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        canvas[y_bin[p_bin], x_bin[p_bin], 1] = 255  
        canvas[y_bin[~p_bin], x_bin[~p_bin], 0] = 255  

        cv2.imshow("Event Visualization", canvas)
        key = cv2.waitKey(33)
        if key == 27:  
            break

        current_time += dt_ns
        idx += 1

    print(idx)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    npz_file = "output/events/seq5.npz"
    if not os.path.exists(npz_file):
        print(f"File {npz_file} not found.")
    else:
        visualize_events(npz_file, resolution=(256, 256), dt_ns=33e6)
