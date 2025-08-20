import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 
import natsort


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--input_dir", default="output/events")
    parser.add_argument("--shape", nargs=2, default=[256, 256])
    args = parser.parse_args()

    event_files = natsort.natsorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    print(f"Found {len(event_files)} event files in {args.input_dir}")
    
    fig, ax = plt.subplots()
    events = np.load(event_files[0])
    img = render(shape=args.shape, **events)

    video_writer = cv2.VideoWriter('event_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (args.shape[1], args.shape[0]))
    video_writer.write(img)

    for f in event_files[1:]:
        events = np.load(f)
        img = render(shape=args.shape, **events)
        video_writer.write(img)

    video_writer.release()
    plt.close(fig)