import imageio
import os

# Cartella con i frame
folder = "output/object_coordinates/seq1/imgs"

# Recupera i file ordinati
frames = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

# Carica le immagini
images = [imageio.imread(os.path.join(folder, f)) for f in frames]

# Salva la GIF
imageio.mimsave("output/gifs/object_coordinates1.gif", images, fps=10)  # fps=10 → 10 frame al secondo
