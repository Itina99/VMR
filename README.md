# Kubric Data Generation Pipeline 🚀

This project uses Kubric (via Docker) to generate rich, synthetic datasets. The pipeline is designed to be robust and scalable, generating data in isolated batches.

The full pipeline generates:
* **📀 Rich Render Passes:** In addition to **RGB**, the pipeline outputs **RGBA**, **Segmentation**, **Depth**, **Forward/Backward Flow**, **Normals**, and **Object Coordinates**.
* **🏷️ Annotations:** A ready-to-use **COCO `annotations.json`** file is saved in the root of each batch folder (e.g., `output_batch_1/`), while the detailed **Kubric metadata** for each sequence is stored in the `annotations/` sub-folder.
* **⚡ Event Data (via `vid2e`):** The pipeline automatically processes the rendered frames. It first *upsamples* the RGB sequences (saving to `upsampled_rgb/`) and then generates event data from them (saving to `events/`).

---

## 🛠️ Installation & Setup

This pipeline depends on **Kubric** for the core simulation and **vid2e** for event generation. For detailed installation and setup instructions, please consult their official repositories:

* **Kubric:** [https://github.com/google-research/kubric](https://github.com/google-research/kubric)
* **vid2e:** [https://github.com/uzh-rpg/rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e)

This project assumes you have a working Docker environment (for Kubric) and a Conda environment set up (for `vid2e`) as used by the `pipeline.sh` script.

---

## ⚙️ How to Run the Dataset Generation

Running the full generation process is simple. The entire pipeline is managed by a main "launcher" script.

1.  **Configure your settings** in the `config.json` file (see details below).
2.  **Make the scripts executable** (you only need to do this once):
    ```bash
    chmod +x run_batches.sh
    chmod +x pipeline.sh
    ```
3.  **Start the generation!**
    ```bash
    ./run_batches.sh
    ```

That's it! The `run_batches.sh` script will:
* Read the `TOTAL_BATCHES` variable inside it (e.g., 5).
* Create a unique output folder for each batch (e.g., `output_batch_1/`, `output_batch_2/`, etc.).
* Call `pipeline.sh` for each batch, which runs the main Docker simulation.
* Each `output_batch_.../` folder will be a complete, self-contained mini-dataset.

---

## 🔧 Configuration (`config.json`)

This file is the main control panel for your dataset. Here is what each parameter does:

### General Settings
* `"resolution": "256x256"`
    * 🖼️ The output resolution for the rendered frames (width x height).
* `"frame_end": 120`
    * ⏱️ The total number of frames to simulate and render per sequence.
* `"frame_rate": 60`
    * 🎞️ The "camera's" frame rate (frames per second).
* `"step_rate": 240`
    * ⚙️ How many physics steps to run for every single rendered frame. A higher value (e.g., 240) results in a more stable and accurate simulation than the default (e.g., 60).

### Generation Mode
* `"standard_generation": true`
    * If set to `true` runs the "standard" mode, which iterates through every single combination of `light_levels`, `camera_positions`, and `camera_modes`.
* `"refresh_item_number": 5`
    * 🔄 How many times to generate a new **set of objects**. For *each* set, the pipeline will iterate through all combinations of scene parameters (`light_levels`, `camera_positions`, `camera_mode`) if `standard_generation` is `true`.
* `"random_generation": true`
    * If set to `true`, the pipeline will run a random generation phase, creating `number_of_random_sequences`. This runs *in addition* to the standard generation (if enabled).
* `"number_of_random_sequences": 50`
    * 🔢 The number of sequences to generate if `random_generation` is `true`.

### Scene & Lighting 💡
* `"light_levels": [1.0, 0.75, 0.5, 0.25]`
    * ✨ A list of brightness levels to simulate (from `0.0` dark to `1.0` bright). The pipeline will loop through these.
* `"camera_positions": ["tilt_30", "tilt_60", "retro_120", "top"]`
    * 📷 A list of camera position names to use. Available options: `tilt_30`, `tilt_60`, `retro_120`, `top`.
* `"camera_mode": ["fixed", "linear_movement", "panning"]`
    * 🎥 A list of camera behaviors to use. Available options: `fixed`, `linear_movement`, `panning`.
* `"max_camera_movement": 2.0`
    * (Used for non-fixed camera modes) Sets the maximum speed or range of camera motion.

### Physics & Object Spawning 📦
* `"min_static_objects": 1` / `"max_static_objects": 3`
    * Defines the random range for the number of **static** (unmoving) objects to place in the scene.
* `"min_dynamic_objects": 1` / `"max_dynamic_objects": 3`
    * Defines the random range for the number of **dynamic** (moving) objects to place in the scene.
* `"spawning_region_static": [[-5, -5, 0], [5, 5, 0]]`
    * Defines the bounding box `[min_corner, max_corner]` where **static** objects will be spawned (a flat area at Z=0).
* `"spawning_region_dynamic": [[-5, -5, 1], [5, 5, 6]]`
    * Defines the bounding box `[min_corner, max_corner]` where **dynamic** objects will be spawned (an area floating above the ground).
* `"velocity_range": [[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0]]`
    * Defines the min and max velocity vectors to apply to dynamic objects, causing them to move across the scene.

---

## 🗃️ Output Structure

After running `./run_batches.sh`, your directory will look like this. Each `output_batch_...` folder is a self-contained dataset.
```
.
├── output_batch_1/
│   ├── annotations.json         <-- COCO file for batch 1
│   ├── annotations/
│   │   └── seq_0.json           <-- Kubric metadata
│   │   └── seq_1.json
│   │   └── ...
│   ├── rgb/
│   │   └── seq_0/
│   │   └── seq_1/
│   │   └── ...
│   ├── rgba/
│   ├── segmentation/
│   ├── depth/
│   ├── forward_flow/
│   ├── backward_flow/
│   ├── normal/
│   ├── object_coordinates/
│   │
│   ├── upsampled_rgb/           <-- Generated by vid2e
│   │   └── seq_0/
│   │   └── ...
│   ├── events/                  <-- Generated by vid2e
│   │   └── seq_0/
│   │   └── ...
│   └── ...
│
├── output_batch_2/
│   ├── annotations.json         <-- COCO file for batch 2
│   ├── annotations/
│   │   └── ...
│   ├── rgb/
│   │   └── ...
│   └── ...
│
├── output_batch_3/
│   └── ...
│
├── config.json
├── pipeline.sh
└── run_batches.sh
└──...
```
