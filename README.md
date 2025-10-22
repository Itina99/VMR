Kubric Data Generation Pipeline 🚀

This project uses Kubric (via Docker) to generate synthetic datasets, complete with RGB frames, event data, and COCO annotations. The pipeline is designed to be robust and scalable, generating data in isolated batches.

⚙️ How to Run the Dataset Generation

Running the full generation process is simple. The entire pipeline is managed by a main "launcher" script.

    Configure your settings in the config.json file (see details below).

    Make the scripts executable (you only need to do this once):
    Bash

chmod +x run_batches.sh
chmod +x pipeline.sh

Start the generation!
Bash

    ./run_batches.sh

That's it! The run_batches.sh script will:

    Read the TOTAL_BATCHES variable inside it (e.g., 5).

    Create a unique output folder for each batch (e.g., output_batch_1/, output_batch_2/, etc.).

    Call pipeline.sh for each batch, which runs the main Docker simulation.

    Each output_batch_.../ folder will be a complete, self-contained mini-dataset with its own annotations.json file.

🔧 Configuration (config.json)

This file is the main control panel for your dataset. Here is what each parameter does:

General Settings

    "refresh_item_number": 1

        (Not currently used in pipeline) A legacy setting.

    "resolution": "256x256"

        🖼️ The output resolution for the rendered frames (width x height).

    "frame_end": 120

        ⏱️ The total number of frames to simulate and render per sequence.

    "frame_rate": 60

        🎞️ The "camera's" frame rate (frames per second).

    "step_rate": 240

        ⚙️ How many physics steps to run for every single rendered frame. A higher value (e.g., 240) results in a more stable and accurate simulation than the default (e.g., 60).

Generation Mode

    "standard_generation": true

        Runs the "standard" mode, which iterates through every single combination of light_levels, camera_positions, and light_colors.

    "random_generation": false

        If set to true, this ignores the combinations above and instead generates number_of_random_sequences, picking a random setting for each one.

    "number_of_random_sequences": 1

        🔢 The number of sequences to generate if random_generation is true.

Scene & Lighting 💡

    "light_levels": [1.0]

        ✨ A list of brightness levels to simulate (from 0.0 dark to 1.0 bright). The pipeline will loop through these.

    "camera_positions": ["tilt_30"]

        📷 A list of camera position names to use.

    "light_colors": ["white"]

        🎨 A list of light color names to use.

    "camera_mode": ["fixed"]

        🎥 The behavior of the camera. "fixed" means it stays still.

    "max_camera_movement": 2.0

        (Used for non-fixed camera modes) Sets the maximum speed or range of camera motion.

Physics & Object Spawning 📦

    "min_static_objects": 1 / "max_static_objects": 3

        Defines the random range for the number of static (unmoving) objects to place in the scene.

    "min_dynamic_objects": 1 / "max_dynamic_objects": 3

        Defines the random range for the number of dynamic (moving) objects to place in the scene.

    "spawning_region_static": [[-5, -5, 0], [5, 5, 0]]

        Defines the bounding box [min_corner, max_corner] where static objects will be spawned (a flat area at Z=0).

    "spawning_region_dynamic": [[-5, -5, 1], [5, 5, 6]]

        Defines the bounding box [min_corner, max_corner] where dynamic objects will be spawned (an area floating above the ground).

    "velocity_range": [[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0]]

        Defines the min and max velocity vectors to apply to dynamic objects, causing them to move across the scene.

🗃️ Output Structure

After running ./run_batches.sh, your directory will look like this:

.
├── output_batch_1/
│   ├── annotations/
│   │   └── annotations.json  <-- COCO file for batch 1
│   ├── rgb/
│   │   └── seq_0/
│   │   └── seq_1/
│   │   └── ...
│   ├── events/
│   ├── upsampled_rgb/
│   └── ...
├── output_batch_2/
│   ├── annotations/
│   │   └── annotations.json  <-- COCO file for batch 2
│   ├── rgb/
│   │   └── seq_1000/
│   │   └── ...
│   └── ...
├── output_batch_3/
│   └── ...
├── config.json
├── pipeline.sh
└── run_batches.sh
