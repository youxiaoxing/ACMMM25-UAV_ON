
# UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents

----------

## Notes

We are currently collecting feedback to help improve this work. A major v2.0 release is planned in the next 2–3 months, and we would greatly appreciate your input in shaping its development and improvement.

Please share your suggestions and comments through the following form: [Feedback Form](https://forms.gle/wQ8Ypw2x2kUb18XVA)

## Content

- [Introduction](#introduction)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [TODO](#todo)
- [Acknowledgment](#acknowledgment)

## Introduction

<p align="center">
  <img src="image/task_demo.png" width="90%">
</p>

Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on step-by-step linguistic instructions, limiting its scalability and autonomy. To bridge this gap, we propose **UAV-ON**, a benchmark designed to facilitate research on large-scale **Object Goal Navigation (ObjectNav)** by aerial agents operating in open-world environments. UAV-ON comprises **14 high-fidelity Unreal Engine environments** with diverse semantic regions and complex spatial layouts, covering **urban, natural, and mixed-use** settings. It defines **1270 annotated target objects**, each paired with a structured semantic prompt that encodes category, estimated physical footprint, and detailed visual descriptors, allowing grounded reasoning. These prompts serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. We also propose **Aerial ObjectNav Agent (AOA)**, a modular baseline policy that integrates prompt semantics with egocentric observations to perform long-horizon, goal-directed exploration. Empirical results demonstrate that standard baselines perform poorly in this setting, underscoring the compounded difficulty of aerial navigation and semantic goal grounding. **UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments**.

**For detailed supplementary information about this project, please refer to the appendix: [Click to view appendix](https://drive.google.com/file/d/11nc_SmsQ5fDNz_wON3vqqATLWKm251Ik/view?usp=drive_link)**

## Demo

Watch a full successful flight of our Aerial ObjectNav Agent in action:

<p align="center">
  <a href="https://youtu.be/Zx-Bhzc5Cv4">
    <img src="https://img.youtube.com/vi/Zx-Bhzc5Cv4/0.jpg" alt="UAV-ON Demo" width="70%"/>
  </a>
</p>

> Click the image above or [this link](https://youtu.be/Zx-Bhzc5Cv4) to view the demo video.



## Getting Started

- **Step1: Install all dependencies**

    ```bash
    conda create -n uavon python==3.8
    conda activate uavon
    pip install -r requirements.txt
    ```

- **Step2: Prepare the simulation environment**

    You can get UAV-ON train environments from [train envs](https://huggingface.co/datasets/Kyaren/UAV-ON-envs-train) (44.1G) and [test envs](https://huggingface.co/datasets/Kyaren/UAV-ON-envs-test) (26.8G)
    The environment directory should be structured as follows:

    ``` text
    TRAIN_ENVS/
    ├── Barnyard/
    ├── BrushifyRoad/
    ├── CabinLake/
    └── ... (other training environments)

    TEST_ENVS/
    ├── Barnyard/
    ├── BrushifyRoad/
    ├── CabinLake/
    └── ... (other testing environments)
    ```

- **Step3: Get dataset json files**

    You can download dataset from [here](https://huggingface.co/datasets/Kyaren/UAV-ON-dataset)，you can use [script](https://github.com/Kyaren/UAV_ON/tree/main/scripts) to merge split data files into a single JSON file

- **Project directory structure**
  
  Your workspace directory should be structured as follows:
  
  ```text
  workspace/
  ├── UAV_ON/        
  ├── DATASET/        
  ├── TRAIN_ENVS/
  └── TEST_ENVS/
  ```

## Usage
  
  1.First, you should launch the AirSim environment server.

  ```bash
  python airsim_plugin/AirVLNSimulatorServerTool.py --port=30000 --root_path= "your workspace path"
  ```

  2.Then, you can execute the bash script to run the simulator

  ```bash
  #AOA-F/V
  bash scripts/eval_fixed.sh
  bash scripts/eval_unfixed.sh
  #CLIP-H
  bash scripts/eval_cliph.sh

  bash scripts/metric.sh
  ```

  If you encounter the "**Ping returned false**" error and **no output in server console**, this is caused by the package version. You can run the following command:

  ```bash 
  pip uninstall msgpack-python msgpack-rpc-python
  pip install msgpack-rpc-python
  ```

## A* Path Image Collection (from GT JSON)

If you already have ground-truth episode JSON (same format as UAV-ON dataset), you can generate an A* path from start pose to target pose and capture RGB/Depth images along the whole path.

```bash
python scripts/astar_collect_images.py \
  --gt_json /path/to/gt.json \
  --output_dir ./logs/astar_path_images \
  --simulator_port 31000 \
  --gpu_id 0 \
  --horizontal_step 5.0 \
  --vertical_step 2.0 \
  --yaw_step_deg 15.0 \
  --state_xy_resolution 1.0 \
  --state_z_resolution 1.0 \
  --search_margin_xy_m 60.0 \
  --search_margin_z_m 20.0 \
  --edge_check_step 1.0 \
  --max_expansions 50000 \
  --goal_tolerance_xy 5.0 \
  --goal_tolerance_z 2.0 \
  --cameras 0,1,2,3
```

This script now uses **action-space-consistent A***:
- State includes `(x, y, z, yaw)` and actions are restricted to:
  `forward`, `left`, `right`, `ascend`, `descend`, `rotl`, `rotr`.
- Candidate nodes/edges are checked with AirSim collision feedback to avoid obstacles.
- `path_meta.json` includes planned action sequence so it can be used for imitation/trajectory supervision.
- If one episode fails (e.g., start in collision), the script will skip it and continue; skipped items are recorded in `<map_name>/skipped_episodes.jsonl`.
- Requires the simulator scene to be running and reachable.

Output structure:

```text
logs/astar_path_images/
└── <map_name>/
    └── episode_<episode_id>/
        ├── path_meta.json
        ├── step_0000/
        │   ├── cam_0_rgb.png
        │   ├── cam_0_depth.png
        │   └── ...
        └── step_0001/
            └── ...
```




### TODO

- Example of Reinforcement Learning using PPO
- Example of training a model using Imitation Learning

### **Acknowledgment**

- The simulation interaction module of this project is built upon the works of [AirVLN](https://github.com/AirVLN/AirVLN/) and [TravelUAV](https://github.com/prince687028/TravelUAV/). We sincerely thank them for their outstanding contributions.
