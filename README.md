<div align="center">

# KUDA: Keypoints to Unify Dynamics Learning and Visual Prompting for Open-Vocabulary Robotic Manipulation

[Project Page](https://kuda-dynamics.github.io/) | [Video](https://kuda-dynamics.github.io/videos/kuda_teaser_v2.mp4) | [Arxiv]()

</div>

<div align="center">
  <img src="assets/teaser.png" alt="KUDA" width="100%"/>
</div>


## 🛠️ Installation

To set up the environment, please follow these steps:

1. Create the conda environment:
   ```bash
   conda create -n kuda python=3.8
   conda activate kuda
   pip install -r requirements.txt
   ```

2. Download the checkpoints for [GroundingSAM](https://github.com/IDEA-Research/GroundingDINO):
   ```bash
   cd perception/models
   mkdir checkpoints
   cd checkpoints
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
   ```

3. Download the checkpoints for [SpaTracker](https://github.com/henry123-boy/SpaTracker):
   ```bash
   cd ../../../dynamics/tracker
   mkdir checkpoints
   cd checkpoints
   pip install gdown
   gdown 18YlG_rgrHcJ7lIYQWfRz_K669z6FdmUX
   ```
   You can manually download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ).


## 🕹️ Usage

### Visual Prompting (No Robot Setup)

To quickly test the visual prompting functionality without setting up a robot, please follow these steps:

1. Replace the `api_key` in `demo.py` with your OpenAI API key.
2. Run the demo:
   ```bash
   python demo.py
   ```
   
   Please modify the `img` and `instruction` variables in `demo.py` to experiment with different tasks. You can see examples in [results](assets/results.md).


### Real-World Execution

To execute tasks in the real world, please follow these steps:

1. **Dynamics Models**:  
   Please download the dynamics model checkpoints from [this link](https://drive.google.com/drive/folders/1szjoTHBZH1kYE_V_1-bv8uVcsd9lxzVT?usp=sharing). Please update the corresponding paths in `dynamics/dyn_utils.py` to ensure the checkpoints are properly loaded and accessible.


2. **Calibration**:  
   We use the xArm6 robot and a ChArUco calibration board. Please run the following code for calibration:
   ```bash
   cd xarm-calibrate
   python calibrate.py
   python camera_to_base_transforms.py
   ```

   Please replace the camera serial number in `xarm-calibrate/real_world/real_env.py` and the robot IP in `xarm-calibrate/real_world/xarm6.py`.

   To verify the calibration results, please run:
   ```bash
   python verify_stationary_cameras.py
   ```

3. **Robot Execution**:  
Please ensure the following steps for real world execution:
   - We employ different end-effctors to manipulate various objects. Specifically, we use the cylinder stick for T shape, ropes, board pusher for cubes and granular pieces. Please update the robot setup in `config/real_config.yaml->planner` and `envs/real_env.py`. Ensure the top-down and side cameras have clear views.
   - Please change hyperparameters such as `radius` in `planner/planner.py` and `box_threshold` in `perception/models/grounding_dino_wrapper.py` for various objects.
   - Please replace the `api_key` in `launch.py` with your OpenAI API key.

   Launch the execution:
   ```bash
   python launch.py
   ```
   It is expected to see the execution results in `logs/low_level` and dynamics predictions in `logs/{material}-planning-{time.time()}`.


## 🔬 Acknowledgements

We thank the authors of the following projects for making their code open source:

- [SpaTracker](https://github.com/henry123-boy/SpaTracker)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## 🏷️ License

This repository is released under the [MIT license](LICENSE).

## 🔭 Citation

```   
```