# Training Pi0 with LeRobot in MuJoCo

## Table of Contents
- [📚 Theory](#-theory)
  - [➡️ Pi0](#️-pi0)
    - [1. Pre-trained VLM](#1-pre-trained-vlm)
    - [2. Action Expert](#2-action-expert)

- [Setup](#setup)
- [1. Collect Demonstration Data](#1-collect-demonstration-data)
- [2. Playback Your Data](#2-playback-your-data)
- [3. Train Action Chunking Transformer (Pi0)](#3-train-action-chunking-transformer-pi0)
- [Reference](#reference)



## 📚 Theory

### ➡️ Pi0 

**π₀** is a **VLA** model developed by Physical Intelligence for general-purpose robot control.

Unlike ACT, which directly predicts action chunks, π₀ combines a pretrained **VLM** with a specialized **action expert** to generate continuous robot actions.  π₀ is a **single transformer with two sets of weights**.

> [!IMPORTANT]
> **VLM** focuses on **understanding the scene**, while the **Action Expert** determines **how the robot should move**. Their parameters are independent, but they exchange information through the **attention mechanism at each Transformer layer**.

<p align="center">
  <img src="assets/overview_pi0.jpg" width="900">
</p>

#### 1. Pre-trained VLM
π₀ builds its VLM backbone on **PaliGemma**, which consists of:

* **SigLIP**: A **ViT (Vision Transformer)** used as the vision encoder. It converts input images into image tokens. A **linear projection layer** then maps these image tokens to the same embedding dimension as the Gemma text tokens, allowing them to be processed together.

* **Gemma**: A **decoder-only Transformer** used as the language-model backbone. It jointly processes the projected image tokens and text tokens to produce contextual visual-language representations.

> [!NOTE]
> **SigLIP**
> Image → Image Tokens
> 
> **Gemma**
> Image Tokens + Text Tokens → Contextual Representations

<p align="center">
  <img src="assets/PaliGemma.jpg" width="800">
</p>

#### 2. Action Expert
**🌊conditional flow matching**: random noise -> valid action trajectories.


Instead of directly predicting the final action, the model learns a **vector field** that tells a noisy action in which direction it should move toward the real action.

**Training** 

A noisy action is constructed as:

$$
A_t^\tau = \tau A_t + (1-\tau)\epsilon
$$

where:

- $A_t$: ground-truth action chunk
- $\epsilon \sim \mathcal{N}(0, I)$: random noise
- $\tau \in [0,1]$: flow-matching timestep

Therefore:

$$
\tau=0 \Rightarrow A_t^\tau=\epsilon
$$

while:

$$
\tau=1 \Rightarrow A_t^\tau=A_t
$$

is the real action.

The model learns:

$$
v_\theta(A_t^\tau,o_t)
$$

which predicts the direction in which the noisy action should move. In π₀, the training goal is:

$$
v_\theta(A_t^\tau,o_t)\approx A_t-\epsilon
$$

**Inference**

At inference time, π₀ starts from random noise:

$$
A_t^0\sim\mathcal N(0,I)
$$

and repeatedly updates the action using:

$$
A_t^{\tau+\delta} =
A_t^\tau+\delta v_\theta(A_t^\tau,o_t)
$$

π₀ uses **10 integration steps** with \($\delta=0.1\$).

## Setup
```bash
conda create -n py310 python=3.10
conda activate py310
```
```bash
git clone git@github.com:Gege4526/LeRobot-MuJoCo-Pi0.git
cd ~/LeRobot-MuJoCo-Pi0
pip install -r requirements.txt
conda install jupyterlab
pip install ipywidgets ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"
code .
```
```bash
cd asset/objaverse
unzip plate_11.zip
```
## 1. Collect Demonstration Data


### 🎮 Keyboard Controls

* `W/A/S/D`: Move along the **x-y plane**
* `R/F`: Move along the **z-axis**
* `Q/E`: Tilt the end-effector
* `Arrow Keys`: Rotate the end-effector
* `Space`: Toggle the gripper state
* `Z`: Reset the environment and discard the cached data from the current episode

### 🖥️ Observation Views

The rendered observation contains four views:

* **Top-right:** Agent view
* **Bottom-right:** First-person wrist camera view
* **Top-left:** Side view
* **Bottom-left:** Top-down view

### Data Structure
```bash
fps = 20,
features={
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 个关节角 + 1 个夹爪
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # 仅物体初始位置，训练中不使用
    },
},
```

## 2. Playback Your Data


## 3. Train Action Chunking Transformer (Pi0)


## Reference
**Paper**
https://arxiv.org/abs/2410.24164

**Blog**
https://www.pi.website/blog/pi0

