<!-- # New Version Command
MUJOCO_EGL_DEVICE_ID=6 CUDA_VISIBLE_DEVICES=6 MUJOCO_GL='egl' python train.py --note "XXXXXXX to confirm your note for this experiment" --env "HandManipulatePenRotateFixed-v1"

MUJOCO_EGL_DEVICE_ID=6 CUDA_VISIBLE_DEVICES=6 MUJOCO_GL='egl' python eval.py -->

# ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning

<!-- [Paper](https://arxiv.org/abs/2311.00924) [Website](https://sferrazza.cc/m3l_site/) -->

[Yufeng Tian*](https://skyrainwind.github.io), [Shuiqi Cheng*](https://github.com/shuiqicheng), [Tianming Wei](https://www.stillwtm.site/), [Tianxing Zhou](https://github.com/ZTX2021), [Yuanhang Zhang](https://yuanhangz.com/), [Zixian Liu](https://storeblank.github.io/), [Qianwei Han](https://der-mark.github.io/), [Zhecheng Yuan](gemcollector.github.io), [Huazhe Xu](hxu.rocks).

\* Equal contribution.

We propose **Vi**sual **Ta**ctile **S**oft Fusion Contrastive Learning (ViTaS), a novel visuo-tactile representation learning framework for visuomotor learning. 

![image](teaser.png)

## Installation
Please install [`tactile_envs`](https://github.com/carlosferrazza/tactile_envs.git) first. Then, install the remaining dependencies:
```

git submodule update --init
git clone https://github.com/carlosferrazza/tactile_envs
pip install -r requirements.txt
```

## Training M3L
```
MUJOCO_GL='egl' python train.py --env tactile_envs/Insertion-v0
```

## Training M3L (vision policy)
```
MUJOCO_GL='egl' python train.py --env tactile_envs/Insertion-v0 --vision_only_control True
```

# Acknowledgement
Our code is generally built upon [M3L](https://github.com/carlosferrazza/M3L). The real-world experiments are conducted with the help of [Galaxea Dynamics](https://galaxea-dynamics.com/). We thank all these authors for their great contributions to the community.