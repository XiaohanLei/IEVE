# [CVPR 2024] Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation

This is the pytorch implementation of CVPR 2024 paper:  Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation (IEVE).

#### Now the action space aligns with the paper reported (velocity control). If you have any other issues, feel free to contact us!

[Project Page](https://xiaohanlei.github.io/projects/IEVE/)<br />

![example](./output.gif)

### Overview:

Inspired by the human behavior of “getting closer to confirm” when recognizing distant objects, we formulate the task of determining whether an object matches the one in the goal image as a sequential decision problem. In addition, we design a novel matching function that relies not
only on the current observation and goal image but also on the Euclidean distance between the agent and the potential target. We categorize the targets into confirmed target, potential target, and no-target (exploration), and allow the agent to actively choose among these three targets.

## Installing Dependencies
- We use v0.2.3 of [habitat-sim](https://github.com/facebookresearch/habitat-sim), please follow the instructions to complete installation
- Install habitat-lab: (make sure you install our modified version of habitat-lab)
```
pip install -e 3rdparty/habitat-lab
```
- Install [pytorch](https://pytorch.org/) according to your system configuration
- cd to the root directory, install requirements
```
pip install -r requirements.txt
```
- Install [LightGlue](https://github.com/cvg/LightGlue), please follow the official guidance
- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration

### Downloading scene dataset and episode dataset
- Follow the instructions in [habitat-lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)
- Move the dataset or create a symlink at `data`

### Pretrained models
- Download all the pretrained models from [google drive](https://drive.google.com/drive/folders/1C3TH9sTTHv18qmGXCOSSUDfhoOcLJsSV?usp=sharing
) and place in pretrained models of root directory.


## Test setup
To test in the val set, run:
```
python main.py
```

## Some tips
- Check `agents\instance_exp.py` to see the Switch Policy
- If you have any questions, feel free to open issues



### Bibtex:
```
@inproceedings{lei2024instance,
  title={Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation},
  author={Lei, Xiaohan and Wang, Min and Zhou, Wengang and Li, Li and Li, Houqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16329--16339},
  year={2024}
}
```

