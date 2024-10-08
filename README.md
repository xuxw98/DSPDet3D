# 3D Small Object Detection with Dynamic Spatial Pruning

## Introduction

This repo contains PyTorch implementation for paper [3D Small Object Detection with Dynamic Spatial Pruning](https://arxiv.org/abs/2305.03716) based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Look here for [中文解读](https://zhuanlan.zhihu.com/p/714402773).

> 3D Small Object Detection with Dynamic Spatial Pruning  
> [Xiuwei Xu](https://xuxw98.github.io/)*, Zhihao Sun\*, [Ziwei Wang](https://ziweiwangthu.github.io/), Hongmin Liu, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)
>

![teaser](./images/teaser2.png)

## News
- [2024/7/01]: DSPDet3D is accepted to ECCV 2024!
- [2023/6/04]: We transfer DSPDet3D to extremely large scenes and show great performance! Demo will be released in our [project page](https://xuxw98.github.io/DSPDet3D/).
- [2023/5/01]: Code release.

## Method
Overall pipeline of DSPDet3D:

![overview](./images/pipeline.png)


## Getting Started
For data preparation and environment setup:
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/data.md)

For training and evaluation:
- [Train and Eval](docs/run.md)


## Demo
This is a tutorial on how to use class-agnostic DSPDet3D on custom data and visualize the results. Please download checkpoint from [HERE](https://cloud.tsinghua.edu.cn/f/96549c23580b478a9c64/?dl=1) and move it to `demo` folder.
We provide two demo scenes from ScanNet and Matterport3D. You can download ([ScanNet](https://cloud.tsinghua.edu.cn/f/12fad2697c1644769187/?dl=1), [Matterport3D](https://cloud.tsinghua.edu.cn/f/f6c1446c0e1a437b9b2c/?dl=1)) and also put them into `demo` folder. 
Then run the following command for detection and visualization.

| Dataset  | Scannet                  | Matterport3D                  |
|:--------:|:------------------------:|:-------------------------:|
| Command  | `bash demo/demo.sh demo/scannet.ply demo/config_room.py` | `bash demo/demo.sh demo/mp3d.ply demo/config_building.py` |
| Result   | ![vis](./images/demo_vis1.png)    | ![vis2](./images/demo_vis2.png)    |

You can also try DSPDet3D on your own data in ply format. Run
```
bash demo/demo.sh /path/to/your/ply demo/config_{}.py
```
We use different hyperparamters of 3D NMS for different scales of scenes. For room-size scenes, use `config_room.py`. For building-level scenes, use `config_building.py`. You can also adjust the `prune_threshold` in the config file to tradeoff between accuracy and efficiency.


## Main Results
We provide the checkpoints for quick reproduction of the results reported in the paper. The pruning threshold can be adjusted freely to tradeoff between accuracy and efficiency without any finetuning.
 Benchmark | mAP@0.25 | mAP@0.5 | Downloads |
 :----: | :----: | :----: | :----: |
 [ScanNet-md40](https://github.com/wyf-ACCEPT/BackToReality) | 65.04 | 54.35 | [model](https://cloud.tsinghua.edu.cn/f/bd49db94cb7548beba63/?dl=1)
 [TO-SCENE-down](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene) | 66.12 | 58.55 | [model](https://cloud.tsinghua.edu.cn/f/0e425d5d053b46c18b73/?dl=1)

Comparison with state-of-the-art methods on TO-SCENE dataset:

<p align="left"><img src="./images/teaser.png" alt="drawing" width="70%"/></p>

Visualization results on ScanNet:

![vis](./images/vis.png)

Visualization results on Matterport3D:

![vis2](./images/vis2.png)


## Acknowledgement
We thank a lot for the flexible codebase of [FCAF3D](https://github.com/SamsungLabs/fcaf3d) and valuable datasets provided by [ScanNet](https://github.com/ScanNet/ScanNet) and [TO-SCENE](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene).


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{xu2023dsp, 
      title={3D Small Object Detection with Dynamic Spatial Pruning}, 
      author={Xiuwei Xu and Zhihao Sun and Ziwei Wang and Hongmin Liu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2305.03716},
      year={2023}
}
```
