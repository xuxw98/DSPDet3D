# DSPDet3D: Dynamic Spatial Pruning for 3D Small Object Detection

## Introduction

This repo contains PyTorch implementation for paper [DSPDet3D: Dynamic Spatial Pruning for 3D Small Object Detection](https://arxiv.org/abs/2305.03716) based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

> DSPDet3D: Dynamic Spatial Pruning for 3D Small Object Detection  
> [Xiuwei Xu](https://xuxw98.github.io/), Zhihao Sun, [Ziwei Wang](https://ziweiwangthu.github.io/), Hongmin Liu, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)
>

![teaser](./images/teaser2.png)

## Method
Overall pipeline of DSPDet3D:

![overview](./images/framework.jpg)


## Getting Started
For data preparation and environment setup:
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/data.md)

For training and evaluation:
- [Train, Eval and Visualize](docs/run.md)


## Main Results
We provide the checkpoints for quick reproduction of the results reported in the paper. The pruning threshold can be adjusted freely to tradeoff between accuracy and efficiency without any finetuning.
 Benchmark | mAP@0.25 | mAP@0.5 | Downloads |
 :----: | :----: | :----: | :----: |
 [ScanNet-md40](https://github.com/wyf-ACCEPT/BackToReality) | 65.25 | 53.66 | [model](https://cloud.tsinghua.edu.cn/f/bd49db94cb7548beba63/?dl=1)
 [TO-SCENE-down](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene) | 63.67 | 55.71 | [model](https://cloud.tsinghua.edu.cn/f/0e425d5d053b46c18b73/?dl=1)

Comparison with state-of-the-art methods on TO-SCENE dataset:

<p align="left"><img src="./images/teaser.jpg" alt="drawing" width="70%"/></p>

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
      title={DSPDet3D: Dynamic Spatial Pruning for 3D Small Object Detection}, 
      author={Xiuwei Xu and Zhihao Sun and Ziwei Wang and Hongmin Liu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2305.03716},
      year={2023}
}
```
