## Processed data
The processed data can be downloaded from [HERE](https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/).  
Run `cat mmdet_xxx.tar.* > mmdet_xxx.tar` to merge the files.  
Then set the variable `data_root` in configs to the path of directory that contains the `.pkl` files.

## ScanNet-md40

**Step 1.** Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the `scans/` folder to `DSPDet3D/data/ScanNet-md40/`.

**Step 2.** Following [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/data/scannet) to process the data.  
First extract point clouds and annotations by running:
```
cd DSPDet3D/data/ScanNet-md40
python batch_load_scannet_data.py
```
Then use `tools/create_data.py` from `mmdetection3d` to generate `.pkl` files.
```
python DSPDet3D/tools/create_data.py scannet --root-path DSPDet3D/data/ScanNet-md40 --out-dir DSPDet3D/data/ScanNet-md40 --extra-tag scannet
```

**Final folder structure:**

```
ScanNet-md40
├── instance_mask/
├── points/
├── seg_info/
├── semantic_mask/
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
└── ...
```

## TO-SCENE-down

**Step 1.** Download TO-SCENE dataset (TO_ScanNet version) from [HERE](https://drive.google.com/file/d/12IVVEt5kUQrz0_Qis58TH6Fj4yr6cJTo/view). Download `meta_data` from [HERE](https://drive.google.com/file/d/16E1Gb91ctGysmWhbeUwYF3-ssQ1Dw0Rm/view) and move it into `TO-scannet/`.

The folder structure:
```
TO-scannet
├── meta_data/
├── train/
├── val/
└── test/
```
Link or move this folder to `DSPDet3D/data/TO-SCENE-down/`.

**Step 2.** Process the data by running: 
```
cd DSPDet3D/data/TO-SCENE-down
python to-scannet_converter.py
```

**Final folder structure:**

```
TO-SCENE-down
├── instance_mask/
├── points/
├── seg_info/
├── semantic_mask/
├── toscene_infos_train.pkl
├── toscene_infos_val.pkl
└── ...
```