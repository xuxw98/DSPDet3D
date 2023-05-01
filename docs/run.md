**Training**

To start training, run [dist_train](../tools/dist_train.sh) with `DSPDet3D` configs.  
For ScanNet-md40:
```shell
bash tools/dist_train.sh configs/dspdet3d/dspdet3d_scannet-3d-22class.py 2 --work-dir work_dirs/scannet_md40
```
For TO-SCENE-down:
```shell
bash tools/dist_train.sh configs/dspdet3d/dspdet3d_toscene-3d-70class.py 2 --work-dir work_dirs/toscene_down
```

**Testing**

Test pre-trained model using [dist_test](../tools/dist_test.sh) with `DSPDet3D` configs. We also provide checkpoints used in the paper for easy reproduction.   
For ScanNet-md40:
```shell
bash tools/dist_test.sh configs/dspdet3d/dspdet3d_scannet-3d-22class.py work_dirs/scannet_md40/latest.pth 2
```
For TO-SCENE-down:
```shell
bash tools/dist_test.sh configs/dspdet3d/dspdet3d_toscene-3d-70class.py work_dirs/toscene_down/latest.pth 2
```