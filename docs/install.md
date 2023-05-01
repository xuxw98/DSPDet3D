The code is tested with Python3.8, PyTorch == 1.10.1, CUDA == 11.3, mmdet3d == 1.0.0rc6, mmcv_full == 1.6.2, mmdet == 2.26.0 and MinkowskiEngine == 0.5.4. We recommend you to use anaconda to make sure that all dependencies are in place. 

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmdet3d python=3.8 -y
conda activate mmdet3d
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
conda install pytorch torchvision -c pytorch
```

**Step 3.** Following [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/get_started.md) to install pytorch, mmcv, mmdet and mmdet3d.

**Step 4.** Install MinkowskiEngine.
```shell
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
```