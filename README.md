<div align="center">
<h1> DFM-Net: A Dual-Frequency Multi-Scale Deformable Network for Medical Image Segmentation </h1>
</div>

##  News

- [2025.9.21] Training and inference code released

##  Abstract

In medical images, the ubiquitous co-occurrence phenomenon makes it difficult for the model to effectively distinguish target features from interfering background information. At the same time, the problems of blurred edge contours and irregular shapes further affect the edge clarity of the segmentation results. To address the above challenges, this paper proposes a new medical image segmentation framework DFM-Net. Our DPFA module performs preliminary spatial frequency decomposition using a frequency-aware block and uses a dual-size patch partitioning strategy to further enhance the perception of local details and global structures. This design effectively eliminates and filters interfering background information across different frequency domains.In addition, we integrate dual-scale convolutional kernels and a bidirectional resampling path, and employ a learnable offset field to dynamically adapt to irregular edge contours. Through extensive experiments on four benchmark medical image datasets, it is demonstrated that our method achieves state-of-the-art performance and effectiveness. 

##  Overview

<div align="center">
<img width="800" alt="image" src="asserts/DFM-Net.png?raw=true">
</div>

The overall architecture of DFM-Net.

##  TODO

- [x] Release code

##  Getting Started

### 1. Install Environment

```
conda create -n DFM-Net python=3.10
conda activate DFM-Net
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), COVID-19 from this [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0), and Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the DFM-Net

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the DFM-Net

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```







<div align="center">
   Visualization comparing DFM-Net with other methods. (a) Input images. (b) Ground truth. (c) DFM-Net(Ours). (d) U-Net. (e) UCTransNet. (f) MLWNet. (g) UltraLight-VMUNet. (h) MFMSA. (i) VPTTA. (j) EMCAD. (k) MambaU-Lite. (l) VM-UNet. (m) H-vmunet. Green lines denote the boundaries of the ground truth.
</div>

##  License

The content of this project itself is licensed under [LICENSE](https://github.com/Anonymous-Submission2025/NetWork/DFM-Net/blob/main/LICENSE).
