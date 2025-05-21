<div align="center">
<h1> MFS-Net: Medical Image Segmentation via Frequency Domain Dual-Size Patch Partitioning and Multi-Scale Deformable Learning </h1>
</div>

##  News

- [2025.5.21] Training and inference code released

##  Abstract

The co-occurrence phenomenon in medical images makes it difficult for the model to distinguish target features from background interference. In addition, the problem of blurred and irregular edge contours is particularly common in medical image segmentation, which usually leads to unclear edges in the segmentation results. To address these problems, this paper proposes a novel medical image segmentation framework MFS-Net. The network adopts a dual-branch feature fusion module (DFFM) as its backbone. 
In the DPFE block, we propose a novel patch partitioning strategy, which uses the design of dual-size patches to efficiently separate and capture high-frequency textures and low-frequency overall structures in the frequency domain. And by introducing a learnable quantization matrix W, the key frequency components are adaptively screened and amplified to suppress noise interference. In addition, we use multi-scale deformable learning to dynamically adjust the sampling grid to accurately capture multi-scale edge features that are robust to boundary blur. 
Through extensive experiments on four medical image datasets, it is demonstrated that our method achieves state-of-the-art performance and universality.

##  Introduction

<div align="center">
    <img width="800" alt="image" src="asserts/challen_.png?raw=true">
</div>

Major challenges in medical image segmentation.

##  Overview

<div align="center">
<img width="800" alt="image" src="asserts/MFS-Net.png?raw=true">
</div>

The overall framework of the proposed MFS-Net. Our network architecture adopts a dual-branch structure and consists of four levels. Each level contains a multi-frequency noise suppression (MFNS) module and a multi-scale edge enhancement (MSEE) module. MFNS contains our proposed dual patch Fourier enhancement (DPFE). Each MSEE contains a multi-scale deformable convolution (MSDC) module and a spatial channel attention mechanism.

##  TODO

- [x] Release code

##  Getting Started

### 1. Install Environment

```
conda create -n MFS-Net python=3.10
conda activate MFS-Net
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

### 3. Train the MFS-Net

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the MFS-Net

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```


##  Quantitative comparison

<div align="center">
<img width="800" alt="image" src="asserts/compara.jpg?raw=true">
</div>

<div align="center">
    Comparison with other methods on the ISIC2018, Kvasir, COVID-19 and Moun-Seg datasets.
</div>


##  Visualization

<div align="center">
<img width="800" alt="image" src="asserts/Visualization.jpg?raw=true">
</div>



<div align="center">
   Visualization comparing MFS-Net with other methods. (a) Input images. (b) Ground truth. (c) MFS-Net(Ours). (d) U-Net. (e) UCTransNet. (f) MLWNet. (g) UltraLight-VMUNet. (h) MFMSA. (i) VPTTA. (j) EMCAD. (k) MambaU-Lite. (l) VM-UNet. (m) H-vmunet. Green lines denote the boundaries of the ground truth.
</div>

##  License

The content of this project itself is licensed under [LICENSE](https://github.com/Anonymous-Submission2025/NetWork/MFS-Net/blob/main/LICENSE).
