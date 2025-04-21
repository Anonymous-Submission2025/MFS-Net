<div align="center">
<h1> MFS-Net: A General Medical Image Segmentation Framework via Multi-Frequency and Multi-Scale Feature Fusion </h1>
</div>

## 🎈 News

- [2025.4.21] Training and inference code released

## ⭐ Abstract

Medical image segmentation (MIS) is crucial in improving clinical diagnostic accuracy and reducing the risk of misdiagnosis. However, it still faces two major challenges. On the one hand, co-occurrence phenomena often occur in medical image processing, which can cause noise interference and make the model unable to distinguish between target features and co-occurrence interference information, resulting in model segmentation errors. On the other hand, fuzzy boundaries and low contrast are commonly present in medical images, and irregular and blurry edge information can lead to unclear edge segmentation. To address these challenges, we propose a universal framework called Multi-Frequency and Multi-Scale Medical Image Segmentation (MFS-Net), which is based on the U-Net architecture and innovatively proposed a Dual Branch Feature Fusion Module (DFFM) to extract and fuse multi frequency and multi-scale features separately. Firstly, we designed a Multi Frequency Noise Suppression (MPNS) module that combines attention mechanism with Fast Fourier Transform (FFT), utilizing FFT to optimize the fusion of details and global information in the frequency domain, achieving the optimal fusion of noise suppression and effective features. Secondly, a multi-scale edge enhancement (MSEE) module for fuzzy boundary problems is proposed, which dynamically adjusts the receptive field through deformable convolution and channel space attention mechanism to accurately capture multi-scale edge features. 
Through extensive experiments on four medical image datasets, the most advanced performance of our method is proved, and its progressiveness and universal applicability to various medical image segmentation scenes are proved.

## 🚀 Introduction

<div align="center">
    <img width="800" alt="image" src="asserts/challen_.jpg?raw=true">
</div>

Major challenges in medical image segmentation.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/MFS-net.jpg?raw=true">
</div>

Overall framework of the proposed MFS-Net. (a) DFFM is Dual Branch Feature Fusion Module. (b) MFNS is Multi-Frequency Noise Suppression. (c) MSEE is Multi-Scale Edge Enhancement. (d) FEB is Feature Enhancement Block. (e) Spectral is A module constructed using fast Fourier transform(FFT).


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

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


## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="asserts/compara.jpg?raw=true">
</div>

<div align="center">
    Comparison with other methods on the ISIC2018, Kvasir, COVID-19 and Moun-Seg datasets.
</div>


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/Visualization.jpg?raw=true">
</div>



<div align="center">
    Qualitative comparison of other methods and MFS-Net. (a) Input images. (b) Ground truth. (c) MFS-Net(Ours). (d) U-Net. (e) UCTransNet. (f) MLWNet. (g) UltraLight-VMUNet. (h) MFMSA. (i) VPTTA. (j) EMCAD. (k) MambaU-Lite. (l) VM-UNet. (m) H-vmunet. Green lines denote the boundaries of the ground truth.
</div>

## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/Anonymous-Submission2025/NetWork/MFS-Net/blob/main/LICENSE).
