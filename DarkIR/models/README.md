---
license: cc-by-4.0
pipeline_tag: image-to-image
---

# DarkIR: Robust Low-Light Image Restoration

[![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/Cidaut/DarkIR) 
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://huggingface.co/papers/2412.13443)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github)](https://github.com/cidautai/DarkIR)

This repository contains the official model for the paper [DarkIR: Robust Low-Light Image Restoration](https://huggingface.co/papers/2412.13443), presented at CVPR 2025.

**[Daniel Feijoo](https://scholar.google.com/citations?hl=en&user=hqbPn4YAAAAJ), [Juan C. Benito](https://scholar.google.com/citations?hl=en&user=f186MIUAAAAJ), [Alvaro Garcia](https://scholar.google.com/citations?hl=en&user=c6SJPnMAAAAJ), [Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en)** (CIDAUT AI and University of Wuerzburg)

**TLDR.** Photography in low-light conditions often results in noisy and blurry images. While previous methods address these issues separately, DarkIR proposes the first all-in-one approach for low-light restoration, including illumination, noise, and blur enhancement with a single model.

<details>
<summary> <b> ABSTRACT </b> </summary>
Photography during night or in dark conditions typically suffers from noise, low light and blurring issues due to the dim environment and the common use of long exposure. Although Deblurring and Low-light Image Enhancement (LLIE) are related under these conditions, most approaches in image restoration solve these tasks separately. In this paper, we present an efficient and robust neural network for multi-task low-light image restoration. Instead of following the current tendency of Transformer-based models, we propose new attention mechanisms to enhance the receptive field of efficient CNNs. Our method reduces the computational costs in terms of parameters and MAC operations compared to previous methods. Our model, DarkIR, achieves new state-of-the-art results on the popular LOLBlur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night and dark images.
</details>

DarkIR achieves new state-of-the-art results on popular datasets like LOLBlur, LOLv2, and Real-LOLBlur, demonstrating strong generalization capabilities on real-world night and dark images.

| <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/0085_low.png" alt="Low-light w/ blur" width="450"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/0085_retinexformer.png" alt="RetinexFormer" width="450"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/0085_darkir.png" alt="DarkIR (ours)" width="450"> |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Low-light w/ blur                | RetinexFormer                 | **DarkIR** (ours)    |
| <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/low00747.png" alt="Low-light w/o blur" width="450"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/low00747_lednet.png" alt="LEDNet" width="450"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/teaser/low00747_darkir.png" alt="DarkIR (ours)" width="450"> |
| Low-light w/o blur                 | LEDNet    | **DarkIR** (ours)                 |

&nbsp;

## Network Architecture

![Network Architecture](https://github.com/cidautai/DarkIR/raw/main/assets/networks-scheme.png)

## Dependencies and Installation

- Python == 3.10.12
- PyTorch == 2.5.1
- CUDA == 12.4
- Other required packages in `requirements.txt`

```bash
# git clone this repository
git clone https://github.com/Fundacion-Cidaut/DarkIR.git
cd DarkIR

# create python environment
python3 -m venv venv_DarkIR
source venv_DarkIR/bin/activate

# install python dependencies
pip install -r requirements.txt
```

## Datasets
The datasets used for training and/or evaluation are:

|Dataset     | Sets of images | Source  |
| -----------| :---------------:|------|
|LOL-Blur    | 10200 training pairs / 1800 test pairs| [LEDNet](https://github.com/sczhou/LEDNet) |
|LOLv2-real        | 689 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|LOLv2-synth        | 900 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|LOL      | 485 training pairs / 15 test pairs | [Official Site](https://daooshee.github.io/BMVC2018website/)  |
|Real-LOLBlur | 1354 unpaired images  | [LEDNet](https://github.com/sczhou/LEDNet)  |
|LSRW-Nikon | 3150 training pairs / 20 test pairs | [R2RNet](https://github.com/JianghaiSCU/R2RNet) |
|LSRW-Huawei | 2450 training pairs / 30 test pairs | [R2RNet](https://github.com/JianghaiSCU/R2RNet) |

You can download each specific dataset and put it on the `/data/datasets` folder for testing. 

## Results 
We present results in different datasets for DarkIR of different sizes. While **DarkIR-m** has channel depth of 32, 3.31 M parameters and 7.25 GMACs, **DarkIR-l** has channel depth 64, 12.96 M parameters and 27.19 GMACs.

|Dataset     | Model| PSNR| SSIM  | LPIPS |
| -----------| :---------------:|:------:|------|------|
|LOL-Blur    | DarkIR-m| 27.00| 0.883| 0.162|
|   | DarkIR-l| 27.30| 0.898| 0.137|
|LOLv2-real  | DarkIR-m| 23.87| 0.880| 0.186|
|LOLv2-synth | DarkIR-m| 25.54| 0.934| 0.058|
|LSRW-Both | DarkIR-m| 18.93| 0.583| 0.412|

We present perceptual metrics for Real-LOLBlur dataset:

| Model| MUSIQ| NRQM  | NIQE |
| -----------| :---------------:|:------:|:------:|
| DarkIR-m| 48.36| 4.983| 4.998|
| DarkIR-l| 48.79| 4.917| 5.051|

## Evaluation

To check our results you could run the evaluation of DarkIR in each of the datasets:

- Download the weights of the model from [OneDrive](https://cidautes-my.sharepoint.com/:f:/g/personal/alvgar_cidaut_es/Epntbl4SucFNpeIT_jyYZ-cB9BamMbacbyq_svrkMCpShA?e=XB9YBB) and put them in `/models`.
- run `python testing.py -p ./options/test/<config.yml>`. Default is LOLBlur.

You may also check the qualitative results in `Real-LOLBlur` and LLIE unpaired by running `python testing_unpaired.py -p ./options/test/<config.yml>`. Default is RealBlur.

## Inference

You can restore a whole set of images in a folder by running: 

```bash
python inference.py -i <folder_path>
```

Restored images will be saved in `./images/results`.

To inference a video you can run

```bash
python inference_video.py -i /path/to/video.mp4
```

which will be saved in `./videos/results`.

## Gallery

<p align="center"> <strong>  LOLv2-real </strong> </p>

| <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2real/low00733_low.png" alt="Low-light" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2real/00733_snr.png" alt="SNR-Net" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2real/low00733_retinexformer.png" alt="RetinexFormer" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2real/low00733_darkir.png" alt="DarkIR (ours)" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2real/normal00733.png" alt="Ground Truth" width="300"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| Low-light                | SNR-Net | RetinexFormer    | **DarkIR** (ours) | Ground Truth                 |

&nbsp;

<p align="center"> <strong>  LOLv2-synth </strong> </p>

| <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2synth/r13073518t_low.png" alt="Low-light" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2synth/r13073518t_snr.png" alt="SNR-Net" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2synth/r13073518t_retinexformer.png" alt="RetinexFormer" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2synth/r13073518t_darkir.png" alt="DarkIR (ours)" width="300"> | <img src="https://github.com/cidautai/DarkIR/raw/main/assets/lolv2synth/r13073518t_normal.png" alt="Ground Truth" width="300"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| Low-light                | SNR-Net | RetinexFormer    | **DarkIR** (ours) | Ground Truth                 |

&nbsp;

<p align="center"> <strong>  Real-LOLBlur-Night </strong> </p>

<p align="center">  <img src="https://github.com/cidautai/DarkIR/raw/main/assets/qualis_realblur_night.jpg" alt="Example Image" width="70%"> </p>

## Citation and acknowledgement

This work has been accepted for publication and presentation at The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025.

```bibtex
@InProceedings{Feijoo_2025_CVPR,
    author    = {Feijoo, Daniel and Benito, Juan C. and Garcia, Alvaro and Conde, Marcos V.},
    title     = {DarkIR: Robust Low-Light Image Restoration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10879-10889}
}
```

## Contact

If you have any questions, please contact danfei@cidaut.es and marcos.conde@uni-wuerzburg.de