# Poise-and-React: Continual Test-Time Adaptation in Spiking Neural Networks

<p align="center">
  <img src="./figure/framework.png" width="95%">
</p>

## Overview

Spiking Neural Networks (SNNs) offer compelling advantages in energy efficiency and temporal information processing, yet remain particularly vulnerable to distribution shifts due to their temporal coding and sparse spike transmission mechanisms. While continual test-time adaptation (CTTA) provides a practical means of addressing evolving distribution shifts at inference, existing methods are designed for conventional artificial neural networks and rely on backpropagation-based parameter updates, introducing excessive overhead and accelerating error accumulation and source-domain forgetting when applied to SNNs. To address this, we propose Poise-and-React (PAR), the first CTTA method for SNNs, balancing rapid on-the-fly responsiveness with stable long-horizon refinement. PAR combines two complementary mechanisms: Transient Hebbian Plasticity (THP) for immediate backpropagation-free adaptation and Persistent Hidden-State Adaptation (PHSA) for stable hidden-state refinement with temporal alignment and reliability-aware filtering. This design enables efficient adaptation to evolving shifts with minimal overhead. PAR achieves state-of-the-art performance across multiple distribution-shift benchmarks on SNNs with CNN, Transformer, and ConvLSTM architectures, while requiring 100× fewer trainable parameters and over 30× lower runtime.

---

> 🔥 Tunable modules: THP + PHSA

> ❄️ Frozen modules: backbone

---

## Getting Started

### Requirements

We recommend the following environment (you may adjust based on your setup):

- python >= 3.8
- torch >= 1.13
- torchvision >= 0.14
- torchaudio >= 0.13
- timm
- numpy, tqdm, scikit-learn

### Prepare Data

```bash
mkdir -p ./data
cd ./data
wget -O CIFAR-10-C.tar [https://zenodo.org/record/2535967/files/CIFAR-10-C.tar](https://zenodo.org/record/2535967/files/CIFAR-10-C.tar)
tar -xvf CIFAR-10-C.tar
cd ..
```

### Prepare Pre-trained Models

The pre-trained models are already provided in the `./ckpt/` directory. You can use them directly for evaluation or fine-tuning without needing to download them separately. 
The directory structure should look like this:

```text
ckpt/
├── cifar10vgg9_timestep25_lr0.3_epoch100_leak0.95_bestmodel.pth.tar
```

## Run Test-Time Adaptation
Example: CIFAR-10C
```bash
cd cifar10/bash
bash ours.sh
```

## Acknowledgement
- SPACE code is heavily used. [official](https://github.com/ethanxyluo/SPACE)
- Spike-Driven-Transformer-V3 code is heavily used. [official](https://github.com/BICLab/Spike-Driven-Transformer-V3)
