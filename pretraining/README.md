# GlyphByT5 Pretraining


This folder contains the code and data used in the **glyph-alignment pretraining** stage. This codebase is developed based on [OpenCLIP](https://github.com/mlfoundations/open_clip). 

**Note: Currently, we release a subset of fonts that we use (see `assets/fonts`) folder, containing 100 with free commercial use license.**

## :wrench: Installation

```
sudo docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

pip install -e .
```

## :mag_right: Training 

Glyph-ByT5 requires 4xA100 GPUs for training. An example training script is provided in [here](pretraining/scripts/train_glyph_byt5.sh).

Run the code:

```
bash scripts/train_glyph_byt5.sh
```
