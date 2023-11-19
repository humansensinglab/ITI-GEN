# ITI-GEN: Inclusive Text-to-Image Generation

[![paper](https://img.shields.io/badge/paper-iccv2023-cyan)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ITI-GEN_Inclusive_Text-to-Image_Generation_ICCV_2023_paper.html)
[![arXiv](https://img.shields.io/badge/arxiv-2309.05569-red)](https://arxiv.org/abs/2309.05569)
[![video](https://img.shields.io/badge/webpage-green)](https://czhang0528.github.io/iti-gen)
[![poster](https://img.shields.io/badge/poster-orange)](https://czhang0528.github.io/iti-gen/static/images/iti-gen_poster.pdf)


## Overview

<p align="center">
    <img src="docs/teaser.png" width="600px"/>
</p>

> **ITI-GEN: Inclusive Text-to-Image Generation**<br>
> [Cheng Zhang](https://czhang0528.github.io/), 
[Xuanbai Chen](https://xuanbai-chen.github.io/), 
[Siqi Chai](https://www.linkedin.com/in/siqi-chai/), 
[Chen Henry Wu](https://scholar.google.com/citations?hl=en&user=WFKit_4AAAAJ&view_op=list_works&sortby=pubdate),
[Dmitry Lagun](https://www.linkedin.com/in/dmitry-lagun-738b1221/), 
[Thabo Beeler](https://thabobeeler.com/), 
[Fernando De la Torre](https://www.cs.cmu.edu/~ftorre/) <br>
> **ICCV 2023 Oral, Best Paper Finalist**

This repo contains the code for training ITI-GEN and generating images that uniformly span across 
the categories of selected attributes. The main idea behind our approach is leveraging reference images to better represent diverse attributes. 
Key features of our method are:
- Only need datasets that capture the marginal distribution of individual attributes, bypassing the need for datasets that represent joint distributions of multiple attributes.
- The learned token embeddings are generalizable across different generative models.



## Updates

**[Oct 28 2023]** Evaluation code added [here](#evaluation).

**[Sep 18 2023]** Code released. Generation using [Stable Diffusion](https://github.com/CompVis/stable-diffusion) is supported. Will support [ControlNet](https://github.com/lllyasviel/ControlNet), [InstructionPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) later.

**[Sep 11 2023]** Paper released to [Arxiv](https://arxiv.org/pdf/2309.05569.pdf).


## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training ITI-GEN](#training-iti-gen)
- [Prompt Prepending](#optional-prompt-prepending)
- [Generation](#generation)
  - [Stable Diffusion installation](#stable-diffusion-installation)
  - [Image generation](#image-generation)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)




## Installation
The code has been tested with the following environment:
```angular2html
git clone https://github.com/humansensinglab/ITI-GEN.git
cd ITI-GEN
conda env create --name iti-gen --file=environment.yml
source activate iti-gen
```


## Data Preparation

<p align="center">
    <img src="docs/fig_sample.png" width="600px"/>
</p>

1. We provide _processed_ reference images as follows:

|   Dataset    |      Description      |       Attribute Used        |                                        Google Drive                                        |
|:------------:|:---------------------:|:---------------------------:|:------------------------------------------------------------------------------------------:|
|  [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |   Real face images    | 40 binary facial attributes | [Link](https://drive.google.com/file/d/1_wxcrzirofEge4i8LTyYBAL0SMQ_LwGO/view?usp=sharing) | 
| [FairFace](https://github.com/joojs/fairface) |   Real face images    |    Age with 9 categories    | [Link](https://drive.google.com/file/d/1_xtui0b0O52u38jbJzrxW8yRRiBHnZaA/view?usp=sharing) |
|   [FAIR](https://trust.is.tue.mpg.de/)   | Synthetic face images |   Skin tone with 6 categories    | [Link](https://drive.google.com/file/d/1_wiqq7FDByLp8Z4WQOeboSEXYsCzmV76/view?usp=sharing) |
|   [LHQ](https://universome.github.io/alis)    |    Natural scenes     | 11 global scene attributes  | [Link](https://drive.google.com/file/d/1_ypk4ouxQptBevUTcWSp0ZbxvqSZGiKg/view?usp=sharing) |

Save the `.zip` files and unzip the downloaded reference images under ```data/``` directory:
```angular2html
|-- data
|   |-- celeba
|   |   |-- 5_o_Clock_Shadow
|   |   |-- Bald
|   |   |-- ...

|   |-- FAIR_benchmark
|   |   |-- Skin_tone

|   |-- fairface
|   |   |-- Age

|   |-- lhq
|   |   |-- Bright
|   |   |-- Colorful
|   |   |-- ...
```

2. **(Optional)** You can also construct _customized_ reference images under ```data/``` directory:
```angular2html
|-- data
|   |-- custom_dataset_name
|   |   |-- Attribute_1
|   |   |   |-- Category_1
|   |   |   |-- Category_2
|   |   |   |-- ..
|   |   |-- Attribute_2
|   |   |-- ...
```
Modify the corresponding functions in `util.py`.




## Training ITI-GEN

<p align="center">
    <img src="docs/loss.png" width="400px"/>
</p>

**1. Train on human domain (only several minutes)**
```shell
python train_iti_gen.py \
    --prompt='a headshot of a person' \
    --attr-list='Male,Skin_tone,Age' \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
  - `--prompt`: prompt that you want to debias.
  - `--attr_list`: attributes should be selected from `Dataset_name_attribute_list` in `util.py`, separated by commas. Empirically, attributes that are easier to train (less # of category, easier to tell the visual difference between categories) should be put in the front, eg. Male < Young < ... < Skin_Tone < Age.
  - Checkpoints are saved every `save_ckpt_per_epochs`. However, it is NOT always the longer, the better. Better to check every ckpt.

**2. Train on scene domain (only several minutes)**
```shell
python train_iti_gen.py \
    --prompt='a natural scene' \
    --attr-list='Colorful' \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```




## (Optional) Prompt Prepending

<p align="center">
    <img src="docs/fig_framework.png"/>
</p>

**1. Prepend on human domain**
```shell
python prepend.py \
    --prompt='a headshot of a person' \
    --attr-list='Male,Skin_tone,Age' \
    --load-model-epoch=19 \
    --prepended-prompt='a headshot of a doctor'
```
  - `--prompt` and `--attr_list` should be align with those used in training ITI-GEN.
  - `--load_model_epoch` indicates the model's epoch you want to load.
  - `--prepended_prompt`: prepend the learnt tokens after this prompt to implement Train-Once-For-All Generation. In human domain, `prompt` and `prepended_prompt` should not differ a lot, better to solely change the occupation.

**2. Prepend on scene domain**
```shell
python prepend.py \
    --prompt='a natural scene' \
    --attr-list='Colorful' \
    --load-model-epoch=19 \
    --prepended-prompt='an alien pyramid landscape, art station, landscape, concept art, illustration, highly detailed artwork cinematic'
```



## Generation
Our ITI-GEN training is standalone from the generative models such as Stable Diffusion, ControlNet, and InstructionPix2Pix.
Here we show one example how to use ITI-GEN to generate images with Stable Diffusion.

### Stable Diffusion installation
```shell
cd models
git clone https://github.com/CompVis/stable-diffusion.git
# ITI-GEN has been tested with this version: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
# Due to licence issues, we cannot share the pre-trained checkpoints directly.
# Download it yourself and put the Stable Diffusion checkpoints at <path/to/sd-v1-4.ckpt>.
mv stable-diffusion sd
mkdir -p sd/models/ldm/stable-diffusion-v1/
ln -s <path/to/sd-v1-4.ckpt> sd/models/ldm/stable-diffusion-v1/model.ckpt
cd sd
pip install -e .
cd ../..
```

### Image generation

**1. Generation on the human domain**

<p align="center">
  <img src="docs/multi_category.png" style="margin-right: 10px;"  width="370px">
  <img src="docs/multi_category_man.png" width="370px">
</p>

```shell
python generation.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --attr-list='Male,Skin_tone,Age' \
    --outdir='./ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/original_prompt_embedding/sample_results' \
    --prompt-path='./ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/original_prompt_embedding/basis_final_embed_19.pt' \
    --n_iter=5 \
    --n_rows=5 \
    --n_samples=1
```
- `--config`: config file for Stable Diffusion.
- `--ckpt`: path to the pre-trained Stable Diffusion checkpoint.
- `--plms`: whether to use the plms sampling.
- `--attr_list`: attributes should be selected from `Dataset_name_attribute_list` in `util.py`, separated by commas. This should align with the attribute list used in training ITI-GEN.
- `--outdir`: output directory of the generated images.
- `--prompt_path`: path to the learnt prompt embeddings with ITI-GEN.
- `--n_iter`: number of iterations for the diffusion sampling.
- `--n_rows`: number of rows in the output image grid.
- `--n_samples`: number of samples per row.


**2. Generation on the scene domain**

<p align="center">
  <img src="docs/scene_3.png" style="margin-right: 10px;"  width="370px">
  <img src="docs/scene_4.png" width="370px">
</p>

```shell
python generation.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --attr-list='Colorful' \
    --outdir='./ckpts/a_natural_scene_Colorful/original_prompt_embedding/sample_results' \
    --prompt-path='./ckpts/a_natural_scene_Colorful/original_prompt_embedding/basis_final_embed_19.pt' \
    --n_iter=5 \
    --n_rows=5 \
    --n_samples=1
```

We are actively adding more features to this repo. Please stay tuned!


## Evaluation
We show using CLIP, which is found superior to the pre-trained classifiers, for evaluating most of the attributes. 
When it might be erroneous for some attributes, we combine the CLIP results with human evaluations.
The output for this script contains the quantitative results of both `KL divergence` and `FID` score, supported by [CleanFID](https://github.com/GaParmar/clean-fid).

```shell
python evaluation.py \
    --img-folder '/path/to/image/folder/you/want/to/evaluate' \
    --class-list 'a headshot of a person wearing eyeglasses' 'a headshot of a person'
```
- `--img_folder`: the image folder that you want to evaluate.
- `--class_list`: the class prompts used for evaluation, separated by a space. The length of the list depends on the number of category combinations for different attributes. In terms of writing evaluation prompts for CelebA attributes, please refer (but not limited) to Table A3 in the supplementary materials.

We should notice FID score can be affected by various factors such as the image number. 
Each FID score in our paper is computed using images over 5K. For sanity check, 
we suggest directly comparing with the FID score of the images from baseline Stable Diffusion in the same setup. 
Please refer to Section 4.1 Quantitative Metrics in the main paper and Section D in the supplementary materials for more details.

## Acknowledgements
- Models
  - [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
  - [ControlNet](https://github.com/lllyasviel/ControlNet)
  - [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
- Data acquisition and processing
  - [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [FairFace](https://github.com/joojs/fairface)
  - [FAIR](https://trust.is.tue.mpg.de/)
  - [LHQ](https://universome.github.io/alis)
  - [CLIP-IQA](https://github.com/IceClear/CLIP-IQA)

## Citation
If you find this repo useful, please cite:
```
@inproceedings{zhang2023inclusive,
  title={{ITI-GEN}: Inclusive Text-to-Image Generation},
  author={Zhang, Cheng and Chen, Xuanbai and Chai, Siqi and Wu, Henry Chen and Lagun, Dmitry and Beeler, Thabo and De la Torre, Fernando},
  booktitle = {ICCV},
  year={2023}
}
```

## License
We use the X11 License. This license is identical to the MIT License, 
but with an extra sentence that prohibits using the copyright holders' names (Carnegie Mellon University and Google in our case) for 
advertising or promotional purposes without written permission.
