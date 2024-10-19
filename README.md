<h1 align="center"> Representation Alignment for Generation: <br>Training Diffusion Transformers Is Easier Than You Think
</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg)](https://arxiv.org/abs/2410.06940)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-alignment-for-generation/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=representation-alignment-for-generation)

<div align="center">
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://www.linkedin.com/in/SangkyungKwak/" target="_blank">Sangkyung&nbsp;Kwak</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://jh-jeong.github.io/" target="_blank">Jongheon&nbsp;Jeong</a><sup>2</sup>
  <br>
  <a href="http://jonathan-huang.org/" target="_blank">Jonathan&nbsp;Huang</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1*</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://www.sainingxie.com/" target="_blank">Saining&nbsp;Xie</a><sup>4*</sup><br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>Korea University &emsp; <sup>3</sup>Scaled Foundations &emsp; <sup>4</sup>New York University &emsp; <br>
  <sup>*</sup>Equal Advising &emsp; <br>
</div>
<h3 align="center">[<a href="https://sihyun.me/REPA">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2410.06940">arXiv</a>]</h3>
<br>

<b>Summary</b>: We propose REPresentation Alignment (REPA), a method that aligns noisy input states in diffusion models with representations from pretrained visual encoders. This significantly improves training efficiency and generation quality. REPA speeds up SiT training by 17.5x and achieves state-of-the-art FID=1.42.

### 1. Environment setup

```bash
conda create -n repa python=3.9 -y
conda activate repa
pip install -r requirements.txt
```

### 2. Dataset

#### Dataset download

Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specifiy it via `--data-path` arguments in training scripts. Please refer to our [preprocessing guide](https://github.com/sihyun-yu/REPA/tree/master/preprocessing).

### 3. Training

```bash
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8" \
  --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--enc-type`: `[dinov2-vit-b, dinov2-vit-l, dinov2-vit-g, dinov1-vit-b, mocov3-vit-b, , mocov3-vit-l, clip-vit-L, jepa-vit-h, mae-vit-l]`
- `--proj-coeff`: Any values larger than 0
- `--encoder-depth`: Any values between 1 to the depth of the model
- `--output-dir`: Any directory that you want to save checkpoints and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)

For DINOv2 models, it will be automatically downloaded from `torch.hub`. For CLIP models, it will be also automatically downloaded from the CLIP repository. For other pretrained visual encoders, please download the model weights from the below links and place into the following directories with these names:

- `dinov1`: Download the ViT-B/16 model from the [`DINO`](https://github.com/facebookresearch/dino) repository and place it as `./ckpts/dinov1_vitb.pth`
- `mocov3`: Download the ViT-B/16 or ViT-L/16 model from the [`RCG`](https://github.com/LTH14/rcg) repository and place them as `./ckpts/mocov3_vitb.pth` or `./ckpts/mocov3_vitl.pth`
- `jepa`: Download the ViT-H/14 model (ImageNet-1K) from the [`I-JEPA`](https://github.com/facebookresearch/ijepa) repository and place it as `./ckpts/ijepa_vith.pth`
- `mae`: Download the ViT-L model from [`MAE`](https://github.com/facebookresearch/mae) repository and place it as `./ckpts/mae_vitl.pth`

### 4. Evaluation
You can generate images (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt YOUR_CHECKPOINT_PATH \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7
```

We also provide the SiT-XL/2 checkpoint (trained for 4M iterations) used in the final evaluation. It will be automatically downloaded if you do not specify `--ckpt`.

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [RCG](https://github.com/LTH14/rcg) repositories.\
We also appreciate [Kyungmin Lee](https://kyungmnlee.github.io/) for providing the initial version of the implementation.

## BibTeX

```bibtex
@article{yu2024repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2024},
  journal={arXiv preprint arXiv:2410.06940},
}
```

