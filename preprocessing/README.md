<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

We follow the preprocessing code used in [edm2](https://github.com/NVlabs/edm2). In this code we made a several edits: (1) we removed unncessary parts expect preprocessing because this code is only used for preprocessing, (2) we use [-1, 1] range for an input to the stable diffusion VAE (similar to DiT or SiT) unlike edm2 that uses [0, 1] range, and (3) we consider preprocessing to 256x256 resolution.

After downloading ImageNet, please run the following scripts;

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tool.py convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=data/images --resolution=256x256 --transform=center-crop-dhariwal
```

```bash
# Convert the pixel data to VAE latents
python dataset_tool.py encode --source=[YOUR_DATA_PATH]/images \
    --dest=[YOUR_DATA_PATH]/vae-sd
```


## Acknowledgement

This code is mainly built upon [edm2](https://github.com/NVlabs/edm2) repository.