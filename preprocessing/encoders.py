# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass
#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.

@persistence.persistent_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = 'stabilityai/sd-vae-ft-mse',  # Name of the VAE to use.
        batch_size  = 8,                            # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)['latent_dist']
        return torch.cat([d.mean, d.std], dim=1)

    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 127.5 - 1
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

#----------------------------------------------------------------------------

def load_stability_vae(vae_name='stabilityai/sd-vae-ft-mse', device=torch.device('cpu')):
    import dnnlib
    cache_dir = dnnlib.make_cache_dir_path('diffusers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = cache_dir

    import diffusers # pip install diffusers # pyright: ignore [reportMissingImports]
    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        vae = diffusers.models.AutoencoderKL.from_pretrained(
            vae_name, cache_dir=cache_dir, local_files_only=True
            )
    except:
        # Could not load the model from cache; try without local_files_only.
        vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir)
    return vae.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------