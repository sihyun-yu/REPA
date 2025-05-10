import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def center_crop(width, height, img):
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    return np.array(img)


def center_crop_wide(width, height, img):
    ch = int(np.round(width * img.shape[0] / img.shape[1]))
    if img.shape[1] < width or ch < height:
        return None

    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    img = np.array(img)

    canvas = np.zeros([width, width, 3], dtype=np.uint8)
    canvas[(width - height) // 2 : (width + height) // 2, :] = img
    return canvas


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    """
    Preprocess data for training.
    """
    assert torch.cuda.is_available(), "Preprocessing currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.dest, exist_ok=True)

    # Create model:
    vae = AutoencoderKL.from_pretrained(args.model_url).to(device)

    if args.transform == "center-crop":
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop(args.resolution, args.resolution, pil_image)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    elif args.transform == "center-crop-wide":
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_wide(args.resolution, args.resolution, pil_image)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    elif args.transform == "center-crop-dhariwal":
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        raise ValueError(f"Invalid transform: {args.transform}")
    dataset = ImageFolder(args.source, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_steps = 0
    labels = []
    labels_2 = []
    dest_images = ""
    for x, y in tqdm(loader, total=len(loader), disable=rank != 0):
        if args.convert:
            index_list = []
            image_list = []
            for i in range(len(x)):
                x_ = x[i]
                y_ = y[i]

                idx = total_steps * args.batch_size * dist.get_world_size() + i * dist.get_world_size() + rank
                idx_str = f"{idx:08d}"
                archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

                x_ = x_.cpu().permute(1, 2, 0).numpy()
                x_ = ((x_ + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                x_ = PIL.Image.fromarray(x_)
                index_list.append(archive_fname)
                image_list.append(x_)
                labels.append([archive_fname, y_.item()])
            with ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 3)) as executor:
                for index, image in zip(index_list, image_list):
                    os.makedirs(os.path.join(args.dest, os.path.dirname(index)), exist_ok=True)
                    executor.submit(image.save, os.path.join(args.dest, index))
        else:
            x = x.to(device)
            with torch.no_grad():
                d = vae.encode(x).latent_dist
                z = torch.cat([d.mean, d.std], dim=1)
            index_list = []
            index_list_2 = []
            image_list = []
            latent_list = []
            for i in range(len(z)):
                x_ = x[i]
                z_ = z[i]
                y_ = y[i]

                idx = total_steps * args.batch_size * dist.get_world_size() + i * dist.get_world_size() + rank
                idx_str = f"{idx:08d}"
                archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
                archive_fname_2 = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

                x_ = x_.cpu().permute(1, 2, 0).numpy()
                x_ = ((x_ + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                x_ = PIL.Image.fromarray(x_)
                z_ = z_.cpu().numpy()
                index_list.append(archive_fname)
                index_list_2.append(archive_fname_2)
                image_list.append(x_)
                latent_list.append(z_)
                labels.append([archive_fname_2, y_.item()])
                labels_2.append([archive_fname, y_.item()])
            with ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 3)) as executor:
                for index, index_2, image, latent in zip(index_list, index_list_2, image_list, latent_list):
                    if args.dest_images == "" and not args.no_images:
                        dest_images = os.path.join(os.path.dirname(args.dest), "images")
                        os.makedirs(os.path.join(dest_images, os.path.dirname(index)), exist_ok=True)
                    elif args.dest_images != "" and not args.no_images:
                        dest_images = args.dest_images
                        os.makedirs(os.path.join(dest_images, os.path.dirname(index)), exist_ok=True)
                    os.makedirs(os.path.join(args.dest, os.path.dirname(index_2)), exist_ok=True)
                    if not args.no_images:
                        executor.submit(image.save, os.path.join(dest_images, index))
                    executor.submit(np.save, os.path.join(args.dest, index_2), latent)

        total_steps += 1
    
    world_size = dist.get_world_size()
    gather_labels = [None for _ in range(world_size)]
    gather_labels_2 = [None for _ in range(world_size)]
    dist.all_gather_object(gather_labels, labels)
    dist.all_gather_object(gather_labels_2, labels_2)

    if rank == 0:
        labels = list(chain(*gather_labels))
        labels_2 = list(chain(*gather_labels_2))

        labels = sorted(labels, key=lambda x: x[0])
        with open(os.path.join(args.dest, "dataset.json"), "w") as f:
            json.dump({"labels": labels}, f)
        if not args.convert and not args.no_images:
            labels_2 = sorted(labels_2, key=lambda x: x[0])
            with open(os.path.join(dest_images, "dataset.json"), "w") as f:
                json.dump({"labels": labels_2}, f)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--dest-images", type=str, default="")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--transform", type=str, choices=["center-crop", "center-crop-wide", "center-crop-dhariwal"], default="center-crop-dhariwal")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
