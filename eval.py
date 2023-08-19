import os

import torch
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader, get_dataset
from pose_utils import visualize_depth, visualize_normals, to8b
from torch.utils.data import Dataset, DataLoader
import numpy as np


def eval(config):
    save_path = config.log_dir + 'eval/'
    os.makedirs(save_path, exist_ok=True)
    d = get_dataset(config.dataset_name, config.base_dir, split="test", factor=config.factor, device=config.device)
    # make the batchsize height*width, so that one "batch" from the dataloader corresponds to one
    # image used to render a video, and don't shuffle dataset
    batch_size = d.w * d.h
    loader = DataLoader(d, batch_size=batch_size, shuffle=False)
    loader.h = d.h
    loader.w = d.w
    loader.near = d.near
    loader.far = d.far

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        config=config
    )
    best_path = path.join(config.log_dir, "best_ckpt.pt")
    ckpt = torch.load(best_path, map_location=config.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print("Generating Video using", len(loader), "different view points")
    psnrs = 0

    for i, (ray, pixel) in enumerate(loader):
        img, dist, acc = model.render_image(ray, loader.h, loader.w, chunks=config.chunks * 2)
        gt = to8b(pixel.reshape(loader.h, loader.w, 3).numpy())
        save_img = np.concatenate([img, gt,
                             (255. * np.concatenate([((img / 255. - gt / 255.) ** 2).sum(axis=-1, keepdims=True),
                                                    np.zeros([loader.w, loader.h, 2])], axis=-1)).astype(np.uint8)], axis=1)
        psnr = -10.0 * np.log10(((img / 255. - gt / 255.) ** 2).mean())
        psnrs += psnr
        imageio.imsave(save_path + f'img_{i}_psnr_{psnr}.png', save_img)
        if config.visualize_depth:
            imageio.imsave(save_path + f'img_{i}_depth.png', to8b(visualize_depth(dist, acc, loader.near, loader.far)))
        if config.visualize_normals:
            imageio.imsave(save_path + f'img_{i}_normal.png',
                           to8b(visualize_normals(dist, acc)))
    mean_psnr = psnrs / (i + 1)
    imageio.imsave(save_path + f'mean_psnr_{mean_psnr}.png', img)


if __name__ == "__main__":
    config = get_config()
    eval(config)