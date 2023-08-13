import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_model(config):
    ckpt_save_path = path.join(config.log_dir, "ckpt.pt")
    best_path = path.join(config.log_dir, "best_ckpt.pt")
    data = iter(cycle(
        get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor,
                       batch_size=config.batch_size, shuffle=True, device=config.device, config=config)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test",
                                              factor=config.factor, batch_size=config.chunks, shuffle=True,
                                              device=config.device, config=config)))

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
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps,
                           lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    best_psnr = 0.0
    if config.continue_training:
        ckpt = torch.load(ckpt_save_path, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.last_epoch = ckpt['epoch']
        best_psnr = ckpt['best_psnr']
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)

    for step in tqdm(range(scheduler.last_epoch, config.max_steps)):
        rays, pixels = next(data)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)
        if step % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr_sum_c, psnr_sum_r = 0, 0
                for i in range(10):
                    psnr = eval_model(config, model, eval_data)
                    psnr = psnr.detach().cpu().numpy()
                    psnr_sum_c += psnr[0]
                    psnr_sum_r += psnr[1]
                psnr_sum_c /= 10
                psnr_sum_r /= 10
                logger.add_scalar('eval/coarse_psnr', float(psnr_sum_c), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr_sum_r), global_step=step)
                ckpt = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'epoch': step + 1,
                    'best_psnr': max(best_psnr, float(psnr[-1]))
                }
                torch.save(ckpt, ckpt_save_path)
                if float(psnr[-1]) > best_psnr:
                    best_psnr = float(psnr[-1])
                    torch.save(ckpt, best_path)


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3]) ** 2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)
