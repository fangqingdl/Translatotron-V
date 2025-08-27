import random

from parti_pytorch import VitVQGanVAE, trunc_normal_
from parti_pytorch.vit_vqgan_trainer import ImageDataset, cycle
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
from torchvision.utils import make_grid, save_image


def VAEDemo():
    prefix = "/Users/fangqing/PycharmProjects/Translatotron-V2"
    vae_config_path = f"{prefix}/src/config/vit_vqgan_8192cb.json"
    vae_weights_path = f"{prefix}/src/demo/vae.8000.pt"
    image_folder = f"{prefix}/data-build/iwslt14.de-en-images/test_en"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)

    image_size = vae_config['image_size']
    vae = VitVQGanVAE(**vae_config)
    vae.load_state_dict(torch.load(vae_weights_path, map_location='cpu'))
    ds = ImageDataset(image_folder, image_size=image_size)
    dl = cycle(DataLoader(
        ds,
        batch_size=8,
        shuffle=True
    ))
    imgs = next(dl)
    vae.train()
    loss, l1, l2, l3, l4 = vae(imgs, return_loss=True,
                               apply_grad_penalty=True)
    print(loss, l1, l2, l3, l4)
    # imgs_and_recons = torch.stack((imgs, recons), dim=0)
    # imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
    # grid = make_grid(imgs_and_recons, nrow=2, normalize=True, value_range=(0, 1))
    # print(imgs_and_recons)
    # save_image(grid, str('vae_demo.png'))


if __name__ == '__main__':
    VAEDemo()
