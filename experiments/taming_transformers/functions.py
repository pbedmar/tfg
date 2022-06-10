import io
import requests
import glob
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2


# functions used to load and preprocess models -- included with taming-transformers
def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec


# auxiliary functions to sample images
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf", 10)
def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=192, map_dalle=True):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    print("Unsqueezed:", img.shape)
    return img


def stack_reconstructions(images, titles=[]):
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images) * w, h))
    for i, image in enumerate(images):
        img.paste(image, (i * w, 0))

    for i, title in enumerate(titles):
        ImageDraw.Draw(img).text((i * w, 0), f'{title}', (0, 0, 0), font=font)  # coordinates, text, color, font
    return img


# reconstruction pipelines to build grid of images
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def reconstruction_pipeline_fixed_data(file, models, titles, origin_format=0, size=192):
    if origin_format == 0:
        x_vqgan = preprocess(download_image(file), target_image_size=size, map_dalle=False)
    elif origin_format == 1:
        file = PIL.Image.open(file)
        x_vqgan = preprocess(file, target_image_size=size, map_dalle=False)
    else:
        x_vqgan = file

    x_vqgan = x_vqgan.to(DEVICE)

    print(f"input is of size: {x_vqgan.shape}")

    pils = []
    pils.append(custom_to_pil(preprocess_vqgan(x_vqgan[0])))
    for m in models:
        pils.append(custom_to_pil(reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), m)[0]))

    img = stack_reconstructions(pils, titles)
    return img


def reconstruction_pipeline_fixed_model(model, files, title, size=192):
    pils = []

    for f in files:
        if f[1] == 0:
            x_vqgan = preprocess(download_image(f[0]), target_image_size=size, map_dalle=False)
        elif f[1] == 1:
            file = PIL.Image.open(f[0])
            x_vqgan = preprocess(file, target_image_size=size, map_dalle=False)
        else:
            x_vqgan = f[0]

        x_vqgan = x_vqgan.to(DEVICE)

        pils.append((custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                     custom_to_pil(reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model)[0])))

    img = Image.new("RGB", (2 * size, len(pils) * size))
    for i, image in enumerate(pils):
        img.paste(image[0], (0, i * size))
        img.paste(image[1], (size, i * size))

    ImageDraw.Draw(img).text((0, 0), f'{title}', (0, 0, 0), font=font)  # coordinates, text, color, font

    return img


def reconstruction_pipeline(models, files, titles, size=192):
    img = Image.new("RGB", ((1 + len(models)) * size, len(files) * size))

    for i, f in enumerate(files):
        if f[1] == 0:
            x_vqgan = preprocess(download_image(f[0]), target_image_size=size, map_dalle=False)
        elif f[1] == 1:
            file = PIL.Image.open(f[0])
            x_vqgan = preprocess(file, target_image_size=size, map_dalle=False)
        else:
            x_vqgan = f[0]

        x_vqgan = x_vqgan.to(DEVICE)

        pils = []
        pils.append(custom_to_pil(preprocess_vqgan(x_vqgan[0])))
        for m in models:
            pils.append(custom_to_pil(reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), m)[0]))

        for j, image in enumerate(pils):
            img.paste(image, (j * size, i * size))
            if i == 0:
                ImageDraw.Draw(img).text((j * size, 0), f'{titles[j]}', (0, 0, 0),
                                         font=font)  # coordinates, text, color, font

    return img


# perlin noise generator - URL:https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy - 25/3/22
def perlin(x, y):
    # permutation table
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def perlin_noise(amplitude, frequency, size=256):
    lin = np.linspace(0, 5, size, endpoint=False)
    x, y = np.meshgrid(lin, lin)

    noise = amplitude * perlin(frequency * x, frequency * y)
    noise = (255*(noise - np.min(noise))/np.ptp(noise)).astype(np.uint8)
    return noise


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


# clean molecule image
def clean_molecule(img, cutoff=210, contrast=2.5):
    mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([cutoff, cutoff, cutoff]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    img = np.asarray(img_pil)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    mask_3ch = np.zeros_like(img)
    mask_3ch[:, :, 0] = mask
    mask_3ch[:, :, 1] = mask
    mask_3ch[:, :, 2] = mask
    white = np.ones_like(mask_3ch)
    img = img * (mask_3ch / 255) + white * (255 - mask_3ch)

    return img.astype("uint8")

