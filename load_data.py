import os
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

def load_image(file):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(file)
    return transform(image).unsqueeze(0)

def load_images(batch_size=128, size=128, path="~/code/git/ComfyUI/output/for_model/"):
    size = size//batch_size
    images = []
    images_size = batch_size * size

    with ThreadPoolExecutor(max_workers=12) as executor:
        for image in tqdm(executor.map(load_image, (path + file for file in os.listdir(path)[:images_size]))):
            images.append(image)

    ret = torch.cat(images,0)
    return ret.reshape([size, batch_size, *ret.shape[1:]])