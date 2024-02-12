import numpy as np
import torch
from PIL import Image
from matplotlib.pyplot import get_cmap
from torchvision.transforms.functional import to_pil_image


# region concat images
def concat_images_square(images, scale_factor, padding):
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))

    w, h = images[0].size
    if scale_factor != 1:
        images = [i.resize((w * scale_factor, h * scale_factor)) for i in images]
        w, h = images[0].size

    concated = Image.new(images[0].mode, (w * columns + padding * (columns - 1), h * rows + padding * (rows - 1)))
    for i in range(len(images)):
        col = (i % columns)
        row = i // columns
        concated.paste(images[i], (w * col + padding * (col - 1), h * row + padding * (row - 1)))
    return concated


def concat_images_vertical(images, scale_factor=1):
    scale_images(images, scale_factor)
    w, h = images[0].size

    concated = Image.new(images[0].mode, (w, h * len(images)))
    for i in range(len(images)):
        concated.paste(images[i], (0, h * i))
    return concated


def concat_images_horizontal(images, scale_factor=1):
    scale_images(images, scale_factor)
    w, h = images[0].size

    concated = Image.new(images[0].mode, (w * len(images), h))
    for i in range(len(images)):
        concated.paste(images[i], (w * i, 0))
    return concated


# endregion


def scale_images(images, scale_factor):
    if scale_factor != 1:
        return [i.resize((i.width * scale_factor, i.height * scale_factor)) for i in images]
    return images


def save_image_tensor(images, out_uri, denormalize=None, scale_range_per_image=False, scale_factor=1., padding=2):
    assert torch.is_tensor(images) and images.ndim == 4
    if denormalize is not None:
        images = [denormalize(img) for img in images]
    if scale_range_per_image:
        images = images - images.min(dim=0)
        images = images / images.max(dim=0)

    pil_images = []
    for i, img in enumerate(images):
        if img.size(0) != 3:
            # use only first channel
            img = img[0]
            # img[0, 0] = 0.
            # img[-1, -1] = 1.
            # apply viridis colormap
            cm = get_cmap("viridis")
            img = img.cpu().numpy()
            img = cm(img)
            img = np.uint8(img * 255)
            pil_image = Image.fromarray(img)
            pil_images.append(pil_image)
        else:
            pil_images.append(to_pil_image(img))
    save_images(pil_images, out_uri, scale_factor, padding)


def save_images(images, out_uri, scale_factor=1., padding=2):
    assert isinstance(images, list)
    concated = concat_images_square(images, scale_factor, padding)
    concated.save(out_uri)


def images_to_gif(image_uris, out_uri, duration=200):
    if len(image_uris) == 0:
        return
    imgs = (Image.open(f) for f in image_uris)
    img = next(imgs)
    img.save(fp=out_uri, format="GIF", append_images=imgs, save_all=True, duration=duration, loop=0)
