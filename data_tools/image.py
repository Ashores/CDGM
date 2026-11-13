"""Image helpers adapted from FCSDM datasets/Data_Loading.py and cv2_imageProcess.py."""

from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(root, recursive=True):
    """Return sorted image paths without loading data or inferring class labels."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def preprocess_image(image, image_size=32, crop_size=None):
    """RGB conversion, optional center crop, bilinear resize, and CHW [-1, 1].

    NumPy input must be a uint8 HW or HWC image. ``crop_size`` is either an
    integer or (height, width). Explicit crop sizes must fit the input image.
    """
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if not isinstance(image, Image.Image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise TypeError("NumPy images must be uint8; normalization is applied once here")
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        image = Image.fromarray(image)
    image = image.convert("RGB")
    if crop_size is not None:
        if isinstance(crop_size, int):
            crop_h = crop_w = crop_size
        else:
            crop_h, crop_w = crop_size
        width, height = image.size
        if not (isinstance(crop_h, int) and isinstance(crop_w, int)):
            raise ValueError("crop dimensions must be integers")
        if not (0 < crop_h <= height and 0 < crop_w <= width):
            raise ValueError("crop_size must fit within the image")
        top = int(round((height - crop_h) / 2.0))
        left = int(round((width - crop_w) / 2.0))
        image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def load_image(path, image_size=32, crop_size=None):
    with Image.open(path) as image:
        return preprocess_image(image, image_size, crop_size)


def inverse_transform(images):
    """Map normalized arrays from [-1, 1] to [0, 1], keeping their layout."""
    return (np.asarray(images, dtype=np.float32) + np.float32(1.0)) / np.float32(2.0)
