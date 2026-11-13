"""CDGM local-file data loaders; source attribution is in PROVENANCE.md.

Each loader returns uint8 images in NHWC layout and int64 class labels.
Train/test partitions stay separate. Apply image.preprocess_image explicitly
when RGB resizing and normalization are needed.
"""

import gzip
import pickle
import struct
from pathlib import Path
import numpy as np


def _read_idx(path, dimensions):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError(f"Truncated IDX header: {path.name}")
        magic = struct.unpack(">I", header)[0]
        if magic != 0x800 + dimensions:
            raise ValueError(f"Unexpected IDX magic number: {path.name}")
        shape_bytes = stream.read(4 * dimensions)
        if len(shape_bytes) != 4 * dimensions:
            raise ValueError(f"Truncated IDX shape: {path.name}")
        shape = struct.unpack(">" + "I" * dimensions, shape_bytes)
        if any(n == 0 for n in shape):
            raise ValueError("IDX dimensions must be positive")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != int(np.prod(shape, dtype=object)):
        raise ValueError(f"IDX size does not match its header: {path.name}")
    return values.reshape(shape).copy()


def _idx_path(root, filename):
    path = Path(root) / filename
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"Expected {filename} or {filename}.gz in {root}")


def load_mnist(root, split="train"):
    """Read MNIST or Fashion-MNIST IDX files; images have one channel."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    prefix = "train" if split == "train" else "t10k"
    images = _read_idx(_idx_path(root, prefix + "-images-idx3-ubyte"), 3)
    labels = _read_idx(_idx_path(root, prefix + "-labels-idx1-ubyte"), 1)
    if len(images) != len(labels) or np.any(labels > 9):
        raise ValueError("Invalid MNIST image/label counts or class labels")
    return images[..., None], labels.astype(np.int64)


def load_cifar10(root, split="train"):
    """Read an extracted official CIFAR-10 Python archive.

    Only use trusted CIFAR pickle files. The root contains data_batch_1..5 and
    test_batch. Channel-major storage is converted to NHWC without shuffling.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    names = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
    images, labels = [], []
    for name in names:
        with (Path(root) / name).open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        x = np.asarray(batch[b"data"])
        y = np.asarray(batch[b"labels"])
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape[1] != 3072:
            raise ValueError(f"Invalid CIFAR image array in {name}")
        if y.ndim != 1 or y.dtype.kind not in "iu" or len(x) != len(y) or np.any((y < 0) | (y > 9)):
            raise ValueError(f"Invalid CIFAR labels in {name}")
        images.append(x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.append(y.astype(np.int64))
    return np.concatenate(images), np.concatenate(labels)


def load_svhn(path, label_mode="legacy"):
    """Read a local SVHN .mat file with fields X and y.

    ``legacy`` preserves the source y-1 mapping (raw label 10 becomes 9).
    ``digit`` maps raw label 10 to digit 0 and leaves labels 1..9 unchanged.
    Choose deliberately: the mapping affects the class order of the stream.
    """
    if label_mode not in {"legacy", "digit"}:
        raise ValueError("label_mode must be 'legacy' or 'digit'")
    from scipy.io import loadmat
    data = loadmat(path)
    images = np.asarray(data["X"])
    labels = np.asarray(data["y"]).reshape(-1)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[2] != 3:
        raise ValueError("Expected uint8 SVHN images with shape H x W x 3 x N")
    if labels.dtype.kind not in "iu" or len(labels) != images.shape[3] or np.any((labels < 1) | (labels > 10)):
        raise ValueError("Invalid SVHN image/label counts or labels")
    labels = labels.astype(np.int64)
    labels = labels - 1 if label_mode == "legacy" else labels % 10
    return np.ascontiguousarray(images.transpose(3, 0, 1, 2)), labels
