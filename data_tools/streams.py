"""CDGM class-group streams; source attribution is in PROVENANCE.md.

Labels and task slices describe offline stream construction/evaluation. The
unsupervised learner receives only the image stream, without these metadata.
"""

from dataclasses import dataclass
import numpy as np


def _class_ids(labels):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        if labels.shape[1] == 0 or not np.all((labels == 0) | (labels == 1)) or not np.all(labels.sum(axis=1) == 1):
            raise ValueError("Expected one-hot rows with exactly one active class")
        return labels.argmax(axis=1)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Expected integer class IDs or a one-hot matrix")
    return labels


def class_incremental_indices(labels, classes_per_task=2, class_order=None,
                              shuffle_within_task=False, seed=0):
    """Group indices by consecutive classes, preserving every input sample.

    The default retains the original within-group sample order. Shuffling uses a
    local NumPy generator. No samples are duplicated or discarded.
    """
    ids = _class_ids(labels)
    if not isinstance(classes_per_task, int) or classes_per_task <= 0:
        raise ValueError("classes_per_task must be a positive integer")
    observed = set(ids.tolist())
    order = sorted(observed) if class_order is None else list(class_order)
    if len(set(order)) != len(order):
        raise ValueError("class_order contains duplicate classes")
    if not observed.issubset(set(order)):
        raise ValueError("class_order omits observed classes")
    rng = np.random.default_rng(seed)
    groups = []
    for start in range(0, len(order), classes_per_task):
        index = np.flatnonzero(np.isin(ids, order[start:start + classes_per_task]))
        if shuffle_within_task:
            rng.shuffle(index)
        groups.append(index)
    return groups


@dataclass
class ImageStream:
    images: np.ndarray
    labels: np.ndarray
    task_slices: tuple


def build_class_incremental_stream(images, labels, classes_per_task=2,
                                   class_order=None, shuffle_within_task=False,
                                   seed=0):
    """Return ordered images and separate construction metadata."""
    images, labels = np.asarray(images), np.asarray(labels)
    if images.ndim < 1 or labels.ndim < 1 or len(images) != len(labels):
        raise ValueError("Images and labels must have matching first dimensions")
    groups = class_incremental_indices(labels, classes_per_task, class_order,
                                       shuffle_within_task, seed)
    order = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    offsets = np.cumsum([0] + [len(group) for group in groups])
    boundaries = tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))
    return ImageStream(images[order], labels[order], boundaries)


def Split_dataset_by5(x, y):
    """Compatibility helper for five two-class, one-hot-label groups."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("Images and labels must have matching lengths")
    indices = class_incremental_indices(y, 2, list(range(10)))
    return tuple(value for index in indices for value in (x[index], y[index]))


def Split_Dataset_Five(train_data, train_labels, test_data, test_labels):
    """Legacy return order, keeping train/test partitions separate."""
    train = Split_dataset_by5(train_data, train_labels)
    test = Split_dataset_by5(test_data, test_labels)
    return list(train[::2]), list(train[1::2]), list(test[::2]), list(test[1::2])
