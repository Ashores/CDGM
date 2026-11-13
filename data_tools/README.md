# Data preparation

This release contains portable data helpers adapted from [FCSDM](https://github.com/Ashores/FCSDM). They run on CPU with NumPy, Pillow, and SciPy. CDGM networks, training, evaluation, checkpoints, and experiment-specific stream configurations are **Coming soon**.

## Install

```bash
python -m pip install -r requirements.txt
```

## Available helpers

| Module | Functionality |
| --- | --- |
| `loaders.py` | Local MNIST/Fashion-MNIST IDX, CIFAR-10 Python batches, and SVHN MAT readers |
| `image.py` | Image-file discovery, RGB conversion, center crop, resize, and normalization |
| `streams.py` | Class-group stream ordering, optional deterministic within-group shuffle, and FCSDM five-task compatibility helpers |

Readers return **uint8 NHWC** images and integer labels. `preprocess_image` returns **float32 CHW** images in **[-1, 1]**, replicating grayscale images into three RGB channels. It uses explicitly specified bilinear interpolation. Dataset contents are not included or downloaded automatically.

## Local data layout

For MNIST or Fashion-MNIST, pass the directory containing `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`. Both uncompressed and `.gz` files are supported.

For CIFAR-10, pass the extracted `cifar-10-batches-py` directory. For SVHN, pass the path to a `.mat` file containing `X` and `y`. Obtain the datasets from their respective maintainers.

## Example

Run from the repository root:

```python
from data_tools import load_mnist, build_class_incremental_stream, preprocess_image

images, labels = load_mnist("data/mnist", split="train")
stream = build_class_incremental_stream(
    images, labels,
    classes_per_task=2,
    class_order=list(range(10)),
)
image = preprocess_image(stream.images[0], image_size=32)
print(image.shape, image.dtype)  # (3, 32, 32), float32
```

Labels are used only to arrange an evaluation stream; an unsupervised learner should consume `stream.images` without `stream.labels` or `stream.task_slices`. Train and test partitions are loaded separately. No few-shot sample count or benchmark split is inferred by these helpers.

SVHN defaults to FCSDM's original `raw_label - 1` convention. Use `load_svhn(path, label_mode="digit")` only when digit identities 0–9 are intended. The two mappings lead to different class orders.

## Provenance and cleanup

- `FCSDM/datasets/Data_Loading.py`: IDX/SVHN reading, center crop, and normalization conventions.
- `FCSDM/datasets/MyCIFAR10.py`: CIFAR batch reading and channel ordering.
- `FCSDM/datasets/cv2_imageProcess.py`: image preprocessing helpers.
- `FCSDM/Task_Split/Task_utilizes.py`: class-group ordering and five-task interfaces.

This subset removes hard-coded machine paths, GPU calls, training/evaluation imports, duplicate definitions, and implicit train/test mixing. It adds input checks and explicit preprocessing choices. It is a portable data utility release; the experiment configurations associated with the manuscript will accompany the method release.

## Checks

```bash
python -m unittest discover -s tests -v
```

The checks use small synthetic files to verify decoding, layouts, ranges, partition separation, and stream ordering. They do not run model training or measure manuscript results.
