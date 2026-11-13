# Data Utility Provenance

The CDGM data preparation utilities are adapted from our [FCSDM codebase](https://github.com/Ashores/FCSDM). This file records the source functions and portability changes.

| CDGM module | Source in FCSDM | Reused functionality |
| --- | --- | --- |
| `loaders.py` | `datasets/Data_Loading.py`, `datasets/MyCIFAR10.py` | IDX and SVHN readers, CIFAR batch reading and channel ordering |
| `image.py` | `datasets/Data_Loading.py`, `datasets/cv2_imageProcess.py` | Center cropping, resizing, and normalization |
| `streams.py` | `Task_Split/Task_utilizes.py` | Class-group ordering and five-task interfaces |

The portable subset removes machine-specific paths, GPU calls, training/evaluation imports, duplicate definitions, and implicit train/test mixing. It adds input validation and explicit preprocessing options. The default SVHN label mapping preserves the source implementation's `raw_label - 1` convention; the optional `digit` mode maps raw label 10 to digit 0.

The experiment configurations associated with the manuscript will accompany the CDGM method release.
