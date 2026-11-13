import gzip
import pickle
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from data_tools import (
    preprocess_image, inverse_transform, list_image_files, load_image,
    load_mnist, load_cifar10, load_svhn, class_incremental_indices,
    build_class_incremental_stream,
)
from data_tools.streams import Split_Dataset_Five


class DataToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_idx(self, name, array, compressed=False):
        path = self.root / (name + (".gz" if compressed else ""))
        payload = struct.pack(">I", 0x800 + array.ndim)
        payload += struct.pack(">" + "I" * array.ndim, *array.shape)
        payload += array.astype(np.uint8).tobytes()
        path.write_bytes(gzip.compress(payload) if compressed else payload)

    def test_grayscale_layout_range_and_inverse(self):
        x = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = preprocess_image(x, 2)
        self.assertEqual(result.shape, (3, 2, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(inverse_transform(result)[0], x / 255)

    def test_center_crop_and_invalid_input(self):
        x = np.zeros((4, 6, 3), dtype=np.uint8)
        x[1:3, 2:4] = 255
        np.testing.assert_array_equal(preprocess_image(x, 2, 2), np.ones((3, 2, 2)))
        with self.assertRaises(ValueError):
            preprocess_image(x, 2, 10)
        with self.assertRaises(TypeError):
            preprocess_image(x.astype(float))

    def test_file_discovery_and_rgb(self):
        (self.root / "nested").mkdir()
        Image.new("RGB", (4, 4), (255, 0, 0)).save(self.root / "nested/a.PNG")
        (self.root / "ignored.txt").write_text("not an image")
        files = list_image_files(self.root)
        self.assertEqual(len(files), 1)
        self.assertEqual(list_image_files(self.root, recursive=False), [])
        x = load_image(files[0], 4)
        np.testing.assert_array_equal(x[:, 0, 0], [1, -1, -1])

    def test_idx_train_test_separation(self):
        for prefix, pixel, label, compressed in [("train", 10, 2, True), ("t10k", 200, 9, False)]:
            self.write_idx(prefix + "-images-idx3-ubyte", np.full((2, 3, 3), pixel, np.uint8), compressed)
            self.write_idx(prefix + "-labels-idx1-ubyte", np.full(2, label, np.uint8), compressed)
        train, train_y = load_mnist(self.root, "train")
        test, test_y = load_mnist(self.root, "test")
        self.assertEqual(train.shape, (2, 3, 3, 1))
        self.assertTrue(np.all(train == 10) and np.all(test == 200))
        self.assertTrue(np.all(train_y == 2) and np.all(test_y == 9))

    def test_idx_corruption_is_rejected(self):
        self.write_idx("train-images-idx3-ubyte", np.zeros((2, 3, 3), np.uint8))
        self.write_idx("train-labels-idx1-ubyte", np.zeros(1, np.uint8))
        with self.assertRaises(ValueError):
            load_mnist(self.root)
        p = self.root / "train-images-idx3-ubyte"
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            load_mnist(self.root)

    def test_cifar_channel_order_and_partition(self):
        x = np.zeros((1, 3072), dtype=np.uint8)
        x[:, :1024] = 255
        for i in range(1, 6):
            with (self.root / f"data_batch_{i}").open("wb") as f:
                pickle.dump({b"data": x, b"labels": [i]}, f)
        with (self.root / "test_batch").open("wb") as f:
            pickle.dump({b"data": 255 - x, b"labels": [0]}, f)
        train, labels = load_cifar10(self.root)
        test, test_y = load_cifar10(self.root, "test")
        self.assertEqual(train.shape, (5, 32, 32, 3))
        np.testing.assert_array_equal(train[0, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(test[0, 0, 0], [0, 255, 255])
        np.testing.assert_array_equal(labels, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(test_y, [0])

    def test_svhn_explicit_label_mapping(self):
        path = self.root / "svhn.mat"
        x = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        x[:, :, 0, 0] = 255
        savemat(path, {"X": x, "y": np.array([[1], [9], [10]], dtype=np.uint8)})
        images, legacy = load_svhn(path)
        _, digit = load_svhn(path, "digit")
        self.assertEqual(images.shape, (3, 2, 2, 3))
        np.testing.assert_array_equal(images[0, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(legacy, [0, 8, 9])
        np.testing.assert_array_equal(digit, [1, 9, 0])

    def test_five_task_legacy_order_and_alignment(self):
        labels = np.array([9, 0, 6, 2, 5, 3, 1, 8, 7, 4])
        x = np.arange(10)
        stream = build_class_incremental_stream(x, labels, class_order=range(10))
        np.testing.assert_array_equal(stream.images, [1, 6, 3, 5, 4, 9, 2, 8, 0, 7])
        np.testing.assert_array_equal(stream.labels, labels[stream.images])
        self.assertEqual(len(stream.task_slices), 5)
        for task, group in enumerate(stream.task_slices):
            self.assertTrue(set(stream.labels[group]).issubset({2 * task, 2 * task + 1}))

    def test_one_hot_compatibility_and_no_partition_mixing(self):
        y = np.eye(10, dtype=np.int64)
        tx, ty, vx, vy = Split_Dataset_Five(np.arange(10), y, np.arange(10) + 100, y)
        self.assertEqual(len(tx), 5)
        np.testing.assert_array_equal(np.concatenate(tx), np.arange(10))
        np.testing.assert_array_equal(np.concatenate(vx), np.arange(10) + 100)
        np.testing.assert_array_equal(np.concatenate(ty), np.concatenate(vy))

    def test_shuffle_is_deterministic_and_covers_all_samples(self):
        y = np.repeat(np.arange(10), 8)
        a = np.concatenate(class_incremental_indices(y, shuffle_within_task=True, seed=12))
        b = np.concatenate(class_incremental_indices(y, shuffle_within_task=True, seed=12))
        c = np.concatenate(class_incremental_indices(y, shuffle_within_task=True, seed=13))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(len(y)))
        self.assertFalse(np.array_equal(a, c))

    def test_invalid_class_order_and_labels(self):
        with self.assertRaises(ValueError):
            class_incremental_indices([0, 1, 2], class_order=[0, 1])
        with self.assertRaises(ValueError):
            class_incremental_indices([0, 1], class_order=[0, 1, 1])
        with self.assertRaises(ValueError):
            class_incremental_indices([[0, 0], [1, 0]])
        with self.assertRaises(ValueError):
            build_class_incremental_stream(np.zeros(2), np.arange(3))

    def test_empty_stream_and_empty_class_groups(self):
        stream = build_class_incremental_stream(np.empty((0, 3, 32, 32)), np.empty(0, dtype=int))
        self.assertEqual(stream.images.shape, (0, 3, 32, 32))
        self.assertEqual(stream.task_slices, ())
        groups = class_incremental_indices([0, 9], class_order=range(10))
        self.assertEqual([len(g) for g in groups], [1, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
