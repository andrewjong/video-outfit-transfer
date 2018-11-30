import os
import unittest
import torch

from src.datasets import WarpDataset
from torch.utils.data import DataLoader

from torch import Tensor


CLOTHING_SEG_PATH = "test/test_resources/clothing_segmentation/"
BODY_SEG_PATH = "test/test_resources/body_segmentation/"
BATCH_SIZE = 1


def get_dataset():
    w_dataset = WarpDataset(CLOTHING_SEG_PATH, BODY_SEG_PATH)
    return w_dataset


def get_dataloader(batch_size):
    w_dataset = get_dataset()
    dl = DataLoader(w_dataset, batch_size=batch_size)
    return dl


def get_first_item(batch_size):
    dl = get_dataloader(batch_size)
    x, y, z = next(iter(dl))
    return x, y, z


class TestWarpDataset(unittest.TestCase):
    def test_get_item_return_type(self):
        """
        Ensures the return type is 3 tensors
        :return:
        """
        x, y, z = get_first_item()
        self.assertTrue(type(x) is Tensor)
        self.assertTrue(type(y) is Tensor)
        self.assertTrue(type(z) is Tensor)

    def test_tensor_shapes(self):
        # ensure multiple batch sizes
        for batch_size in (1, 4):
            x, y, z = get_first_item(batch_size)
            self.assertEqual(torch.Size((batch_size, 19, 960, 540)), x.shape)
            self.assertEqual(torch.Size((batch_size, 19, 960, 540)), y.shape)
            self.assertEqual(torch.Size((batch_size, 3, 960, 540)), z.shape)

    def test_dataset_length(self):
        expected = len(os.listdir(BODY_SEG_PATH))
        self.assertEqual(expected, len(get_dataset()))


if __name__ == "__main__":
    unittest.main()
