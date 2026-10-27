"""Tests for the lung mask and the DenseNet-121.  Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lung_mask import generate_lung_mask, keep_two_largest, remove_border_components  # noqa: E402


def toy_volume(seed=0, depth=24, size=96):
    """Soft tissue with an air border and two lung lobes."""
    rng = np.random.default_rng(seed)
    vol = np.full((depth, size, size), 50.0, dtype=np.float32)
    vol[:, :6, :] = vol[:, -6:, :] = -1000
    vol[:, :, :6] = vol[:, :, -6:] = -1000
    yy, xx = np.ogrid[:size, :size]
    for cx in (30, 66):  # tissue margin to the air border and a gap between the lobes
        lobe = (yy - size // 2) ** 2 + (xx - cx) ** 2 < 14 ** 2
        vol[:, lobe] = rng.normal(-750, 60, (depth, int(lobe.sum()))).astype(np.float32)
    return vol


class LungMask(unittest.TestCase):
    def test_finds_two_lungs_and_ignores_outside_air(self):
        vol = toy_volume()
        mask = generate_lung_mask(vol)
        self.assertEqual(mask.shape, vol.shape)
        self.assertEqual(mask.dtype, bool)
        _, n_components = cc_label(mask, structure=generate_binary_structure(3, 3))
        self.assertEqual(n_components, 2)
        self.assertFalse(mask[:, :3, :].any())        # outside air at the border is not lung
        self.assertTrue(mask[12, 48, 30])             # centre of the left lobe is lung
        self.assertTrue(mask[12, 48, 66])             # centre of the right lobe is lung

    def test_vectorised_helpers_match_the_label_loop_version(self):
        rng = np.random.default_rng(1)
        binary = rng.random((16, 40, 40)) > 0.55

        struct = generate_binary_structure(3, 3)
        labeled, n = cc_label(binary, structure=struct)
        border = set()
        for face in (labeled[:, 0], labeled[:, -1], labeled[:, :, 0], labeled[:, :, -1]):  # in-plane edges only
            border.update(np.unique(face))
        border.discard(0)
        expected_internal = np.isin(labeled, [l for l in range(1, n + 1) if l not in border])
        np.testing.assert_array_equal(remove_border_components(binary), expected_internal)

        sizes = [(labeled == l).sum() for l in range(1, n + 1)]
        top = np.argsort(sizes)[-2:] + 1
        np.testing.assert_array_equal(keep_two_largest(binary), np.isin(labeled, top))

    def test_lungs_touching_first_and_last_slice_are_kept(self):
        # the toy lobes run through every slice, like a scan cropped exactly at the lung apex and base
        vol = toy_volume()
        self.assertTrue(generate_lung_mask(vol)[0, 48, 30] and generate_lung_mask(vol)[-1, 48, 66])

    def test_empty_volume(self):
        mask = generate_lung_mask(np.full((8, 32, 32), 50.0, dtype=np.float32))
        self.assertFalse(mask.any())


class DenseNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")

    def test_3d_parameter_count_matches_monai(self):
        from src.densenet121 import DenseNet121
        ours = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).count_parameters()
        self.assertEqual(ours, 11_244_674)  # the figure printed by the Stage 1 / Stage 2 notebooks (MONAI model)
        try:
            from monai.networks.nets import DenseNet121 as Monai
        except ImportError:
            return
        self.assertEqual(ours, sum(p.numel() for p in Monai(spatial_dims=3, in_channels=1, out_channels=2).parameters()))

    def test_forward_shapes(self):
        import torch
        from src.densenet121 import DenseNet121
        model3d = DenseNet121(spatial_dims=3).eval()
        model2d = DenseNet121(spatial_dims=2, in_channels=3, out_channels=5).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model3d(torch.randn(2, 1, 64, 64, 64)).shape), (2, 2))
            self.assertEqual(tuple(model2d(torch.randn(2, 3, 64, 64)).shape), (2, 5))

    def test_rejects_bad_dimensions(self):
        from src.densenet121 import DenseNet121
        with self.assertRaises(ValueError):
            DenseNet121(spatial_dims=4)


if __name__ == "__main__":
    unittest.main()
