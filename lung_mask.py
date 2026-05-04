"""
Lung Mask — Generic Implementation from Scratch
=================================================
Extracts a binary lung mask from a 3-D CT volume (HU values).

Pipeline:
  1. Threshold      — isolate air voxels (HU < -400)
  2. Closing        — fill small holes from vessels / noise
  3. CC labelling   — find all connected air regions
  4. Remove border  — strip external air (scanner background)
  5. Keep top-2     — left lung + right lung
  6. Fill holes     — fill dense structures inside each lung
  7. Dilate         — expand slightly to include subpleural nodules

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label as cc_label,
)


# ──────────────────────────────────────────────────────────────
#  Individual steps
# ──────────────────────────────────────────────────────────────

def threshold_air(volume: np.ndarray,
                  hu_thresh: float = -400.0) -> np.ndarray:
    """
    Step 1 — Binary threshold.
    Returns True where voxel is air-density (HU < hu_thresh).
    """
    return volume < hu_thresh


def morphological_close(binary: np.ndarray,
                         iterations: int = 2) -> np.ndarray:
    """
    Step 2 — Close small gaps caused by blood vessels or noise.
    Uses 6-connectivity (face-connected) structuring element.
    """
    struct = generate_binary_structure(3, 1)   # 6-conn
    return binary_closing(binary, structure=struct, iterations=iterations)


def remove_border_components(binary: np.ndarray) -> np.ndarray:
    """
    Steps 3 & 4 — Label components, then remove any that touch the volume border.
    Components touching the border are external air (scanner table, room air).
    Returns a bool mask of only internal air regions.
    """
    struct = generate_binary_structure(3, 3)   # 26-conn
    labeled, _ = cc_label(binary, structure=struct)

    # Collect every label that appears on any face of the volume
    border_labels = set()
    for face in [
        labeled[0],  labeled[-1],
        labeled[:, 0],  labeled[:, -1],
        labeled[:, :, 0], labeled[:, :, -1],
    ]:
        border_labels.update(np.unique(face))

    border_labels.discard(0)   # 0 = background (non-air)

    internal = labeled.copy()
    for lbl in border_labels:
        internal[internal == lbl] = 0

    return internal > 0


def keep_two_largest(binary: np.ndarray) -> np.ndarray:
    """
    Step 5 — Keep only the two largest connected components.
    In a standard CT these are the left and right lung lobes.
    """
    struct = generate_binary_structure(3, 3)
    labeled, n = cc_label(binary, structure=struct)

    if n == 0:
        return binary

    # Component sizes (labels are 1-based)
    sizes = np.array([
        (labeled == lbl).sum() for lbl in range(1, n + 1)
    ])
    top_labels = np.argsort(sizes)[-(min(2, n)):] + 1   # two largest labels

    mask = np.zeros_like(binary)
    for lbl in top_labels:
        mask |= (labeled == lbl)
    return mask


def fill_interior(binary: np.ndarray) -> np.ndarray:
    """
    Step 6 — Fill holes slice-by-slice (axial axis).
    Closes off vessels, airways, and nodules that appear dense inside the lung.
    """
    filled = np.zeros_like(binary)
    for z in range(binary.shape[0]):
        filled[z] = binary_fill_holes(binary[z])
    return filled


def dilate_mask(binary: np.ndarray, voxels: int = 2) -> np.ndarray:
    """
    Step 7 — Dilate outward to capture subpleural / pleural regions.
    """
    struct = generate_binary_structure(3, 1)
    return binary_dilation(binary, structure=struct, iterations=voxels)


# ──────────────────────────────────────────────────────────────
#  Main function
# ──────────────────────────────────────────────────────────────

def generate_lung_mask(
    volume: np.ndarray,
    hu_thresh: float  = -400.0,
    close_iters: int  = 2,
    dilate_voxels: int = 2,
) -> np.ndarray:
    """
    Generate a binary lung mask from a 3-D CT volume.

    Args:
        volume        : float array of shape (D, H, W) in Hounsfield Units.
        hu_thresh     : HU cutoff for air detection (default -400).
        close_iters   : morphological closing iterations (default 2).
        dilate_voxels : final outward dilation in voxels (default 2).

    Returns:
        mask : bool array, shape (D, H, W).
               True = inside the lung cavity.
    """
    mask = threshold_air(volume, hu_thresh)
    mask = morphological_close(mask, iterations=close_iters)
    mask = remove_border_components(mask)
    mask = keep_two_largest(mask)
    mask = fill_interior(mask)
    if dilate_voxels > 0:
        mask = dilate_mask(mask, voxels=dilate_voxels)
    return mask


# ──────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    # Build a toy volume: two air cylinders (lungs) inside soft tissue
    rng = np.random.default_rng(0)
    D, H, W = 64, 128, 128
    vol = np.full((D, H, W), 50.0, dtype=np.float32)   # soft tissue

    # External air border
    vol[:, :10, :] = vol[:, -10:, :] = -1000
    vol[:, :, :10] = vol[:, :, -10:] = -1000

    # Two lung lobes
    for cy, cx in [(64, 40), (64, 88)]:
        yy, xx = np.ogrid[:H, :W]
        lobe = (yy - cy)**2 + (xx - cx)**2 < 28**2
        vol[:, lobe] = rng.normal(-750, 80, (D, lobe.sum())).astype(np.float32)

    mask = generate_lung_mask(vol)

    print(f"Volume shape : {vol.shape}")
    print(f"Mask shape   : {mask.shape}  dtype: {mask.dtype}")
    print(f"Lung voxels  : {mask.sum():,} ({100 * mask.mean():.1f}%)")
