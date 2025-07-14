import numpy as np
from torch.utils.data import Dataset
from utils.image_funtions import get_disk_mask
from utils.visual import plot_spot_masked_image

class SpotDataset(Dataset):
    """
    PyTorch Dataset for spatial transcriptomics data.
    Extracts spatial patches from source and target domains using disk masks.
    """

    def __init__(self, x_all_source, y_source, label_source, locs_source, radius_source,
                 x_all_target, y_target, locs_target, radius_target):
        super().__init__()

        self.mask_source = get_disk_mask(radius_source)
        self.mask_target = get_disk_mask(radius_target)

        # Extract patches from source and filter out non-finite values
        self.x_source = get_patches_flat(x_all_source, locs_source, self.mask_source)
        valid_source = np.isfinite(self.x_source).all(axis=(-1, -2))
        self.x_source = self.x_source[valid_source]
        self.y_source = y_source[valid_source]
        self.label_source = get_patches_flat(label_source, locs_source, self.mask_source)
        self.locs_source = locs_source[valid_source]

        # Extract patches from target and filter out non-finite values
        self.x_target = get_patches_flat(x_all_target, locs_target, self.mask_target)
        valid_target = np.isfinite(self.x_target).all(axis=(-1, -2))
        self.x_target = self.x_target[valid_target]
        self.y_target = y_target[valid_target]
        self.locs_target = locs_target[valid_target]

        # Store image dimensions and radius
        self.size_source = x_all_source.shape[:2]
        self.size_target = x_all_target.shape[:2]
        self.radius_source = radius_source
        self.radius_target = radius_target

    def __len__(self):
        return min(len(self.x_source), len(self.x_target))

    def __getitem__(self, idx):
        return (
            self.x_source[idx], self.y_source[idx], self.label_source[idx], self.locs_source[idx]
        ), (
            self.x_target[idx], self.y_target[idx], self.locs_target[idx]
        )

    def show(self, channel_x, channel_y, prefix):
        """
        Save visualizations of selected channels from source and target patches.
        """
        plot_spot_masked_image(
            locs=self.locs_source,
            values=self.x_source[:, :, channel_x].mean(axis=0),
            mask=self.mask_source,
            size=self.size_source,
            outfile=f'{prefix}_source_x{channel_x:04d}.png'
        )

        plot_spot_masked_image(
            locs=self.locs_source,
            values=self.y_source[:, channel_y],
            mask=self.mask_source,
            size=self.size_source,
            outfile=f'{prefix}_source_y{channel_y:04d}.png'
        )

        plot_spot_masked_image(
            locs=self.locs_target,
            values=self.x_target[:, :, channel_x].mean(axis=0),
            mask=self.mask_target,
            size=self.size_target,
            outfile=f'{prefix}_target_x{channel_x:04d}.png'
        )

        plot_spot_masked_image(
            locs=self.locs_target,
            values=self.y_target[:, channel_y],
            mask=self.mask_target,
            size=self.size_target,
            outfile=f'{prefix}_target_y{channel_y:04d}.png'
        )


def get_patches_flat(img, locs, mask):
    """
    Extracts flat patches from an image using a binary mask centered at each location.

    Args:
        img: HxWxC image array.
        locs: list of (y, x) coordinates.
        mask: binary 2D array defining the shape of the patch.

    Returns:
        A (N, H*W, C) or (N, H, W, C) array of patches, flattened according to the mask.
    """
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape - center], axis=-1)
    x_list = []

    for s in locs:
        y, x = s
        if isinstance(y, np.ndarray): y = y.item()
        if isinstance(x, np.ndarray): x = x.item()

        top_left_y = int(y + r[0][0])
        bottom_right_y = int(y + r[0][1])
        top_left_x = int(x + r[1][0])
        bottom_right_x = int(x + r[1][1])

        patch = img[
            max(0, top_left_y):min(img.shape[0], bottom_right_y),
            max(0, top_left_x):min(img.shape[1], bottom_right_x)
        ]

        if patch.shape[:2] != mask.shape:
            patch = np.pad(
                patch,
                (
                    (max(0, -top_left_y), max(0, bottom_right_y - img.shape[0])),
                    (max(0, -top_left_x), max(0, bottom_right_x - img.shape[1])),
                    (0, 0)
                ),
                mode='constant'
            )

        if mask.all():
            x_patch = patch
        else:
            if patch.shape[:2] == mask.shape:
                x_patch = patch[mask]
            else:
                continue

        x_list.append(x_patch)

    return np.stack(x_list) if x_list else np.array([])
