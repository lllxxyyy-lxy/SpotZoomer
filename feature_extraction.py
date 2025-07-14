import os
import gc
import pickle
import argparse
from pathlib import Path
from copy import deepcopy
from time import time
import timm
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import reduce
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

from utils.utils import load_image, smoothen

Image.MAX_IMAGE_PIXELS = None


def pad(image, block_size):
    """Pad image so that its height and width are divisible by block_size."""
    h, w = image.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    assert padded.shape[0] % block_size == 0 and padded.shape[1] % block_size == 0
    return padded


def normalize_data(data, scale=1.0):
    """Min-max normalize data to [0, scale]."""
    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)
    return (data - min_vals) / (max_vals - min_vals) * scale


def smoothen_embeddings(embs, size, kernel, method='cv', device='cuda'):
    """Smooth each embedding channel independently."""
    return [smoothen(c[..., np.newaxis], size=size, kernel=kernel, backend=method, device=device)[..., 0] for c in embs]


def smoothen_all_embeddings(embs_dict, size=4, kernel='uniform', method='cv', device='cuda'):
    """Apply smoothing to each group of embeddings (his, pos, rgb)."""
    return {key: smoothen_embeddings(val, size, kernel, method, device) for key, val in embs_dict.items()}


def create_coordinates_matrix(h, w):
    """Create a (2, H, W) matrix with y and x coordinates."""
    y_coords = np.repeat(np.arange(h)[:, None], w, axis=1)
    x_coords = np.repeat(np.arange(w)[None, :], h, axis=0)
    return np.stack([y_coords, x_coords], axis=0)


def extract_positional_embeddings(h, w, device='cuda'):
    """Extract and smooth positional embeddings."""
    coords = create_coordinates_matrix(h, w).astype('float32')
    coords[0] = normalize_data(coords[0])
    coords[1] = normalize_data(coords[1])
    pos_embs = smoothen_embeddings(coords, size=4, kernel='uniform', method='cv', device=device)
    return pos_embs


def combined_feature_extraction(prefix, result_prefix, token_file_path, cache_dir):
    """Main function to extract histology, RGB, and positional embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load input image
    img = load_image(f'{prefix}re_image.tif')
    h, w = img.shape[:2]

    # Set up hook to capture embeddings from the model
    img_emb = []

    def forward_hook(module, input, output):
        features = output[:, 1:, :].cpu().numpy()
        features = features.reshape(features.shape[0], 14, 14, features.shape[2])
        features = np.concatenate(features, axis=1)
        img_emb.append(features)

    # Authenticate Hugging Face and download model if needed
    os.environ['TORCH_HOME'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    with open(token_file_path, 'r') as file:
        huggingface_token = file.read().strip()
    login(token=huggingface_token, add_to_git_credential=True)

    local_model_dir = cache_dir / "MahmoodLab/UNI"
    model_path = local_model_dir / "pytorch_model.bin"

    if not model_path.exists():
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_model_dir, force_download=True)

    model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_model_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval().to(device)
    hook = model.norm.register_forward_hook(forward_hook)

    # Pad image and extract patches
    img_padded = pad(img, 224)
    patches = [
        img_padded[y:y + 224, x:x + 224]
        for y in range(0, img_padded.shape[0], 224)
        for x in range(0, img_padded.shape[1], 224)
    ]
    patches = np.array(patches)

    class ROIDataset(Dataset):
        def __init__(self, images):
            self.images = images
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.fromarray(self.images[idx].astype('uint8'))
            return self.transform(img)

    loader = DataLoader(ROIDataset(patches), batch_size=img_padded.shape[1] // 224, shuffle=False)

    with torch.inference_mode():
        for batch in loader:
            model(batch.to(device))
        img_emb = np.concatenate(img_emb, axis=0)
    hook.remove()

    # Extract histological features
    his_emb = [img_emb[:, :, i].astype('float32') for i in range(img_emb.shape[2])]

    # Positional embeddings
    pos_emb = extract_positional_embeddings(h, w, device=device)

    # RGB embeddings (downsampled mean pooling)
    rgb_emb = np.stack([
        reduce(img[..., i].astype(np.float16) / 255.0, '(h1 h) (w1 w) -> h1 w1', 'mean', h=16, w=16).astype(np.float32)
        for i in range(3)
    ])

    # Combine and smooth all embeddings
    all_embs = {'his': his_emb, 'pos': pos_emb, 'rgb': rgb_emb}
    smoothed_embs = smoothen_all_embeddings(all_embs, size=4, kernel='uniform', method='cv', device='cuda')
    gc.collect()

    # Concatenate along channels and save
    final_emb = np.concatenate([smoothed_embs['his'], smoothed_embs['pos'], smoothed_embs['rgb']])
    final_emb = final_emb.transpose(1, 2, 0)

    os.makedirs(os.path.dirname(result_prefix + 'embeddings-hist.pickle'), exist_ok=True)
    with open(result_prefix + 'embeddings-hist.pickle', 'wb') as f:
        pickle.dump(final_emb, f)

    print("Feature extraction complete. Saved to", result_prefix + 'embeddings-hist.pickle')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Feature Extraction")
    parser.add_argument('--prefix', type=str, default='/home/lixiaoyu/VISD/data/MouseBrain-HD2V/SourceDomain/',
                        help='Path prefix for the input image')
    parser.add_argument('--result_prefix', type=str, default='/home/lixiaoyu/VISD/result/MouseBrain-HD2V/SourceDomain/',
                        help='Path prefix for saving the result embeddings')
    parser.add_argument('--token_file_path', type=str, default='/home/lixiaoyu/VISD/pretrain-model/token',
                        help='Path to the Hugging Face token file')
    parser.add_argument('--cache_dir', type=str, default='/home/lixiaoyu/VISD/pretrain-model/cache/',
                        help='Path to the cache directory for models')

    args = parser.parse_args()
    combined_feature_extraction(args.prefix, args.result_prefix, args.token_file_path, Path(args.cache_dir))
