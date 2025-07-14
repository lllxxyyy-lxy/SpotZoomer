import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class STDataset(Dataset):
    def __init__(self, data_dir, mode='source'):
        self.data_dir = data_dir
        self.mode = mode

        # Load gene expression data (assumed to be in a CSV file)
        self.gene_expression = pd.read_csv(os.path.join(data_dir, 'pseudo_st_t.csv')).values

        # Load coordinates (assumed to be in a CSV file)
        self.coords = pd.read_csv(os.path.join(data_dir, 'pseudo_locs.csv')).values

        # Load histological image (assumed to be a TIFF file)
        self.img_path = os.path.join(data_dir, 're_image.tif')
        self.img = Image.open(self.img_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load labels if in source mode
        if self.mode == 'source':
            self.labels = pd.read_csv(os.path.join(data_dir, 'labels.csv')).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.gene_expression)

    def __getitem__(self, idx):
        X = torch.tensor(self.gene_expression[idx], dtype=torch.float32)
        A = torch.tensor(self.coords[idx], dtype=torch.float32)
        img = self.transform(self.img)
        if self.labels is not None:
            L = torch.tensor(self.labels[idx], dtype=torch.float32)
            return X, A, img, L
        else:
            return X, A, img