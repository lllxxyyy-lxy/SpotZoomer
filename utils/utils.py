import itertools
from PIL import Image
import pickle
import os

import numpy as np
import pandas as pd
import yaml
import numpy as np
import cv2 as cv
import torch
from torch import nn
import skimage
from scipy.ndimage import uniform_filter

Image.MAX_IMAGE_PIXELS = None

def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='gaussian', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'gaussian':
        sigma = size / 4  # approximate std of uniform filter 1/sqrt(12)
        truncate = 4.0
        winsize = np.ceil(sigma * truncate).astype(int) * 2 + 1
        if backend == 'cv':
            print(f'gaussian filter: winsize={winsize}, sigma={sigma}')
            y = cv.GaussianBlur(
                    x, (winsize, winsize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv.BORDER_REFLECT)
        elif backend == 'skimage':
            y = skimage.filters.gaussian(
                    x, sigma=sigma, truncate=truncate,
                    preserve_range=True, channel_axis=-1)
        else:
            raise ValueError('backend must be cv or skimage')
    elif kernel == 'uniform':
        if backend == 'cv':  # 运行这里
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        elif backend == 'torch':
            assert isinstance(size, int)
            padding = size // 2
            size = size + 1

            pool_dict = {
                    'mean': nn.AvgPool2d(
                        kernel_size=size, stride=1, padding=0),
                    'max': nn.MaxPool2d(
                        kernel_size=size, stride=1, padding=0)}
            pool = pool_dict[mode]

            mod = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    pool)
            y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
            y = y.permute(1, 2, 0)
            y = y.cpu().detach().numpy()
        else:
            raise ValueError('backend must be cv or torch')
    else:
        raise ValueError('kernel must be gaussian or uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]

    return y


def upscale(x, target_shape):
    mask = np.isfinite(x).all(tuple(range(2, x.ndim)))
    x = impute_missing(x, ~mask, radius=3)
    # TODO: Consider using pytorch with cuda to speed up
    # order: 0 == nearest neighbor, 1 == bilinear, 3 == bicubic
    dtype = x.dtype
    x = skimage.transform.resize(
            x, target_shape, order=3, preserve_range=True)
    x = x.astype(dtype)
    if not mask.all():
        mask = skimage.transform.resize(
                mask.astype(float), target_shape, order=3,
                preserve_range=True)
        mask = mask > 0.5
        x[~mask] = np.nan
    return x


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img


def get_disk_mask(radius, boundary_width=None):
    #radius_ceil = np.ceil(radius).astype(int)  # np.ceil 向上取整
    radius_ceil = np.array(radius).astype(int)
    locs = np.meshgrid(  # 二维坐标 -7到7
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)  # (201 201 2)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2  # 看看哪些是spot的范围
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin


def shrink_mask(x, size):
    size = size * 2 + 1
    x = uniform_filter(x.astype(float), size=size)
    x = np.isclose(x, 1)
    return x



def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, domain="Unknown", verbose=True):
    mask = load_image(filename, verbose=False)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    if verbose:
        print(f'Mask loaded from {filename} for {domain} domain')
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, domain="Unknown", verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename} for {domain} domain')
    return x

def load_tsv(filename, index=True, domain="Unknown"):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, header=0, index_col=None)
    print(f'Dataframe loaded from {filename} for {domain} domain')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' in kwargs.keys():
        # kwargs['sep'] = '\t'
        del kwargs['sep']
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]

def pad(emd, num):
    h, w = emd.shape[0], emd.shape[1]

    # 计算高度和宽度需要填充到最接近的7的倍数
    pad_h = (num - h % num) % num  # 高度需要填充的零的数量
    pad_w = (num - w % num) % num  # 宽度需要填充的零的数量

    # 对矩阵进行零填充
    padded_matrix = np.pad(emd,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           'constant', constant_values=0)

    # 验证结果
    new_h, new_w = padded_matrix.shape[:2]
    assert new_h % num == 0 and new_w % num == 0
    return padded_matrix

def map_spots_to_batches(embs, y, locs, batch_size_row):

    h, w, c = embs.shape
    n_batches_row = h // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    del embs
    batches = []

    start_row = 0
    for embs_batch in embs_batches:
        end_row = start_row + embs_batch.shape[0]

        batch_indices = [
            i for i, (x, y) in enumerate(locs) if start_row <= x < end_row
        ]
        if batch_indices:
            batch_y = y[batch_indices]
            batch_coords = locs[batch_indices]
        else:
            batch_y = np.array([])
            batch_coords = np.array([])

        batches.append((embs_batch, batch_y, batch_coords))
        start_row = end_row

    return batches


def map_spots_to_batches_visium(embs, y, locs, batch_size_row):
    # 获取图像的尺寸
    h, w, c = embs.shape
    # 计算批次数量
    n_batches_row = h // batch_size_row + 1
    # 按行切分 embs 成多个批次
    embs_batches = np.array_split(embs, n_batches_row)

    # 批次列表
    batches = []

    # 处理每个批次
    start_row = 0
    for embs_batch in embs_batches:
        end_row = start_row + embs_batch.shape[0]

        # 每个批次对应完整的 y 和 locs
        batch_y = y
        batch_coords = locs

        # 将当前批次的 embs_batch 和对应的 y, locs 添加到批次列表中
        batches.append((embs_batch, batch_y, batch_coords))

        # 更新 start_row，准备处理下一个批次
        start_row = end_row

    return batches


def map_spots_to_batches_visium_v2(embs, y, locs, gra_size):
    h, w, c = embs.shape

    embs_1 = pad(embs, gra_size)

    n_batches_row = embs_1.shape[0] // gra_size

    n_batches_col = embs_1.shape[1] // gra_size

    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)


    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]

    batches = []

    start_row = 0
    for embs_batch_row in embs_batches:
        for embs_batch in embs_batch_row:
            batch_y = y
            batch_coords = locs
            batches.append((embs_batch, batch_y, batch_coords))

    return batches


# def map_spots_to_grids(embs, cnts, locs, gra_size):
#     """提取嵌入块并返回一个单一的总批次"""
#     embs_batches, cnts_batches, locs_batches = [], [], []
#
#     # 对嵌入图像进行填充
#     embs_padded = pad(embs, gra_size)
#     pad_size = gra_size // 2
#
#     # 遍历所有 locs，提取 gra_size x gra_size 的块
#     for i, (x, y) in enumerate(locs):
#         x_padded = x + pad_size
#         y_padded = y + pad_size
#
#         # 计算块的起始和结束位置
#         row_start = int(x_padded - pad_size)
#         row_end = int(x_padded + pad_size + 1)
#         col_start = int(y_padded - pad_size)
#         col_end = int(y_padded + pad_size + 1)
#
#         # 检查是否超出边界
#         if row_start < 0 or row_end > embs_padded.shape[0] or col_start < 0 or col_end > embs_padded.shape[1]:
#             continue
#
#         # 提取嵌入块
#         embs_block = embs_padded[row_start:row_end, col_start:col_end, :]
#         if embs_block.shape[0] != gra_size or embs_block.shape[1] != gra_size:
#             continue
#
#         embs_batches.append(embs_block)
#         cnts_batches.append(cnts[i])
#         locs_batches.append(locs[i])
#
#     # 转换为 NumPy 数组并返回一个单一的总批次
#     embs_batches = np.array(embs_batches)  # [num, 7, 7, 1029]
#     cnts_batches = np.array(cnts_batches)  # [num, 1000]
#     locs_batches = np.array(locs_batches)  # [num, 2]
#
#     # 返回一个三元组
#     return embs_batches, cnts_batches, locs_batches


#
def map_spots_to_batches_t(embs, y, locs, batch_size_row):
    """
    按照 batch_size_row 对 embs 进行分块，同时将 y 和 locs 对应分配到各个批次中。
    如果第 0 个或最后一个批次为空，则自动填充随机数据。
    """
    h, w, c = embs.shape
    n_batches_row = max(1, h // batch_size_row + 1)  # 确保至少有一个批次
    embs_batches = np.array_split(embs, n_batches_row, axis=0)
    del embs  # 删除原始 embs 释放内存
    batches = []

    start_row = 0
    for embs_batch in embs_batches:
        end_row = start_row + embs_batch.shape[0]

        # 筛选当前批次的 locs 和 y
        batch_indices = [i for i, (x, y) in enumerate(locs) if start_row <= x < end_row]
        if batch_indices:
            batch_y = y[batch_indices]
            batch_coords = locs[batch_indices]
        else:
            batch_y = np.array([])
            batch_coords = np.array([])


        batches.append((embs_batch, batch_y, batch_coords))
        start_row = end_row



    return batches











