import argparse
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram

from utils.utils import load_pickle, save_pickle, sort_labels, load_mask
from utils.image_funtions import smoothen, upscale
from utils.visual import plot_labels, plot_label_masks
from smooth_functions import relabel_small_connected, cluster_connected
from dimmention_funcitons import reduce_dim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='embeddings-hist.pickle')
    parser.add_argument('data_prefix', type=str)
    parser.add_argument('result_prefix', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--method', type=str, default='km')
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    parser.add_argument('--clusters-path', type=str, default='clusters-hist')
    return parser.parse_args()


def cluster_sub(embs, labels, n_clusters, location_weight, method):
    if labels.ndim == 2:
        labels = labels[..., np.newaxis]
    labs_uniq = np.unique(labels.reshape(-1, labels.shape[-1]), axis=0)
    labels_sub = np.full_like(labels[..., [0]], -1)
    for lab in labs_uniq:
        isin = (labels == lab).all(-1)
        if (lab >= 0).all():
            embs_sub = embs.copy().transpose(1, 2, 0)
            embs_sub[~isin] = np.nan
            embs_sub = embs_sub.transpose(2, 0, 1)
            labs_sub, _ = cluster(embs_sub, n_clusters, method, location_weight)
            labs_sub[isin] -= labs_sub[isin].min()
            labels_sub[isin] = labs_sub[isin][..., np.newaxis]
    return labels_sub[..., 0], None


def plot_dendrogram(model, outfile):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        counts[i] = sum(1 if child < n_samples else counts[child - n_samples] for child in merge)

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    plt.figure(figsize=(8, 8))
    dendrogram(Z=linkage_matrix, p=model.n_clusters_, truncate_mode='lastp', color_threshold=-1)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def cluster(embs, n_clusters, method='mbkm', location_weight=None, sort=True):
    x, mask = prepare_for_clustering(embs, location_weight)
    print(f'Clustering using {method}...')
    t0 = time()
    if method == 'mbkm':
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    elif method == 'km':
        model = KMeans(n_clusters=n_clusters, random_state=0)
    elif method == 'gm':
        model = GaussianMixture(n_components=n_clusters, covariance_type='diag', init_params='k-means++', random_state=0, verbose=1)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')

    labels = model.fit_predict(x)
    print(f'{int(time() - t0)} sec | n_clusters: {np.unique(labels).size}')
    if sort:
        labels = sort_labels(labels)[0]

    labels_arr = np.full(mask.shape, labels.min() - 1, dtype=int)
    labels_arr[mask] = labels
    return labels_arr, model


def prepare_for_clustering(embs, location_weight):
    mask = np.all([np.isfinite(c) for c in embs], axis=0)
    embs = np.stack([c[mask] for c in embs], axis=-1)

    if location_weight is None:
        return embs, mask

    embs -= embs.mean(0)
    embs /= embs.var(0).sum()**0.5

    locs = np.meshgrid(*[np.arange(mask.shape[i]) for i in range(mask.ndim)], indexing='ij')
    locs = np.stack(locs, -1).astype(float)[mask]
    locs -= locs.mean(0)
    locs /= locs.var(0).sum()**0.5

    embs *= 1 - location_weight
    locs *= location_weight
    return np.concatenate([embs, locs], axis=-1), mask


def reduce_embs_dim(x, **kwargs):
    x = reduce_dim(np.stack(x, -1), **kwargs)[0]
    return x.transpose(2, 0, 1)


def cluster_hierarchical(x_major, method, n_clusters, x_minor=None, min_cluster_size=None, reduce_dimension=False, location_weight=None):
    if reduce_dimension:
        x_major = reduce_embs_dim(x_major, method='pca', n_components=0.99)
        if x_minor is not None:
            x_minor = reduce_embs_dim(x_minor, method='pca', n_components=0.99)

    labels_cls, _ = cluster(x_major, method=method, n_clusters=n_clusters, location_weight=location_weight)
    if min_cluster_size is not None:
        labels_cls = relabel_small_connected(labels_cls, min_size=min_cluster_size)

    labels_con = cluster_connected(labels_cls)

    if x_minor is not None:
        labels_sub, _ = cluster_sub(x_minor, labels=labels_cls, method=method, n_clusters=4, location_weight=None)
        return np.stack([labels_cls, labels_sub, labels_con], -1)
    return np.stack([labels_cls, labels_con], -1)


def plot_scatter(x, y, lab, outfile):
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=lab, cmap='tab10', alpha=0.2)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def upscale_label(lab, target_shape):
    onehot = [lab == la for la in range(lab.max() + 1)]
    prob = [upscale(oh.astype(np.float32)[..., np.newaxis], target_shape)[..., 0] for oh in onehot]
    return np.argmax(prob, 0)


def cluster_rescale(x, stride, method, n_clusters):
    img_shape = x[0].shape
    isin = np.isfinite(x[0])
    x = x[:, stride // 2::stride, stride // 2::stride]
    lab, model = cluster(x, method=method, n_clusters=n_clusters)
    label = upscale_label(lab, img_shape)
    label[~isin] = -1
    return label, model


def flatten_label(label):
    img_shape = label.shape[:-1]
    label = label.reshape(-1, label.shape[-1])
    n_bins = label.max()
    label = label[:, ::-1].T
    label = np.sum([lab * n_bins**i for i, lab in enumerate(label)], 0)
    label[label < 0] = -1
    label = np.unique(label, return_inverse=True)[1] - 1
    return label.reshape(img_shape)


def cluster_mbkmagglo(x, n_clusters):
    img_shape = x[0].shape
    isin = np.isfinite(x[0])

    n_clusters_small = n_clusters * 50
    min_cluster_size = max(1, isin.sum() // n_clusters_small // 1000)
    lab_small = cluster_hierarchical(x, method='mbkm', n_clusters=n_clusters_small, min_cluster_size=min_cluster_size, location_weight=0.1)
    lab_flat = flatten_label(lab_small)

    centroids = [x[:, lab_flat == la].mean(1) for la in range(lab_flat.max() + 1)]
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)

    print(np.shape(centroids))
    t0 = time()
    lab_cent = model.fit_predict(centroids)
    print(f'{int(time() - t0)} sec')
    lab_cent = sort_labels(lab_cent)[0]

    lab_super = lab_cent[lab_flat]
    lab_super = upscale_label(lab_super, img_shape)
    lab_super[~isin] = -1
    return lab_super, model


def smooth(x, filter_size):
    x = x.transpose(1, 2, 0)
    x = smoothen(x, filter_size)
    return x.transpose(2, 0, 1)


def cluster_and_save(x, method, n_clusters, min_cluster_size=None, result_prefix=None):
    labels, _ = cluster(x, method=method, n_clusters=n_clusters)
    if min_cluster_size is not None:
        labels = relabel_small_connected(labels, min_size=min_cluster_size)
    if result_prefix:
        save_pickle(labels, result_prefix + 'labels.pickle')
        plot_labels(labels, result_prefix + 'labels.png', white_background=True)
        plot_label_masks(labels, result_prefix + 'masks/')
    return labels


def preprocess_and_cluster(x, result_prefix=None, n_components=None, filter_size=None, n_clusters=None, min_cluster_size=None, method='km'):
    if n_components is not None:
        x = reduce_embs_dim(x, method='pca', n_components=n_components)
    if filter_size is not None:
        print('Smoothing embeddings...')
        t0 = time()
        x = smooth(x, filter_size)
        print(f'{int(time() - t0)} sec')

    if n_clusters is None:
        n_clusters_list = [10, 20, 30, 40, 50, 60, 70]
    elif np.size(n_clusters) > 1:
        n_clusters_list = n_clusters
    else:
        n_clusters_list = [n_clusters]

    labels_list = []
    for n_clusters in n_clusters_list:
        pref = f'{result_prefix}nclusters{n_clusters:03d}/' if result_prefix and len(n_clusters_list) > 1 else result_prefix
        labels = cluster_and_save(x, n_clusters=n_clusters, min_cluster_size=min_cluster_size, method=method, result_prefix=pref)
        labels_list.append(labels)
    return labels_list


def main():
    args = get_args()
    data_folder = args.data_prefix.rstrip('/').split('/')[-1]
    result_path = f"{args.result_prefix}istar/{data_folder}/{args.clusters_path}/"
    os.makedirs(result_path, exist_ok=True)

    embeddings_path = f"{args.result_prefix}istar/{data_folder}/{args.embeddings}"
    print(f"Loading embeddings from: {embeddings_path}")
    embs = load_pickle(embeddings_path)

    if isinstance(embs, dict):
        x = np.array(embs.get('cls', embs.get('sub')))
    else:
        x = embs

    if args.mask:
        mask = load_mask(args.mask)
        x[:, ~mask] = np.nan

    preprocess_and_cluster(
        x,
        n_components=args.n_components,
        filter_size=args.filter_size,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        method=args.method,
        result_prefix=result_path
    )


if __name__ == '__main__':
    main()
