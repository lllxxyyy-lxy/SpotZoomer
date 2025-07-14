import argparse
import multiprocessing
import torch
import numpy as np
import os
from basic_functions import get_gene_counts, get_embeddings, get_locs, get_label, get_mask
from utils.utils import read_lines, read_string, save_pickle, map_spots_to_batches
from utils.train import train_load_model
from model.VISD import VISD
from dataset import SpotDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force CUDA operations to be synchronous (useful for debugging)

# Load embeddings, gene counts, locations and optionally labels and masks
def get_data(data_prefix, result_prefix, domain="Unknown"):
    gene_names = read_lines(f'{data_prefix}gene_names.txt')
    cnts = get_gene_counts(data_prefix, domain=domain).astype(np.float32)
    cnts = cnts[gene_names]
    embs = get_embeddings(result_prefix, domain=domain).astype(np.float32)
    locs = get_locs(data_prefix, target_shape=embs.shape[:2], domain=domain).astype(np.float32)
    if domain == "Source":
        label = get_label(data_prefix, domain=domain).astype(np.float32)
        mask = get_mask(data_prefix, domain=domain).astype(np.float32)
        return embs, cnts, locs, label, mask
    return embs, cnts, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

# Create and train the model, returning model and dataset
def get_model(
        src_x, src_y, src_locs, src_label, src_mask, src_radius,
        tgt_x, tgt_y, tgt_locs, tgt_radius,
        prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):

    dataset = SpotDataset(src_x, src_y, src_label, src_locs, src_radius, tgt_x, tgt_y, tgt_locs, tgt_radius)
    model = train_load_model(
        model_class=VISD,
        model_kwargs=dict(
            src_n_inp=src_x.shape[-1],
            tgt_n_inp=tgt_x.shape[-1],
            src_n_out=src_y.shape[-1],
            tgt_n_out=tgt_y.shape[-1],
            lr=lr,
            src_gnn_input_dim=src_y.shape[-1],
            tgt_gnn_input_dim=tgt_y.shape[-1],
            gnn_hidden_dim=256,
            gnn_output_dim=1000
        ),
        dataset=dataset, prefix=prefix,
        batch_size=batch_size, epochs=epochs,
        load_saved=load_saved, device=device)

    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset

# Normalize embeddings and counts
def normalize(embs, cnts):
    embs_mean = np.nanmean(embs, axis=(0, 1))
    embs_std = np.nanstd(embs, axis=(0, 1))
    embs = (embs - embs_mean) / (embs_std + 1e-12)

    cnts_min = cnts.min(axis=0)
    cnts_max = cnts.max(axis=0)
    cnts_range_safe = np.where(cnts_max - cnts_min == 0, 1.0, cnts_max - cnts_min)
    cnts = (cnts - cnts_min) / cnts_range_safe
    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)

# Predict gene expression using trained model
def predict_gene(model, x, indices, names, y_range):
    his_embs, y_exp, locs = x
    device = 'cuda'
    his_embs = torch.tensor(his_embs, device=device)
    y_exp = torch.tensor(y_exp, device=device)
    locs = torch.tensor(locs, device=device)
    gene = model.predict_gene_expression(his_embs, y_exp, locs, indices=indices)
    gene = gene.cpu().detach().numpy()
    gene *= y_range[:, 1] - y_range[:, 0]
    gene += y_range[:, 0]
    return gene

# Batch prediction for all genes, with model ensemble
def predict(model_states, x_batches, name_list, y_range, prefix, device='cuda'):
    batch_size_outcome = 100
    model_states = [mod.to(device) for mod in model_states]
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)

    for idx_grp in idx_groups:
        max_out_dim = model_states[0].gene_predict.layer2.linear.out_features
        idx_arr = np.array(idx_grp)
        valid_mask = (idx_arr >= 0) & (idx_arr < max_out_dim)
        valid_indices = idx_arr[valid_mask]

        if len(valid_indices) == 0:
            print(f"[Skipped] All indices in current group are out of bounds")
            continue

        name_grp = name_list[valid_indices]
        y_ran = y_range[valid_indices]
        y_grp = np.concatenate([
            np.median([
                predict_gene(mod, x_states, valid_indices, name_grp, y_ran)
                for mod in model_states
            ], axis=0)
            for x_states in x_batches
        ])

        for i, name in enumerate(name_grp):
            save_pickle(y_grp[..., i], f'{prefix}cnts-super/{name}.pickle')

# Main pipeline for SR4Visium training and prediction
def SR4Visium(
        src_embs, src_cnts, src_locs, src_labels, src_mask, src_radius,
        tgt_embs, tgt_cnts, tgt_locs, tgt_radius,
        epochs, batch_size, src_prefix, tgt_prefix, result_prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    src_names = src_cnts.columns
    tgt_names = tgt_cnts.columns

    src_cnts = src_cnts.to_numpy().astype(np.float32)
    tgt_cnts = tgt_cnts.to_numpy().astype(np.float32)

    _, src_cnts, _, (src_cnts_min, src_cnts_max) = normalize(src_embs, src_cnts)
    _, tgt_cnts, _, (tgt_cnts_min, tgt_cnts_max) = normalize(tgt_embs, tgt_cnts)

    kwargs_list = [
        dict(
            src_x=src_embs, src_y=src_cnts, src_locs=src_locs, src_label=src_labels,
            src_mask=src_mask, src_radius=src_radius,
            tgt_x=tgt_embs, tgt_y=tgt_cnts, tgt_locs=tgt_locs, tgt_radius=tgt_radius,
            batch_size=batch_size, epochs=epochs, lr=1e-4,
            prefix=f'{result_prefix}states/{i:02d}/',
            load_saved=load_saved, device=device)
        for i in range(n_states)
    ]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states

    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask_target.sum()

    cnts_range = np.stack([tgt_cnts_min, tgt_cnts_max], axis=-1)
    cnts_range /= mask_size

    batch_size_row = 50
    embs_batches = map_spots_to_batches(tgt_embs, tgt_cnts, tgt_locs, batch_size_row)

    predict(model_states=model_list, x_batches=embs_batches,
            name_list=tgt_names, y_range=cnts_range,
            prefix=result_prefix, device=device)

# Parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prefix', type=str, default='/home/lixiaoyu/VISD/data/MouseBrain-HD2V/SourceDomain/')
    parser.add_argument('--tgt_prefix', type=str, default='/home/lixiaoyu/VISD/data/MouseBrain-HD2V/TargetDomain/')
    parser.add_argument('--result_prefix', type=str, default='/home/lixiaoyu/VISD/result/MouseBrain-HD2V/')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-states', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--load-saved', action='store_true')
    return parser.parse_args()

# Entry point
def main():
    args = get_args()
    src_embs_prefix = args.result_prefix + args.src_prefix.rstrip('/').split('/')[-1] + '/'
    tgt_embs_prefix = args.result_prefix + args.tgt_prefix.rstrip('/').split('/')[-1] + '/'
    result_prefix = args.result_prefix

    src_embs, src_cnts, src_locs, src_label, src_mask = get_data(args.src_prefix, src_embs_prefix, domain="Source")
    tgt_embs, tgt_cnts, tgt_locs = get_data(args.tgt_prefix, tgt_embs_prefix, domain="Target")

    radius = float(read_string(f'{args.src_prefix}radius.txt')) / 16
    batch_size = 128

    SR4Visium(
        src_embs=src_embs, src_cnts=src_cnts, src_locs=src_locs, src_labels=src_label,
        src_mask=src_mask, src_radius=radius,
        tgt_embs=tgt_embs, tgt_cnts=tgt_cnts, tgt_locs=tgt_locs, tgt_radius=radius,
        epochs=args.epochs, batch_size=batch_size,
        n_states=args.n_states, src_prefix=args.src_prefix,
        tgt_prefix=args.tgt_prefix, result_prefix=result_prefix,
        load_saved=args.load_saved, device=args.device, n_jobs=args.n_jobs)

if __name__ == '__main__':
    main()
