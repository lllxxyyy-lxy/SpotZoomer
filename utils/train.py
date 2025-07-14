import os
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot as plt

from utils.utils import load_pickle, save_pickle

matplotlib.use('Agg')


class MetricTracker(pl.Callback):
    """
    Callback to collect metrics during training epochs.
    """

    def __init__(self):
        self.collection = []

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        metrics = deepcopy(trainer.logged_metrics)
        self.collection.append(metrics)

    def clean(self):
        keys = set().union(*[set(e.keys()) for e in self.collection])
        for elem in self.collection:
            for key in keys:
                if key in elem:
                    if isinstance(elem[key], torch.Tensor):
                        elem[key] = elem[key].item()
                else:
                    elem[key] = float('nan')


def train_load_model(
        model_class, model_kwargs, dataset, prefix,
        epochs=None, device='cuda', load_saved=False, **kwargs):
    """
    Load model from checkpoint if available, otherwise train a new model.
    Saves model checkpoint and training history.
    """
    checkpoint_file = prefix + 'model.pt'
    history_file = prefix + 'history.pickle'

    if load_saved and os.path.exists(checkpoint_file):
        print(f'Loading model from {checkpoint_file}')
        model = model_class.load_from_checkpoint(checkpoint_file)
        history = load_pickle(history_file)
    else:
        print('Training a new model...')
        model = model_class(**model_kwargs)
        history = []

        if epochs and epochs > 0:
            model, hist, trainer = train_model(
                model=model,
                model_class=model_class,
                model_kwargs=model_kwargs,
                dataset=dataset,
                epochs=epochs,
                device=device,
                **kwargs
            )
            trainer.save_checkpoint(checkpoint_file)
            print(f'Model saved to {checkpoint_file}')
            history += hist
            save_pickle(history, history_file)
            print(f'History saved to {history_file}')
            plot_history(history, prefix)

    return model


def train_model(
        dataset, batch_size, epochs,
        model=None, model_class=None, model_kwargs={},
        device='cuda'):
    """
    Train a PyTorch Lightning model on the given dataset.
    """
    if model is None:
        model = model_class(**model_kwargs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tracker = MetricTracker()

    accelerator = {'cuda': 'gpu', 'cpu': 'cpu'}[device]
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[tracker],
        deterministic=True,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    model.train()
    t0 = time()
    trainer.fit(model=model, train_dataloaders=dataloader)
    print(f'Training time: {int(time() - t0)} sec')

    tracker.clean()
    return model, tracker.collection, trainer


def plot_history(history, prefix):
    """
    Plot training loss history and save as PNG.
    """
    plt.figure(figsize=(16, 16))
    metrics = ['src_loss', 'tgt_loss', 'pretrain_mse_loss', 'triplet_loss', 'target_mse_loss', 'total_loss']

    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        hist = [entry[metric] for entry in history if not np.isnan(entry[metric])]
        if not hist:
            continue

        hist = np.array(hist)
        hmin, hmax = hist.min(), hist.max()
        label = f'{metric} ({hmin:+013.6f}, {hmax:+013.6f})'
        hist = (hist - hmin) / (hmax - hmin + 1e-12)

        plt.plot(hist, label=label)
        plt.legend()
        plt.ylim(0, 1)
        plt.title(metric)

    outfile = f'{prefix}history.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Saved training history plot to {outfile}')
