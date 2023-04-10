from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from engine import SpVGCNTraining


def main(clargs):
    # load data
    with open(clargs.data_path / 'train.pkl', 'rb') as f:
        X_tr, Y_tr = pickle.load(f)
    with open(clargs.data_path / 'test.pkl', 'rb') as f:
        X_te, Y_te = pickle.load(f)
    nC = int(Y_tr.max()) + 1
    print(f'nC = {nC}')

    X_tr = np.transpose(X_tr, [0, 3, 1, 2])[..., np.newaxis]
    X_te = np.transpose(X_te, [0, 3, 1, 2])[..., np.newaxis]
    print(f'X_tr shape={X_tr.shape}')

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr = torch.tensor(Y_tr, dtype=torch.int64)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    Y_te = torch.tensor(Y_te, dtype=torch.int64)

    train_data = TensorDataset(X_tr, Y_tr)
    train_data_size = int(len(train_data) * 0.8)
    valid_data_size = len(train_data) - train_data_size
    # split 20% for validation
    train_data, val_data = random_split(train_data, [train_data_size, valid_data_size])
    test_data = TensorDataset(X_te, Y_te)

    train_loader = DataLoader(train_data, batch_size=clargs.batch_size)
    val_loader = DataLoader(val_data, batch_size=clargs.batch_size)
    test_loader = DataLoader(test_data, batch_size=clargs.batch_size)
    model = SpVGCNTraining(lr=clargs.lr)

    clargs.logger = pl_loggers.TensorBoardLogger(clargs.log_dir)
    trainer = pl.Trainer.from_argparse_args(
        clargs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='validation_accuracy',
                mode='max'
            ),
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(dataloaders=test_loader)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path('data'))
    parser.add_argument('--log-dir', type=Path, default=Path('log'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoint'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    pl.Trainer.add_argparse_args(parser)
    clargs = parser.parse_args()

    main(clargs)