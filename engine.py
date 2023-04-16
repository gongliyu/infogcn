from pathlib import Path
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy

from models import SpVGCN
from loss import LabelSmoothingCrossEntropy, get_mmd_loss


class SpVGCNTraining(pl.LightningModule):
    def __init__(self, num_classes=8, lr=3e-4):
        super().__init__()
        self.model = SpVGCN()
        self.lr = lr
        self.loss_fn = LabelSmoothingCrossEntropy()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.lambda_1 = 0.0001
        self.lambda_2 = 0.1
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, z = self.model(x)
        mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.num_classes)
        cls_loss = self.loss_fn(logits, y)
        loss = self.lambda_2 * mmd_loss + self.lambda_1 * l2_z_mean + cls_loss
        self.train_acc(logits, y)
        self.log('mmd_loss', mmd_loss)
        self.log('l2_z_mean', l2_z_mean)
        self.log('cls_loss', cls_loss)
        self.log('loss', loss)
        self.log('training_accuracy', self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        self.val_acc(logits, y)
        self.log('validation_accuracy', self.train_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        self.test_acc(logits, y)
        self.log('test_accuracy', self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': optimizer}
