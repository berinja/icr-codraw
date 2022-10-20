#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytorch Lightining module to structure the experiment.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy,  F1Score
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryCohenKappa, BinaryROC,
                                         BinaryPrecisionRecallCurve,
                                         BinaryPrecision, BinaryRecall,
                                         BinaryAveragePrecision,
                                         BinaryConfusionMatrix)

from aux import labels_dic
from model import CoreNetwork, UniversalTransformer, BasicNetwork
from logs import Outputs


class LitClassifier(pl.LightningModule):
    """Lightining experiment."""
    def __init__(self, datasets, config):
        """Initialize the experiment object.

        Args:
            datasets (dictionary of dataloader.CoDraw): train, val, test sets.
            config (dataclass): all experiment configuration hyperparameters.
        """
        super(LitClassifier, self).__init__()
        self.datasets = datasets
        self.config = config
        self.model = self._define_model()
        self.labels_dic = labels_dic
        self.reduction = 'sum'
        self._build_metrics()
        self._define_loss()

    def _define_model(self):
        """Load the NN model."""
        if self.config.model == 'core':
            return CoreNetwork(self.config)
        elif self.config.model == 'transformer':
            return UniversalTransformer(self.config)
        elif self.config.model == 'basic':
            return BasicNetwork(self.config)
        return None

    def _define_loss(self):
        """Define the cross entropy loss."""
        weight_cr = self.config.weight_cr
        if self.config.no_regression:
            class_weights = None
            if weight_cr > 0:
                assert weight_cr <= 1, 'weight_cr should be a probability'
                class_weights = torch.tensor([1-weight_cr, weight_cr])
            self.criterion = nn.CrossEntropyLoss(weight=class_weights,
                                                 reduction=self.reduction)
        else:
            if weight_cr == 0:
                n_positive = self.datasets['train'].sizes['train: n_cr']
                n_negative = self.datasets['train'].sizes['train: n_other']
                # according to Pytorch documentation
                # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bce#torch.nn.BCEWithLogitsLoss
                weight_cr = n_negative / n_positive
            pos_class_weight = torch.tensor(weight_cr)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_class_weight,
                                                  reduction=self.reduction)
        print(f'\nUsing weight {weight_cr} for CR class.\n')

    def _build_metrics(self):
        """Initialize all metrics."""
        threshold = self.config.decision_threshold
        n_labels = self.config.n_labels
        metrics_collection = MetricCollection([
            BinaryAccuracy(threshold=threshold),
            BinaryPrecision(threshold=threshold),
            BinaryRecall(threshold=threshold),
            BinaryAveragePrecision(),
            BinaryF1Score(threshold=threshold),
            BinaryCohenKappa(threshold=threshold),
            BinaryROC(),
            BinaryConfusionMatrix(threshold=threshold),
            BinaryPrecisionRecallCurve()
        ])

        metrics_by_class = MetricCollection([
            Accuracy(num_classes=n_labels, average='none'),
            F1Score(num_classes=n_labels, average='macro', threshold=threshold),
        ])

        self.train_metrics = metrics_collection.clone(prefix='train_')
        self.val_metrics = metrics_collection.clone(prefix='val_')
        self.test_metrics = metrics_collection.clone(prefix='test_')

        self.train_metrics_by_class = metrics_by_class.clone(prefix='train_')
        self.val_metrics_by_class = metrics_by_class.clone(prefix='val_')
        self.test_metrics_by_class = metrics_by_class.clone(prefix='test_')

        self.non_scalar_metrics = [
            'BinaryROC', 'BinaryConfusionMatrix', 'BinaryPrecisionRecallCurve']

        self.val_outputs = Outputs('val', self.datasets['val'])
        self.test_outputs = Outputs('test', self.datasets['test'])

    def forward(self, context, last_msg, image, gallery, cr_flag):
        """Call the model's forward pass."""
        output = self.model(context, last_msg, image, gallery, cr_flag)
        return output

    def _loss_pred(self, batch):
        """Get the predictions in a batch."""
        idxs, context, last_msg, image, gallery, cr_flag, label = batch
        logits = self(context, last_msg, image, gallery, cr_flag)

        if self.config.no_regression:
            loss = self.criterion(logits, label)
            # softmax not strictly necessary for prediction, but for
            # consistency we need probabilities here too, in order to
            # compute ROC curves
            # WARNING: ROC curves in this case will not make much sense
            probs = torch.softmax(logits, dim=1)
            probs_cr = probs[:, self.labels_dic['cr']]
            pred = probs.argmax(dim=1)
        else:
            loss = self.criterion(logits.view(-1), label.float())
            probs_cr = torch.sigmoid(logits).view(-1)
            pred = (probs_cr >= self.config.decision_threshold).long()


        return idxs, probs_cr, label, pred, loss

    def _compute_log_reset_metrics(self, epoch_metrics, epoch_metrics_class, split):
        """Compute, log and reset metrics."""
        # accuracy per class
        results_class = epoch_metrics_class.compute()
        acc_class = results_class[f'{split}_Accuracy']
        acc_cr, acc_not_cr = self._split_metrics(acc_class)
        self.log(f'{split}_macro-f1_epoch', results_class[f'{split}_F1Score'])
        self.log(f'{split}_acc_cr_epoch', acc_cr)
        self.log(f'{split}_acc_not_cr_epoch', acc_not_cr)

        results = epoch_metrics.compute()
        filtered = {k: v for k, v in results.items() if k.split('_')[1] not in self.non_scalar_metrics}
        self.log_dict(filtered)

        # ROC curve and area under the curve
        fpr, tpr, _ = results[f'{split}_BinaryROC']
        auc, fig = self._plot_roc(fpr, tpr)
        self.log(f'{split}_auc_epoch', auc)
        if split != 'train':
            self.logger.experiment.log_figure(
                    figure=fig,
                    figure_name=f'roc_{split}_{self.current_epoch}')
        # confusion matrix
        conf_matrix = results[f'{split}_BinaryConfusionMatrix']
        id2label = {value: key for key, value in self.labels_dic.items()}
        if split != 'train':
            self.logger.experiment.log_confusion_matrix(
                matrix=conf_matrix.cpu().numpy(),
                labels=[id2label[0], id2label[1]],
                epoch=self.current_epoch,
                file_name=f'cf_{split}_{self.current_epoch}')
        # precision-recall curve and area under the curve
        prec, recall, _ = results[f'{split}_BinaryPrecisionRecallCurve']
        if split != 'train':
            # TODO it does not work for the train split
            avp, auc_pr, fig_pr = self._plot_prcurve(
                prec, recall, epoch_metrics['BinaryPrecisionRecallCurve'], split)
            self.log(f'{split}_auc_pr_epoch', auc_pr)
            self.logger.experiment.log_figure(
                figure=fig_pr,
                figure_name=f'precrec_{split}_{self.current_epoch}')

        epoch_metrics.reset()
        epoch_metrics_class.reset()
        plt.close('all')

    def training_step(self, batch, batch_idx):
        """Training step in one batch."""
        _, probs, label, pred, loss = self._loss_pred(batch)
        self.train_metrics.update(probs, label)
        self.train_metrics_by_class.update(pred, label)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step in one batch."""
        idxs, probs, label, pred, loss = self._loss_pred(batch)

        self.val_metrics.update(probs, label)
        self.val_metrics_by_class(pred, label)
        self.val_outputs.update(idxs, pred, label, probs)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step in one batch."""
        idxs, probs, label, pred, loss = self._loss_pred(batch)
        self.test_metrics.update(probs, label)
        self.test_metrics_by_class.update(pred, label)
        self.test_outputs.update(idxs, pred, label, probs)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        """Finalize training epoch, collect metrics.""" 
        self._compute_log_reset_metrics(
            self.train_metrics, self.train_metrics_by_class, 'train')

    def validation_epoch_end(self, outputs):
        """Finalize validation, collect metrics."""
        if not self.trainer.sanity_checking:
            # sanity check that metric gets rebooted
            #assert sum([len(x) for x in self.val_roc.target]) == 7714
            self._compute_log_reset_metrics(
                self.val_metrics, self.val_metrics_by_class, 'val')
            self.val_outputs.save(self.current_epoch, self.logger)
        else:
            self.val_metrics.reset()
            self.val_metrics_by_class.reset()
        self.val_outputs.reset()

    def test_epoch_end(self, outputs):
        """Finalize test, collect metrics."""
        self._compute_log_reset_metrics(
            self.test_metrics, self.test_metrics_by_class, 'test')
        self.test_outputs.save(self.current_epoch, self.logger)
        self.test_outputs.reset()

    def on_train_end(self):
        """Log the best model."""
        version = self.logger.version
        ckpts = Path(self.config.path_to_checkpoints)
        ckpt_dir = ckpts / f'model_{version}'
        self.logger.experiment.log_asset_folder(ckpt_dir)
        best_epoch = self._get_best_epoch()
        self.logger.experiment.log_metric('best_epoch', best_epoch)

        fname = Path(f'outputs/{version}/best-epoch.txt')
        with open(fname, 'w') as file:
            file.write(str(best_epoch))

    def configure_optimizers(self):
        """Define optimizer (and scheduler)."""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            )
        if self.config.lr_scheduler != 'none':
            lr_scheduler = self._define_lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        return [optimizer]

    def _define_lr_scheduler(self, optimizer):
        """Define LR scheduler."""
        gamma = self.config.gamma
        if self.config.lr_scheduler == 'step':
            step = self.config.lr_step
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step, gamma=gamma)
        elif self.config.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma)
        return lr_scheduler

    def train_dataloader(self):
        """Define train data loader."""
        return DataLoader(self.datasets['train'],
                          batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        """Define val data loader."""
        return DataLoader(self.datasets['val'],
                          batch_size=self.config.batch_size, shuffle=False)

    def test_dataloader(self):
        """Define test data loader."""
        return DataLoader(self.datasets['test'],
                          batch_size=self.config.batch_size, shuffle=False)

    def _plot_roc(self, fpr, tpr):
        """Return AUC and ROC plot."""
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          pos_label=self.labels_dic['cr'])
        display.plot()
        display.ax_.plot([0, 1], [0, 1], 'r--')
        return roc_auc, display.figure_

    def _plot_prcurve(self, precision, recall, pr_obj, split):
        """Return AVP, AUC-PR and PR plot."""
        precision = precision.cpu().numpy()
        recall = recall.cpu().numpy()
        target = [x.cpu().tolist() for batch in pr_obj.target for x in batch]
        pred = [x.cpu().tolist() for batch in pr_obj.preds for x in batch]
        assert len(target) == len(self.datasets[split])
        assert len(pred) == len(self.datasets[split])
        avp = metrics.average_precision_score(target, pred)
        display = metrics.PrecisionRecallDisplay(precision, recall,
                                                 average_precision=avp)
        display.plot()
        auc = metrics.auc(recall, precision)
        rand_line = self.datasets[split].sizes[f'{split}: %_cr'] / 100
        display.ax_.axhline(rand_line, color='r', ls='--')
        return avp, auc, display.figure_

    def _split_metrics(self, accs):
        """Extract metrics per class."""
        acc_cr = accs[self.labels_dic['cr']]
        acc_not_cr = accs[self.labels_dic['not_cr']]
        return acc_cr, acc_not_cr

    def _get_best_epoch(self):
        """Hack the best epoch."""
        path = self.trainer.checkpoint_callback.best_model_path
        begin = re.search('model-epoch=', path).span()[1]
        end = re.search('-val_BinaryAver', path).span()[0]
        return int(path[begin: end])
