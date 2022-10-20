#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auxiliary functions to verify and log an experiment's hyperparameters.
Outputs class accumulates results throught an epoch and saves it.
"""

import csv
import json
import os
from pathlib import Path


def log_all(logger, params, datasets):
    """Log CLI arguments to the logger and to a local folder."""
    path_name = Path(f'outputs/{logger.version}')
    os.mkdir(path_name)
    params_dic = {k: v for k, v in params.__dict__.items() if k != 'comet_key'}
    logger.experiment.log_parameters(params_dic)
    logger.experiment.log_code('model.py')
    logger.experiment.log_code('dataloader.py')
    logger.experiment.log_code('plmodel.py')
    logger.experiment.log_code('aux.py')
    logger.experiment.log_code('main.py')
    logger.experiment.log_code('logs.py')
    logger.experiment.log_others(datasets['train'].sizes)
    logger.experiment.log_others(datasets['val'].sizes)
    logger.experiment.log_others(datasets['test'].sizes)

    out_path = Path(f'outputs/{logger.version}/config.json')
    with open(out_path, 'w') as file:
        json.dump(params_dic, file)


def filter_params(config):
    """Remove or verify hyperparameters according to architecture."""
    img_models = ('resnet101', 'resnet101_resized_centered', 'vgg16')
    if config.img_pretrained in img_models:
        assert config.img_input_dim == 2048
    text_models = ('all-mpnet-base-v2',)
    if config.text_pretrained in text_models:
        assert config.context_input_dim == 768
    # TODO assert bow dimension
    if config.no_regression:
        assert 0 <= config.weight_cr <= 1
    if config.model == 'basic':
        config.univ_embedding_dim = None
    elif config.model == 'transformer':
        config.img_embedding_dim = None
        config.context_embedding_dim = None
        config.last_msg_embedding_dim = None
        config.gallery_embedding_dim = None
    if config.lr_scheduler == 'exp':
        config.lr_step = None
    if config.task == 'teller':
        assert not config.use_gallery, "Teller should not see the gallery!"
    return config


class Outputs:
    """Accumulate and save a model's outputs in an epoch."""
    def __init__(self, split, dataset):
        """Initialize object to log outputs.

        Args:
            split (str): train, val or test
            dataset (dataloader.CodrawData): the loaded CoDraw dataset
        """
        self.split = split
        self.datapoints = dataset.datapoints
        self.predictions = {}
        self.labels = {}
        self.probs = {}

    def update(self, batch_idxs, batch_pred, batch_label, batch_prob):
        """Accumulate outputs for a batch.

        Args:
            batch_idxs (torch.tensor): indexes of datapoints in dataset
            batch_pred (torch.tensor): predicted labels
            batch_label (torch.tensor): gold labels
            batch_prob (torch.tensor): probability assigned to positive label
        """
        for idx, pred, label, prob in zip(batch_idxs, batch_pred,
                                          batch_label, batch_prob):
            self.predictions[idx.item()] = pred.item()
            self.labels[idx.item()] = label.item()
            self.probs[idx.item()] = prob.item()

    def reset(self):
        """Restart accumulation."""
        self.predictions = {}
        self.labels = {}
        self.probs = {}

    def save(self, epoch, logger):
        """Log outputs into a csv file."""
        fname = Path(f'outputs/{logger.version}/{self.split}_{epoch}.csv')
        with open(fname, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            header = ['idx', 'game_id', 'turn', 'label', 'pred', 'prob']
            writer.writerow(header)
            for idx, (game_id, turn) in self.datapoints.items():
                writer.writerow([idx, game_id, turn,
                                 self.labels[idx], self.predictions[idx],
                                 self.probs[idx]])
        logger.experiment.log_table(fname)
