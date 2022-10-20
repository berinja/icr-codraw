#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run experiments for the Instruction Clarification Requests
paper.

The model is build according to the CLI arguments, trained and tested.

Use Pytorch Lightning to structure the experiment and comet.ml to log. 
Outputs and checkpoitns are also saved locally.
"""

from pathlib import Path
import warnings

import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from logs import log_all, filter_params
from dataloader import CodrawData
from plmodel import LitClassifier
import config

# ignore Lightning warning about increasing the number of workers
warnings.filterwarnings("ignore", ".*does not have many workers.*")

params = filter_params(config.args())
pl.seed_everything(params.random_seed)

datasets = {}
datasets['train'] = CodrawData('train', params)
datasets['val'] = CodrawData('val', params, vocab=datasets['train'].vocab)
datasets['test'] = CodrawData('test', params, vocab=datasets['train'].vocab)
params.n_labels = len(datasets['train'].labels_dic)

model = LitClassifier(datasets, config=params)

logger = CometLogger(
    api_key=params.comet_key,
    workspace=params.comet_workspace,
    save_dir="comet-logs/",
    project_name=params.comet_project,
    disabled=params.ignore_comet,
    auto_metric_logging=False
)
log_all(logger, params, datasets)

lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
checkpoint_callback = ModelCheckpoint(
    dirpath=Path(params.path_to_checkpoints) / f'model_{logger.version}',
    filename='model-{epoch}-{val_BinaryAveragePrecision:.5f}',
    monitor='val_BinaryAveragePrecision',
    mode='max',
    save_top_k=1,
    )

trainer = pl.Trainer(
    accelerator=params.device,
    devices=[1],
    # limit_train_batches=10,
    # fast_dev_run=True,
    # log_every_n_steps=50,
    max_epochs=params.n_epochs,
    logger=logger,
    gradient_clip_val=params.clip if params.clip > 0 else None,
    accumulate_grad_batches=params.accumulate_grad,
    callbacks=[lr_monitor, checkpoint_callback],
    )

trainer.fit(model=model)
trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path)
