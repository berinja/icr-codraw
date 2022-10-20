#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect configuration of the experiment passed as command line args.
# Using format idea from:
"""

import argparse
from dataclasses import dataclass


@dataclass
class Parameters:
    """Store all experiment details as a dataclass."""
    path_to_codraw: str
    path_to_annotation: str
    path_to_preprocessed: str
    path_to_cliparts: str
    path_to_checkpoints: str

    ignore_comet: bool
    comet_key: str
    comet_project: str
    comet_workspace: str

    img_input_dim: int
    context_input_dim: int
    utterance_input_dim: int
    img_embedding_dim: int
    context_embedding_dim: int
    last_msg_embedding_dim: int
    gallery_embedding_dim: int
    univ_embedding_dim: int
    use_gallery: bool
    no_context: bool
    no_image: bool
    no_msg: bool
    use_cr_flag: bool
    weight_cr: float
    decision_threshold: float
    downsample: float
    upsample: int
    no_regression: bool
    filter: bool
    only_until_peek: bool
    text_pretrained: str
    img_pretrained: str
    msg_bow: bool
    remove_first: bool

    model: str
    random_seed: int
    device: str
    task: str
    batch_size: int
    n_epochs: int
    lr: float
    dropout: float
    clip: float
    accumulate_grad: int
    weight_decay: float
    gamma: float
    lr_scheduler: str
    lr_step: int


def args():
    """Parse arguments given by user and return a dataclass."""
    parser = argparse.ArgumentParser(
        description='Training the ICR recognition models in CoDraw task.')
    # _________________________________ PATHS _________________________________
    parser.add_argument('-path_to_codraw',
                        default='../data/CoDraw-master/dataset/CoDraw_1_0.json',
                        type=str, help='Path to CoDraw json file.')
    parser.add_argument('-path_to_annotation',
                        default='../data/cr_anno-main/data.tsv',
                        type=str, help='Path to CR annotation tsv file.')
    parser.add_argument('-path_to_preprocessed',
                        default='../data/preprocessed/',
                        type=str, help='Path to dir of preprocessed files.')
    parser.add_argument('-path_to_cliparts',
                        default='../data/CoDraw-master/Pngs/',
                        type=str, help='Path to dir of cliparts.')
    parser.add_argument('-path_to_checkpoints',
                        default='./checkpoints/',
                        type=str, help='Path to log checkpoints.')

    # _________________________________ COMET _________________________________
    parser.add_argument('-ignore_comet', action='store_true',
                        help='Do not log details to Comet_ml.')
    parser.add_argument('-comet_key', default='',
                        type=str, help='Comet.ml personal key.')
    parser.add_argument('-comet_project', default='cr-codraw-dev',
                        type=str, help='Comet.ml project name.')
    parser.add_argument('-comet_workspace', default='',
                        type=str, help='Comet.ml workspace name.')

    # ______________________________ SETTING __________________________________
    parser.add_argument('-random_seed', default=2719, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('-device', default='gpu', type=str,
                        choices=['cpu', 'gpu'], help='Which device to use.')
    parser.add_argument('-task', default='drawer', type=str,
                        choices=['teller', 'drawer'],
                        help='Which task to train classifier on.')
    parser.add_argument('-text_pretrained', default='all-mpnet-base-v2',
                        type=str,
                        #choices=['all-mpnet-base-v2', 'all-distilroberta-v1'],
                        help='Which pretrained text embedding model to use.')
    parser.add_argument('-img_pretrained', default='resnet101', type=str,
                        choices=['symbolic', 'vgg16', 'resnet101',
                                 'resnet101_resized_centered'],
                        help='Which pretrained image embedding model to use.')
    parser.add_argument('-no_context', action='store_true',
                        help='Do not use dialogue context as input.')
    parser.add_argument('-no_image', action='store_true',
                        help='Do not use image as input.')
    parser.add_argument('-use_gallery', action='store_true',
                        help='Do not use gallery cliparts as input.')
    parser.add_argument('-no_msg', action='store_true',
                        help='Do not use the last utterance as input.')
    parser.add_argument('-use_cr_flag', action='store_true',
                        help='Use cr_flag in input.')
    parser.add_argument('-no_regression', action='store_true',
                        help='No regression, use two-class output instead.')
    parser.add_argument('-msg_bow', action='store_true',
                        help='Represent msg with BOW.')
    parser.add_argument('-downsample', default=1, type=float,
                        help='Proportion of not CRs in the train data.')
    parser.add_argument('-upsample', default=0, type=int,
                        help='Proportion of extra CRs in the train data.')
    parser.add_argument('-filter', action='store_true',
                        help='Do not use dialogues with no CRs.')
    parser.add_argument('-only_until_peek', action='store_true',
                        help='Use dialogues only up to the peek action.')
    parser.add_argument('-remove_first', action='store_true',
                        help='Ignore first turns.')

    # __________________________ TRAINING PARAMS ______________________________
    parser.add_argument('-model', default='core', type=str,
                        choices=['basic', 'transformer', 'core'],
                        help='Which model architecture to use.')
    parser.add_argument('-batch_size', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('-img_input_dim', default=2048, type=int,
                        help='Size of pretrained image embedding.')
    parser.add_argument('-context_input_dim', default=768, type=int,
                        help='Size of pretrained context embedding.')
    parser.add_argument('-utterance_input_dim', default=768, type=int,
                        help='Size of pretrained utterance embedding.')
    parser.add_argument('-img_embedding_dim', default=128, type=int,
                        help='Size of internal image embedding.')
    parser.add_argument('-context_embedding_dim', default=128, type=int,
                        help='Size of internal context embedding.')
    parser.add_argument('-last_msg_embedding_dim', default=128, type=int,
                        help='Size of internal utterance embedding.')
    parser.add_argument('-gallery_embedding_dim', default=128, type=int,
                        help='Size of internal symbols embedding.')
    parser.add_argument('-univ_embedding_dim', default=128, type=int,
                        help='Size of all embedding dims in transformer.')
    parser.add_argument('-n_epochs', default=20, type=int,
                        help='Number of epochs.')
    parser.add_argument('-lr', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('-dropout', default=0.2, type=float,
                        help='Droupout.')
    parser.add_argument('-clip', default=1, type=float,
                        help='Clipping size, use 0 for no clipping.')
    parser.add_argument('-weight_cr', default=1.233669057742608, type=float,
                        help='Weight for positive class in loss function, \
                            0 for automatic computation.')
    parser.add_argument('-decision_threshold', default=0.5, type=float,
                        help='Threshold for regression decision.')
    parser.add_argument('-accumulate_grad', default=25, type=int,
                        help='Steps for batch gradient accumulation.')
    parser.add_argument('-gamma', default=0.99, type=float,
                        help='Gamma for the LR scheduler.')
    parser.add_argument('-weight_decay', default=0.0001, type=float,
                        help='Weight decay for L2 regularisation.')
    parser.add_argument('-lr_scheduler', default='none', type=str,
                        choices=['none', 'exp', 'step'],
                        help='Which lr scheduler to use.')
    parser.add_argument('-lr_step', default=4, type=int,
                        help='Which step to use if lr_scheduler is step.')

    parameters = Parameters(**vars(parser.parse_args()))

    return parameters
