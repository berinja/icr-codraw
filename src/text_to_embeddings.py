#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract utterance embeddings and incremental, truncated dialogue context
embeddings using SentenceTransformers.
https://www.sbert.net/docs/pretrained_models.html

MODEL_NAME, EMB_DIM and MAX_SEQ_LEN should be manually selected accordingly!

For each split, this script builds 2 h5py objects. All of them contain one
dataset for each scene ID in CoDraw.

1. 'codraw_utterances_{SPLIT}.hdf5'
2. 'codraw_dialogues_{SPLIT}.hdf5'

In 1, each dataset contains a tensor of dim (n_rounds, 2, EMB_DIM),
and the two dimensions in the middle correspond to (teller, drawer).
In 2, each dataset contains a tensor of dim (n_rounds, 2, EMB_DIM),
and the two dimensions in the middle correspond to (context before teller,
context after teller but before drawer).

"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
from numpy import concatenate as cat
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from aux import Game


# Select one SentenceTransformer pretrained model, update the embedding 
# dimension accordingly and pick a number lower than the model's maximum
# sequence length to truncate the context (we don't pick the maximum itself
# because of subtokenization)
MODEL_NAME = 'all-mpnet-base-v2'
EMB_DIM = 768
MAX_SEQ_LEN = 200

SPLITS = ('train', 'val', 'test')
CODRAW_PATH = Path('../data/CoDraw-master/dataset/CoDraw_1_0.json')
OUTPUT_DIR = Path(f'../data/preprocessed/text/{MODEL_NAME}/')
EMBS = 'codraw_utterances_{}.hdf5'
CUM_EMBS = 'codraw_dialogues_{}.hdf5'

# Define which separators to use between turns.
SEP_TELLER = '/T'
SEP_DRAWER = '/D'
SEP_PEEK = '/PEEK'

os.mkdir(Path(f'../data/preprocessed/text/{MODEL_NAME}'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME).to(device)


def truncate(text):
    """Truncate left context, otherwise model probably truncates right."""
    return " ".join(text.split()[-MAX_SEQ_LEN:])


with open(CODRAW_PATH, 'r') as file:
    codraw = json.load(file)
# collect all game IDs in each split
idxs = {split: [] for split in SPLITS}
for name, game in codraw['data'].items():
    split = name.split('_')[0]
    idxs[split].append(name)


with torch.no_grad():
    for split in SPLITS:
        embeddings = OUTPUT_DIR / EMBS.format(split)
        cumulative_embeddings = OUTPUT_DIR / CUM_EMBS.format(split)
        with h5py.File(embeddings, 'w') as f1:
            with h5py.File(cumulative_embeddings, 'w') as f2:
                for name in tqdm(idxs[split]):
                    # passing an empty CR set, because they are not needed here
                    game = Game(name, codraw['data'][name], set())
                    dialogue = game.get_dialogue()
                    game_id = str(int(name.split('_')[1]))
                    n_rounds = len(codraw['data'][name]['dialog'])
                    # initialize empty context
                    context = ''
                    # store the sequence of (teller's emb, drawer's emb)
                    sequence = []
                    # store the sequence of dialogue contexts embeddings
                    # (before round, after teller's utterance)
                    # the first "before round" one is always an empty context
                    # the one after the last drawer's utterance is not
                    # necessary because it is not a context for any observed
                    # next round
                    cum_sequence = []

                    for t, turn in enumerate(dialogue):
                        # add a marker before the round if a peek occurs
                        if t == game.peek_turn:
                            context += f' {SEP_PEEK} '
                            context = truncate(context)
                        # context before round, i.e. empty at first and then
                        # after drawer's last utterance
                        embedding_c1 = model.encode([context])
                        # teller turn embedding
                        turn_t = dialogue[t].teller
                        embedding_t = model.encode([turn_t])
                        # context after teller and before drawer
                        context += f' {SEP_TELLER} {turn_t}'
                        context = truncate(context)
                        embedding_c2 = model.encode([context])
                        # drawer turn embedding
                        turn_d = dialogue[t].drawer
                        embedding_d = model.encode([turn_d])
                        # update context with drawer turn, for next context
                        context += f' {SEP_DRAWER} {turn_d}'
                        context = truncate(context)
                        # the last context won't be necessary because there is
                        # no utterance after the drawer's last utterance

                        # create tuples for this round
                        seq_round = cat([embedding_t, embedding_d], axis=0)
                        cum_seq_round = cat([embedding_c1, embedding_c2], axis=0)
                        # update sequences
                        sequence.append(seq_round)
                        cum_sequence.append(cum_seq_round)

                    sequence = np.array(sequence)
                    cum_sequence = np.array(cum_sequence)
                    assert sequence.shape == (n_rounds, 2, EMB_DIM)
                    assert cum_sequence.shape == (n_rounds, 2, EMB_DIM)
                    f1.create_dataset(game_id, data=sequence)
                    f2.create_dataset(game_id, data=cum_sequence)
