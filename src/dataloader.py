#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load all the necessary CoDraw data, CR labels, pretrained embeddings into 
a single Pytorch Dataset object.
"""

from collections import namedtuple, Counter
import json
from pathlib import Path
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aux import Game, get_objects, labels_dic
from cliparts import file2obj

# 56 objects + 2 people in 7 body poses and 5 facial expressions
N_CLIPARTS = 126
# maximum number of cliparts in a scene, necessary for padding
MAX_CLIPARTS = 17
# canvas size
WIDTH = 500
HEIGHT = 400
# number of features in symbolic representation
# features: clipart index, position x, position y, size, orientation
N_FEATURES = 5

Turn = namedtuple('Turn', ['teller', 'drawer', 'label'])
Embeddings = namedtuple("Embeddings", ['utterances', 'contexts'])


class CodrawData(Dataset):
    """Build all datapoints."""
    def __init__(self, split, config, vocab=None, quick_load=True):
        """Initialize a Dataset object.

        Args:
            split (str): train, val or test
            config (dataclass): dataclass with experiment configuration
        """
        self.quick_load = quick_load
        self.split = split
        self.labels_dic = labels_dic
        self.clipart_dic = self._build_clipart_dic()
        self.extract_config(config)
        self.images, self.games, self.datapoints, self.embs = self._construct()
        self.vocab = vocab
        if self.split == 'train' and self.use_bow:
            self.vocab = self._build_vocab()
        self.sizes = self.stats()

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        game_id, turn = self.datapoints[idx]
        context, last_msg, label, img, gallery, cr_flag = self._get_data(
            game_id, turn)
        return idx, context, last_msg, img, gallery, cr_flag, label

    @property
    def n_labels(self):
        """Return number of class labels."""
        return len(self.labels_dic)

    def extract_config(self, config):
        """Extract necessary hyperparameters."""
        self.codraw_path = Path(config.path_to_codraw)
        self.annotation_path = Path(config.path_to_annotation)
        self.preproc_path = Path(config.path_to_preprocessed)
        self.task = config.task
        self.img_pretrained = config.img_pretrained
        self.text_pretrained = config.text_pretrained
        self.downsample = config.downsample
        self.upsample = config.upsample
        self.filter = config.filter
        self.with_peek = not config.only_until_peek
        self.use_bow = config.msg_bow
        self.remove_first = config.remove_first
        self.no_msg = config.no_msg

    def _build_clipart_dic(self):
        """Map ids to cliparts in CoDraw."""
        all_cliparts = [x for x in file2obj.keys() if x != 'background.png']
        clip2id = {name: i for i, name in enumerate(all_cliparts)}
        assert len(clip2id) == N_CLIPARTS
        id2clip = {i: clip for clip, i in clip2id.items()}
        assert len(id2clip) == N_CLIPARTS
        return id2clip

    def _construct(self):
        """Load all information and create datapoints dictionary."""
        images = self._load_images()
        codraw = self._load_codraw()
        crs = self._load_crs()
        text_embeddings = self._load_text_embeddings()
        datapoints = {}
        games = {}
        for name, infos in codraw['data'].items():
            if self.split in name:
                game_idx = int(name.split('_')[1])
                game = Game(name, infos, crs, quick_load=self.quick_load,
                            with_peek=self.with_peek)
                if self.filter and not game.cr_turns:
                    # ignore dialogues without CRs
                    continue
                games[game_idx] = game
                for turn in range(game.n_turns):
                    if turn == 0 and self.remove_first:
                        continue
                    label = self._get_label(game, turn)
                    # downsample negative label on train set
                    if (self.split == 'train' and label == 0
                       and random.random() > self.downsample):
                        # skip the datapoint
                        continue
                    idx = len(datapoints)
                    datapoints[idx] = (game_idx, turn)
                    if (self.split == 'train' and label == 1
                        and self.upsample > 0):
                        # duplicate the datapoint
                        for i in range(self.upsample):
                            idx = len(datapoints)
                            datapoints[idx] = (game_idx, turn)
        return images, games, datapoints, text_embeddings

    def stats(self):
        """Print and store basic descriptive statistics."""
        n_cr = sum([1 for idx in range(len(self))
                    if self[idx][-1] == self.labels_dic['cr']])
        n_other = sum([1 for idx in range(len(self))
                       if self[idx][-1] == self.labels_dic['not_cr']])
        perc_cr = 100 * n_cr / len(self)
        perc_other = 100 * n_other / len(self)

        print(f'{"---"*10}\n Loaded {self.split} set with:')
        print(f'   {len(self)} datapoints')
        print(f'   {len(self.games)} dialogues')
        print(f'   {n_cr} ({perc_cr:.2f}%) clarifications')
        print(f'   {n_other} ({perc_other:.2f}%) other \n{"---"*10}')

        stats = {f'{self.split}: n_cr': n_cr,
                 f'{self.split}: %_cr': perc_cr,
                 f'{self.split}: n_other': n_other,
                 f'{self.split}: %_other': perc_other,
                 f'{self.split}: n_games': len(self.games),
                 f'{self.split}: n_datapoints': len(self)}
        return stats

    def _load_codraw(self):
        """Read CoDraw JSON file."""
        with open(self.codraw_path, 'r') as f:
            data = json.load(f)
        return data

    def _load_images(self):
        """Read image embeddings."""
        if self.img_pretrained == 'symbolic':
            return None
        directory = Path(f'{self.preproc_path}/images/{self.img_pretrained}/')
        which = '' if self.task == 'drawer' else 'orig-'
        fname = directory / f'codraw_{which}img-embeddings_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        images = {int(key): value[:] for key, value in file.items()}
        file.close()
        # we want to shift the images, because at step 0 the image the drawer
        # sees is empty and then it must see the image state at the beginning
        # of the current round; so we add an empty scene to all sequences
        if self.task == 'drawer':
            # FIXME: this is a hack to get an empty scene embedding
            if self.split == 'train':
                index = 0
            if self.split == 'val':
                index = 8
            if self.split == 'test':
                index = 9
            empty_scene = np.expand_dims(images[index][0], axis=0)
            images = {key: np.concatenate([empty_scene, emb])
                      for key, emb in images.items()}
        return images

    def _load_crs(self):
        """Read CR labels."""
        annotated = pd.read_csv(self.annotation_path, sep='\t')
        sentences = annotated['drawer\'s utterance']
        labels = annotated['is CR?']
        crs = set([sent for sent, label in zip(sentences, labels) if label])
        return crs

    def _load_text_embeddings(self):
        """Read text embeddings for contexts and utterances."""
        # utterance embeddings
        directory = Path(f'{self.preproc_path}/text/{self.text_pretrained}/')
        fname = directory / f'codraw_utterances_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        utt_embs = {int(key): value[:] for key, value in file.items()}
        file.close()
        # cumulative context embeddings
        fname = directory / f'codraw_dialogues_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        cum_embs = {int(key): value[:] for key, value in file.items()}
        file.close()

        # if no_msg is True, we'll either not use it or default back to 
        # random vectors for the case where we feed no inputs at all
        if self.no_msg:
            utt_embs = {int(key): np.random.rand(*value.shape).astype(np.float32) 
                        for key, value in utt_embs.items()}
        # return a namedtuple("Embeddings", ['utterances', 'contexts'])
        return Embeddings(utt_embs, cum_embs)

    def _get_data(self, game_id, turn):
        """Retrieve a datapoint's components."""
        game = self.games[game_id]
        context = self._get_context(game_id, turn)
        last_msg = self._get_utterance(game_id, turn)
        label = self._get_label(game, turn)
        img = self._get_img(game_id, turn)
        gallery = self._get_symbols(game, turn, gallery=True)
        cr_flag = self._get_cr_flag(game, turn)
        return context, last_msg, label, img, gallery, cr_flag

    def _get_utterance(self, game_id, turn):
        """Retrive an utterance's representation."""
        if self.task == 'drawer' and self.use_bow:
            plain_utterance = self.games[game_id].teller_turns[turn].split()
            return self._build_bow(plain_utterance)
        if self.task == 'drawer':
            # get the last teller's utterance that the drawer has to classify
            dim = 0
        elif self.task == 'teller':
            # get the last drawer's utterance that the teller has to classify
            dim = 1

        return torch.tensor(self.embs.utterances[game_id][turn][dim])

    def _build_bow(self, utterance):
        unk_id = self.vocab['UNK']
        indexed_utt = [self.vocab[w] if w in self.vocab else unk_id
                       for w in utterance]
        bow = [1 if i in indexed_utt else 0 for i in range(len(self.vocab))]
        return torch.tensor(bow).float()

    def _get_context(self, game_id, turn):
        """Retrieve a context's representation."""
        if self.task == 'drawer':
            # get context up to the teller's utterance
            dim = 0
        elif self.task == 'teller':
            # get context after teller's and up to the drawer's utterance
            dim = 1
        return torch.tensor(self.embs.contexts[game_id][turn][dim])

    def _get_label(self, game, turn):
        """Retrieve a label."""
        return (self.labels_dic['cr'] if turn in game.cr_turns
                else self.labels_dic['not_cr'])

    def _get_img(self, game_id, turn):
        """Retrieve an image representation."""
        if self.img_pretrained == 'symbolic':
            game = self.games[game_id]
            img = self._get_symbols(game, turn)
        else:
            if self.task == 'drawer':
                img = torch.tensor(self.images[game_id][turn])
            elif self.task == 'teller':
                img = torch.tensor(self.images[game_id])
            img = img.float().squeeze(0)
        return img

    def _get_symbols(self, game, turn, gallery=False):
        """Retrieve a list of objects in a gallery or scene."""
        scene = game.scenes[turn]
        is_empty = bool(scene)
        # sometimes a scene is empty in JSON, so we get the next one
        # because the objects in the gallery are fixed throught a game
        step = 0
        while not scene:
            step += 1
            scene = game.scenes[turn + step]
        cliparts = get_objects(scene)
        if gallery:
            # return the gallery representation
            symbols = self._build_gallery(cliparts, is_empty)
        else:
            # return the symbolic scene representation
            symbols = self._build_symbolic_img(cliparts, is_empty)
        return torch.tensor(symbols)

    @staticmethod
    def _get_cr_flag(game, turn):
        """Retrive the CR flag."""
        crs_before = [x for x in range(turn) if x in game.cr_turns]
        if crs_before:
            return 1
        return 0

    def _build_gallery(self, cliparts, is_empty):
        """Build gallery representation."""
        canvas, gallery = self._get_canvas_gallery(cliparts, is_empty)
        symbolic = []
        for i in range(N_CLIPARTS):
            png = self.clipart_dic[i]
            if png not in gallery:
                assert png not in canvas
            else:
                symbolic += [i]
        # some galleries have duplicate objects, so we pad the dimensions
        if len(symbolic) != 28:
            symbolic += (28 - len(symbolic))*[N_CLIPARTS]
            assert len(symbolic) == 28
        return symbolic

    def _build_symbolic_img(self, cliparts, is_empty):
        """Build an image's symbolic representation.

        Args:
            cliparts (list of aux.Clipart): cliparts in a gallery.
            is_empty (bool): True if the scene is empty.

        Returns:
            np.array: Array of dim=(MAX_CLIPARTS, N_FEATURES) representing 
            the state of the scene, and padding at the bottom.
        """
        canvas, gallery = self._get_canvas_gallery(cliparts, is_empty)
        symbolic = np.zeros((MAX_CLIPARTS, N_FEATURES))
        n = 0
        # always add the existing cliparts in the same order
        for i in range(N_CLIPARTS):
            png = self.clipart_dic[i]
            if png not in gallery:
                assert png not in canvas
            else:
                if png in canvas:
                    symbolic[n, :] = self._clipart_features(i, canvas[png])
                    n += 1
        # pad the remaining rows with corresponding indexes
        for i in range(n, MAX_CLIPARTS):
            # TODO: 0.0 is not optimal to pad position because it's valid
            symbolic[i, :] = [N_CLIPARTS, 0.0, 0.0, 3, 2]
        return symbolic

    @staticmethod
    def _get_canvas_gallery(cliparts, is_empty):
        """Get lists of objects in the scene and in the gallery.

        'is_empty' is needed for cases where we borrowed a scene string
        from another turn, to deal with empty scene string in the JSON.

        Args:
            cliparts (list of aux.Clipart): list of cliparts in a scene str.
            is_empty (bool): Is the scene empty?

        Returns:
            tuple of lists: cliparts in scene, cliparts in gallery
        """
        gallery_objects = [c.png for c in cliparts]
        if is_empty:
            canvas_objects = []
        else:
            canvas_objects = {c.png: c for c in cliparts if c.exists}
            # check that no duplicate objects occur in scene
            # to ensure that the dictionary representation contains all
            assert len(canvas_objects) == len([x for x in cliparts if x.exists])
        return canvas_objects, gallery_objects

    @staticmethod
    def _clipart_features(i, clipart):
        """Build a vector representing a clipart in a scene.

        Args:
            i (int): clipart id in internal dictionary
            clipart (aux.Clipart): the Clipart object with its info

        Returns:
            list: features of a clipart in a scene
        """
        x = float(clipart.x) / WIDTH
        y = float(clipart.y) / HEIGHT
        z = float(clipart.z)
        flip = float(clipart.flip)
        # features: index, position x, position y, size, orientation
        return [i, x, y, z, flip]

    def _build_vocab(self):
        vocab = Counter()
        MIN_FREQ = 2
        for game in self.games.values():
            dialogue = game.get_dialogue()
            for turn in dialogue:
                if self.task == 'drawer':
                    utterance = turn.teller
                    vocab.update(utterance.split())
                else:
                    raise NotImplementedError

        filtered = {word: i for i, (word, count) in enumerate(vocab.items())
                    if count >= MIN_FREQ}
        filtered['UNK'] = len(filtered)
        return filtered
