#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural network models used in the main architecture.
"""

import torch
from torch import nn
from torch.nn import Embedding

N_CLIPARTS = 126
MAX_CLIPARTS = 17
N_SIZES = 3
N_FLIPS = 2
N_FLAGS = 2
GALLERY_SIZE = 28
HEADS = 8
LAYERS = 6

# TODO: turn into parameter?
HIDDEN_DIM = 256


class UniversalTransformer(nn.Module):
    """Transformer model that take all components as input."""
    def __init__(self, config):
        super(UniversalTransformer, self).__init__()
        self._extract_config(config)
        concat_dim = self._define_concat_dim(self.d_model)
        ### Define embeddings such that all input components have same dim
        # IMAGE
        if self.use_image:
            if self.img_pretrained == 'symbolic':
                self.emb_index = Embedding(
                    N_CLIPARTS+1, self.d_model, padding_idx=N_CLIPARTS)
                self.emb_size = Embedding(
                    N_SIZES+1, self.d_model, padding_idx=N_SIZES)
                self.emb_flip = Embedding(
                    N_FLIPS+1, self.d_model, padding_idx=N_FLIPS)
                self.emb_x = nn.Linear(MAX_CLIPARTS, self.d_model)
                self.emb_y = nn.Linear(MAX_CLIPARTS, self.d_model)
            else:
                self.emb_img = nn.Linear(self.img_input_dim, self.d_model)
        # GALLERY
        if self.use_gallery:
            self.emb_gallery = Embedding(
                N_CLIPARTS+1, self.d_model, padding_idx=N_CLIPARTS)
        # UTTERANCE
        self.emb_utt = nn.Linear(self.utterance_input_dim, self.d_model)
        # CONTEXT
        if self.use_context:
            self.emb_context = nn.Linear(self.context_input_dim, self.d_model)
        # CR FLAG
        if self.use_cr_flag:
            self.emb_flag = Embedding(N_FLAGS+1, self.d_model)
        # Transformer
        self.layer = nn.TransformerEncoderLayer(
            self.d_model, HEADS, batch_first=True, dropout=self.dropout)
        self.transform = nn.TransformerEncoder(self.layer, LAYERS)
        # Classifier
        self.fc = DeeperClassifier(concat_dim, self.output_dim, self.dropout)

    def _extract_config(self, config):
        self.use_context = not config.no_context
        self.use_image = not config.no_image
        self.use_gallery = config.use_gallery
        self.use_cr_flag = config.use_cr_flag
        self.no_msg = config.no_msg
        self.context_input_dim = config.context_input_dim
        self.utterance_input_dim = config.utterance_input_dim
        self.img_input_dim = config.img_input_dim
        self.img_pretrained = config.img_pretrained
        self.text_pretrained = config.text_pretrained
        self.dropout = config.dropout
        self.cur_device = config.device
        self.d_model = config.univ_embedding_dim

        self.output_dim = config.n_labels if config.no_regression else 1

    def _define_concat_dim(self, d_model):
        dim = 0
        if not self.no_msg:
            dim = d_model  
        if self.use_context:
            dim += d_model
        if self.use_image:
            if self.img_pretrained == 'symbolic':
                dim += (d_model * MAX_CLIPARTS * 3) + (d_model * 2)
            else:
                dim += d_model
        if self.use_gallery:
            dim += d_model * GALLERY_SIZE
        if self.use_cr_flag:
            dim += d_model
        if dim == 0:
            dim = self.last_msg_embedding_dim
        return dim

    @staticmethod
    def _split_symbolic(image):
        indexes = image[:, :, 0].long()
        position_x = image[:, :, 1].float()
        position_y = image[:, :, 2].float()
        sizes = image[:, :, 3].long()
        flips = image[:, :, 4].long()
        return indexes, position_x, position_y, sizes, flips

    def forward(self, context, last_msg, image, gallery, cr_flag):
        if self.no_msg:
            x_input =  torch.tensor([]).to(last_msg.device)
        else:
            x_input = self.emb_utt(last_msg).unsqueeze(1)
        if self.use_context:
            z_context = self.emb_context(context).unsqueeze(1)
            x_input = torch.cat([x_input, z_context], dim=1)
        if self.use_image:
            if self.img_pretrained == 'symbolic':
                elements = self._split_symbolic(image)
                indexes, position_x, position_y, sizes, flips = elements
                # embed all components
                z_indexes = self.emb_index(indexes)
                z_sizes = self.emb_size(sizes)
                z_flips = self.emb_flip(flips)
                z_position_x = self.emb_x(position_x).unsqueeze(1)
                z_position_y = self.emb_y(position_y).unsqueeze(1)
                z_aux = [z_position_x, z_position_y, z_indexes, z_sizes, z_flips]
                z_img = torch.cat(z_aux, dim=1)
            else:
                z_img = self.emb_img(image).unsqueeze(1)
            x_input = torch.cat([x_input, z_img], dim=1)
        if self.use_gallery:
            z_gallery = self.emb_gallery(gallery)
            x_input = torch.cat([x_input, z_gallery], dim=1)
        if self.use_cr_flag:
            z_flag = self.emb_flag(cr_flag.unsqueeze(1))
            x_input = torch.cat([x_input, z_flag], dim=1)
        
        if len(x_input) == 0:
            # if it gets here it's because we are using no input
            # if no_msg is true, last_msg will be a random vector
            # so for this sanity check we use this random input
            x_input = self.emb_utt(last_msg).unsqueeze(1)
            # x_input = torch.zeros(aux.shape)    

        h1 = self.transform(x_input)
        h2 = torch.flatten(h1, 1)
        y = self.fc(h2)
        return y


class BasicNetwork(nn.Module):
    def __init__(self, config):
        super(BasicNetwork, self).__init__()
        self._extract_config(config)
        concat_dim = self._define_concat_dim()
        # CLASSIFIER
        self.fc = DeeperClassifier(concat_dim, self.output_dim, self.dropout)

    def _extract_config(self, config):
        self.use_context = not config.no_context
        self.use_image = not config.no_image
        self.use_gallery = config.use_gallery
        self.use_cr_flag = config.use_cr_flag
        self.no_msg = config.no_msg
        self.context_input_dim = config.context_input_dim
        self.utterance_input_dim = config.utterance_input_dim
        self.img_input_dim = config.img_input_dim
        self.img_pretrained = config.img_pretrained
        self.text_pretrained = config.text_pretrained
        self.dropout = config.dropout
        self.cur_device = config.device
        self.output_dim = config.n_labels if config.no_regression else 1

    def _define_concat_dim(self):
        internal_dim = 0
        if not self.no_msg:
            internal_dim = self.utterance_input_dim
        if self.use_image:
            internal_dim += self.img_input_dim
        if self.use_context:
            internal_dim += self.context_input_dim
        if self.use_cr_flag:
            internal_dim += 1
        
        if internal_dim == 0:
            internal_dim = self.utterance_input_dim
        return internal_dim

    def forward(self, context, last_msg, image, gallery, cr_flag):
        if self.no_msg:
            x_input = torch.tensor([]).to(last_msg.device)
        else:
            x_input = last_msg
        if self.use_image:
            x_input = torch.cat([x_input, image], dim=1)
        #if self.use_gallery:
        #    encoded_gallery = self.symb_encoder(gallery)
        #    x_input = torch.cat([x_input, encoded_gallery], dim=1)
        if self.use_context:
            x_input = torch.cat([x_input, context], dim=1)
        if self.use_cr_flag:
            x_input = torch.cat([x_input, cr_flag.unsqueeze(-1)], dim=1)  
        # when no input, we use a dummy input, for sanity check
        if len(x_input) == 0:
            # if it gets here it's because we are using no input
            # if no_msg is true, last_msg will be a random vector
            # so for this sanity check we use this random input
            x_input = self.msg_encoder(last_msg)
            # x_input = torch.zeros(aux.shape)        
        output = self.fc(x_input)
        return output






class CoreNetwork(nn.Module):
    """Model in the paper."""
    def __init__(self, config):
        super(CoreNetwork, self).__init__()
        self._extract_config(config)
        concat_dim = self._define_concat_dim()
        # IMAGE COMPONENT
        if self.use_image:
            self.img_encoder = self._define_img_encoder()
        if self.use_gallery:
            self.symb_encoder = self._define_gallery_encoder()
        # TEXT COMPONENTS
        self.context_encoder, self.msg_encoder = self._define_text_encoders()
        # CLASSIFIER
        self.fc = DeeperClassifier(concat_dim, self.output_dim, self.dropout)

        #self.apply(self._init_weights)

    def _extract_config(self, config):
        self.use_context = not config.no_context
        self.use_image = not config.no_image
        self.use_gallery = config.use_gallery
        self.use_cr_flag = config.use_cr_flag
        self.no_msg = config.no_msg
        self.context_input_dim = config.context_input_dim
        self.utterance_input_dim = config.utterance_input_dim
        self.img_input_dim = config.img_input_dim
        self.img_embedding_dim = config.img_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.last_msg_embedding_dim = config.last_msg_embedding_dim
        self.gallery_embedding_dim = config.gallery_embedding_dim
        self.img_pretrained = config.img_pretrained
        self.text_pretrained = config.text_pretrained
        self.dropout = config.dropout
        self.cur_device = config.device

        self.output_dim = config.n_labels if config.no_regression else 1

    def _define_text_encoders(self):
        msg_encoder = TextEncoder(self.utterance_input_dim,
                                  self.last_msg_embedding_dim)
        context_encoder = None
        if self.use_context:
            context_encoder = TextEncoder(self.context_input_dim,
                                          self.context_embedding_dim)
        return context_encoder, msg_encoder

    def _define_img_encoder(self):
        if self.img_pretrained == 'symbolic':
            # TODO: pass dim as parameter
            img_enc = SymbolicImgTEncoder(128, self.img_embedding_dim)
        else:
            img_enc = ImageEncoder(self.img_input_dim, self.img_embedding_dim)
        return img_enc

    def _define_gallery_encoder(self):
        # TODO: pass dim as parameter
        gallery_enc = GalleryTEncoder(128, self.gallery_embedding_dim)
        return gallery_enc

    def _define_concat_dim(self):
        internal_dim = 0
        if not self.no_msg:
            internal_dim = self.last_msg_embedding_dim
        if self.use_image:
            internal_dim += self.img_embedding_dim
        if self.use_gallery:
            internal_dim += self.gallery_embedding_dim
        if self.use_context:
            internal_dim += self.context_embedding_dim
        if self.use_cr_flag:
            internal_dim += 1
        
        if internal_dim == 0:
            internal_dim = self.last_msg_embedding_dim
        return internal_dim

    def forward(self, context, last_msg, image, gallery, cr_flag):
        if self.no_msg:
            x_input = torch.tensor([]).to(last_msg.device)
        else:
            x_input = self.msg_encoder(last_msg)
        if self.use_image:
            encoded_image = self.img_encoder(image)
            x_input = torch.cat([x_input, encoded_image], dim=1)
        if self.use_gallery:
            encoded_gallery = self.symb_encoder(gallery)
            x_input = torch.cat([x_input, encoded_gallery], dim=1)
        if self.use_context:
            encoded_context = self.context_encoder(context)
            x_input = torch.cat([x_input, encoded_context], dim=1)
        if self.use_cr_flag:
            x_input = torch.cat([x_input, cr_flag.unsqueeze(-1)], dim=1)  
        # when no input, we use a dummy input, for sanity check
        if len(x_input) == 0:
            # if it gets here it's because we are using no input
            # if no_msg is true, last_msg will be a random vector
            # so for this sanity check we use this random input
            x_input = self.msg_encoder(last_msg)
            # x_input = torch.rand_like(aux)
            # x_input = torch.zeros(aux.shape)        
        output = self.fc(x_input)
        return output

    def _init_weights(self, module):
        # https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if isinstance(module, nn.Linear):
            #module.weight.data.normal_(mean=0.0, std=1.0)
            #torch.nn.init.ones_(module.weight)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                #module.bias.data.zero_()
                torch.nn.init.ones_(module.bias)
                #torch.nn.init.xavier_uniform_(module.bias)


class Classifier(nn.Module):
    """Simple linear classifier."""
    def __init__(self, concat_dim, output_dim, dropout):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(concat_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DeeperClassifier(nn.Module):
    """Linear classifier with two layers."""
    def __init__(self, concat_dim, output_dim, dropout):
        super(DeeperClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(concat_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ImageEncoder(nn.Module):
    """Linear classifier for the image."""
    def __init__(self, img_input_dim, img_embedding_dim):
        super(ImageEncoder, self).__init__()
        self.encoder = nn.Linear(img_input_dim, img_embedding_dim)

    def forward(self, image):
        return self.encoder(image)


class TextEncoder(nn.Module):
    """Linear classifier for the texts."""
    def __init__(self, text_input_dim, text_embedding_dim):
        super(TextEncoder, self).__init__()
        self.encoder = nn.Linear(text_input_dim, text_embedding_dim)

    def forward(self, text):
        return self.encoder(text)


class SymbolicImgTEncoder(nn.Module):
    """Transformer encoder for the symbolic image representation."""
    def __init__(self, d_model, gallery_embedding_dim):
        super(SymbolicImgTEncoder, self).__init__()
        self.emb_index = Embedding(N_CLIPARTS+1, d_model, padding_idx=N_CLIPARTS)
        self.emb_size = Embedding(N_SIZES+1, d_model, padding_idx=N_SIZES)
        self.emb_flip = Embedding(N_FLIPS+1, d_model, padding_idx=N_FLIPS)
        self.emb_x = nn.Linear(MAX_CLIPARTS, d_model)
        self.emb_y = nn.Linear(MAX_CLIPARTS, d_model)

        self.layer = nn.TransformerEncoderLayer(
            d_model, HEADS, batch_first=True)
        self.transform = nn.TransformerEncoder(self.layer, LAYERS)

        dim = d_model*MAX_CLIPARTS*3 + d_model*2
        self.fc = nn.Linear(dim, gallery_embedding_dim)

    @staticmethod
    def _split_symbolic(image):
        indexes = image[:, :, 0].long()
        position_x = image[:, :, 1].float()
        position_y = image[:, :, 2].float()
        sizes = image[:, :, 3].long()
        flips = image[:, :, 4].long()
        return indexes, position_x, position_y, sizes, flips

    def forward(self, x):

        elements = self._split_symbolic(x)
        indexes, position_x, position_y, sizes, flips = elements

        z_indexes = self.emb_index(indexes)
        z_sizes = self.emb_size(sizes)
        z_flips = self.emb_flip(flips)
        z_position_x = self.emb_x(position_x).unsqueeze(1)
        z_position_y = self.emb_y(position_y).unsqueeze(1)
        zs = [z_position_x, z_position_y, z_indexes, z_sizes, z_flips]
        z = torch.cat(zs, dim=1)
        h = self.transform(z)
        h2 = torch.flatten(h, 1)
        y = self.fc(h2)
        return y


class GalleryEncoder(nn.Module):
    """Linear classifier for the gallery."""
    def __init__(self, gallery_input_dim, gallery_embedding_dim):
        super(GalleryEncoder, self).__init__()
        self.encoder = nn.Linear(gallery_input_dim, gallery_embedding_dim)

    def forward(self, image):
        return self.encoder(image)


class GalleryTEncoder(nn.Module):
    """Transformer encoder with classifier for the gallery."""
    def __init__(self, d_model, gallery_embedding_dim):
        super(GalleryTEncoder, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, 8, batch_first=True)
        self.encode = nn.Sequential(
            Embedding(N_CLIPARTS+1, d_model, padding_idx=N_CLIPARTS),
            nn.TransformerEncoder(self.layer, 1)
        )
        self.fc = nn.Linear(d_model*28, gallery_embedding_dim)

    def forward(self, x):
        output = self.encode(x)
        z = torch.flatten(output, 1)
        y = self.fc(z)
        return y
