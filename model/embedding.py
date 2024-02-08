import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class MyEmbedding(nn.Module):
    def __init__(self, config):
        super(MyEmbedding, self).__init__()
        self.config = config

        self.num_locations = config.Dataset.num_locations
        self.base_dim = config.Embedding.base_dim
        self.num_users = config.Dataset.num_users

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

    def forward(self, batch_data):
        location_x = batch_data['location_x']

        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=location_x.device))

        return loc_embedded, timeslot_embedded, user_embedded
