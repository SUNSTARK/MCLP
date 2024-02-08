import math

import torch
from torch import nn
from embedding import MyEmbedding, PositionalEncoding
from encoder import TransEncoder, LSTMEncoder
from fullyconnect import MyFullyConnect
from arrivaltime import ArrivalTime
from preference import UserNet


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.topic_num = config.Dataset.topic_num

        self.embedding_layer = MyEmbedding(config)

        if config.Encoder.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = TransEncoder(config)
        if config.Encoder.encoder_type == 'lstm':
            self.encoder = LSTMEncoder(config)

        fc_input_dim = self.base_dim + self.base_dim

        if config.Model.at_type != 'none':
            self.at_net = ArrivalTime(config)
            fc_input_dim += self.base_dim

        if self.topic_num > 0:
            self.user_net = UserNet(input_dim=self.topic_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim

        self.fc_layer = MyFullyConnect(input_dim=fc_input_dim,
                                       output_dim=config.Dataset.num_locations)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch_data):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc']
        batch_size, sequence_length = loc_x.shape

        loc_embedded, timeslot_embedded, user_embedded = self.embedding_layer(batch_data)
        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded

        if self.config.Encoder.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                       src_mask=future_mask)
        if self.config.Encoder.encoder_type == 'lstm':
            encoder_out = self.encoder(lt_embedded)
        combined = encoder_out + lt_embedded

        user_embedded = user_embedded[user_x]

        if self.config.Model.at_type != 'none':
            at_embedded = self.at_net(timeslot_embedded, batch_data)
            combined = torch.cat([combined, at_embedded], dim=-1)

        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded], dim=-1)

        if self.topic_num > 0:
            pre_embedded = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)

        out = self.fc_layer(combined.view(batch_size * sequence_length, combined.shape[2]))

        return out
