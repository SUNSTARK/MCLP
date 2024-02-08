import torch
from torch import nn


class ArrivalTime(nn.Module):
    def __init__(self, config):
        super(ArrivalTime, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.base_dim = config.Embedding.base_dim
        self.num_heads = 4
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = config.Dataset.num_users
        self.base_dim = config.Embedding.base_dim
        self.timeslot_num = 24

        if config.Model.at_type == 'attn':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim + self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

    def forward(self, timeslot_embedded, batch_data):
        user_x = batch_data['user']
        hour_x = batch_data['hour']
        batch_size, sequence_length = hour_x.shape
        auxiliary_y = batch_data['timeslot_y']
        hour_mask = batch_data['hour_mask'].view(batch_size * sequence_length, -1)
        if self.config.Model.at_type == 'truth':
            at_embedded = timeslot_embedded[auxiliary_y]
            return at_embedded
        if self.config.Model.at_type == 'attn':
            hour_x = hour_x.view(batch_size * sequence_length)
            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x]
            query = torch.cat([user_feature, time_feature], dim=-1)
            key = timeslot_embedded
            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask == 1, float('-inf'))
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            return self.unify_heads(head_outputs)

        if self.config.Model.at_type == 'static':
            time_trans_prob_mat = batch_data['prob_matrix_time_individual']
            at_embedded_user = torch.matmul(time_trans_prob_mat, timeslot_embedded)
            batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(hour_x)
            at_embedded = at_embedded_user[batch_indices, hour_x, :]
            return at_embedded
