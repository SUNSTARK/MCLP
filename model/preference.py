from torch import nn


class UserNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserNet, self).__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim

        self.block = nn.Sequential(
            nn.Linear(self.topic_num, self.topic_num * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.topic_num * 2, self.topic_num),
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.topic_num),
            nn.Linear(self.topic_num, self.output_dim)
        )

    def forward(self, topic_vec):
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec

        return self.final(topic_vec)

