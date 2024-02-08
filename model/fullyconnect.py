import torch.nn as nn


class MyFullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyFullyConnect, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim*2, input_dim),
            nn.Dropout(0.1),
        )

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)

        num_locations = output_dim
        self.linear_class1 = nn.Linear(input_dim, num_locations)

    def forward(self, out):
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)

        return self.linear_class1(out)
