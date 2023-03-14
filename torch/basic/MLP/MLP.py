import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activation(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs, dim=1)
        return probs


net = MLP(input_dim=4, hidden_dim=5, num_class=2)
inputs = torch.rand((3,4))
probs = net(inputs)
print(probs)