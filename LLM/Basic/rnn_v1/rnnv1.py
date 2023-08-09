import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn_layer.hidden_size
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)


    def forward(self, input, state):
        X = F.one_hot(input.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) # Y -> (time_step , batch_size, num_hiddens)
        # Y is (time_step * batch_size, num_hiddens) - > (time_step * batch_size, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
