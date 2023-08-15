import torch
from torch import Tensor
import torch.nn as nn
import math
import random
from d2l import torch as d2l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_mask(x: Tensor, valid_len: Tensor, value: int = 0):
    '''
    :param x: (batch_size, seq_len)
    :param valid_len: (batch_size)
    :param value: int, mask number
    :return: (batch_size, seq_len)
    example:
        X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        print(sequence_mask(X, torch.tensor([1, 2])))
        tensor([[1, 0, 0], [4, 5, 0]])
    '''
    max_len = x.size()[1]
    mask = torch.arange((max_len), dtype=torch.float32, device=device)  # (max_len)
    mask = mask.unsqueeze(0)  # (batch_size, max_len)
    valid_len = valid_len.unsqueeze(1)  # (batch_size, max_len)
    mask = mask < valid_len  # (batch_size, max_len)
    x[~mask] = value
    return x


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred: Tensor, label: Tensor, valid_len: Tensor) -> Tensor:
        '''
        :param pred: (batch_size, num_steps, vacab_size)
        :param label: (batch_size, num_steps)
        :return: (batch_size,)
        '''
        weights = torch.ones(label)
        weights = sequence_mask(weights, valid_len, 0)
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=-1)
        return weighted_loss


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout)

    def forward(self, x: Tensor, *args):
        x = self.embeddings(x)  # (batch_size, num_steps, embed_size)
        x = x.permute(1, 0, 2)  # (num_steps, batch_size, embed_size)
        output, state = self.rnn(x)
        # output: (num_steps, batch_size, num_hiddens), state:(num_layers, batch_size, num_hiddens)
        return output, state


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size + num_hiddens, hidden_size=num_hiddens, num_layers=num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x: Tensor, state):
        x = self.embedding(x).permute(1, 0, 2) # (num_steps, batch_size, n_hiddens)
        context = state[-1].repeat(x.size()[0], 1, 1) #(num_steps, batch_size, n_hiddens)
        x_and_context = torch.cat((x, context), dim=2) #(num_steps, batch_size, n_hiddens + em)
        output, state = self.rnn(x_and_context, state)
        # output: (num_steps, batch_size, num_hiddens ), state:(num_layers, batch_size, num_hiddens )
        output = self.dense(output).permute(1, 0, 2) #  (batch_size, num_steps, num_hiddens)
        return output, state

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def train(net, data_iter, lr, num_epochs, tgt_vocab):
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
