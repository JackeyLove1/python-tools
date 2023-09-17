from d2l import torch as d2l
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size, num_steps = 2, 5
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
print("src len:", len(src_vocab))
print("tgt len:", len(tgt_vocab))
encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                             dropout)
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers)
net = d2l.EncoderDecoder(encoder, decoder)
loss = d2l.MaskedSoftmaxCELoss()
for batch in train_iter:
    X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
    print("X:", X)
    print("X_valid_len:", X_valid_len)
    print("Y:", Y)
    print("Y_valid_len:", Y_valid_len)
    bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                       device=device).reshape(-1, 1)
    print("bos:", bos)
    dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
    print("dec_input:", dec_input)
    Y_hat, _ = net(X, dec_input, X_valid_len)
    print("Y_hat:", Y_hat)
    print("Y_hat.size:", Y_hat.size())
    l = loss(Y_hat, Y, Y_valid_len)
    l.sum().backward()  # 损失函数的标量进行“反向传播”
    d2l.grad_clipping(net, 1)
    num_tokens = Y_valid_len.sum()
    print("num_tokens:", num_tokens)
    break
