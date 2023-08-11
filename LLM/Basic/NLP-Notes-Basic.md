# 基础
## 
### TF
“TF”通常指的是“词频”（term frequency）。它是一种用于衡量一个词在文本中出现频率的方法。
具体来说，词频是指一个给定单词在文本中出现的次数。在文本分析和信息检索中，词频是一种常用的特征表示方法，
可以用于诸如文本分类、信息检索、聚类等任务。通常情况下，词频是与文本中的每个单词一起计算的，
以便生成一个文本的向量表示形式，也被称为词袋模型（bag of words model）。
TF的计算方法很简单，就是将一个特定的词在文本中出现的次数除以文本总词数。例如，在一篇包含100个单词的文章中，
单词“apple”出现了10次，那么它的TF值就是10/100 = 0.1。

### TF-IDF Representation
考虑一组专利文件。您可能希望它们中的大多数都有诸如claim、system、method、procedure等单词，并且经常重复多次。
TF表示对更频繁的单词进行加权。然而，像“claim”这样的常用词并不能增加我们对具体专利的理解。相反，
如果“tetrafluoroethylene”这样罕见的词出现的频率较低，但很可能表明专利文件的性质，我们希望在我们的表述中赋予它更大的权重。
反文档频率(IDF)是一种启发式算法，可以精确地做到这一点。
IDF表示惩罚常见的符号，并奖励向量表示中的罕见符号。


## active function
### sigmoid
torch.sigmoid(input)
### tanh
### ReLU
 f(x)=max(0,x) 
ReLU单元所做的就是将负值裁剪为零
relu = torch.nn.ReLU()
torch.relu
x = torch.range(-5., 5., 0.1)
relu(x)
### Leaky ReLU
ReLU的裁剪效果有助于消除梯度问题，随着时间的推移，网络中的某些输出可能会变成零，再也不会恢复。
这就是所谓的“dying ReLU”问题。为了减轻这种影响，提出了Leaky ReLU或 Parametric ReLU (PReLU)等变体，
其中泄漏系数a是一个可学习参数: f(x)=max(x,ax)
import torch
import matplotlib.pyplot as plt
prelu = torch.nn.PReLU(num_parameters=1)
x = torch.arange(-5., 5., 0.1)
y = prelu(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

### softmax
softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)
print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))

## Loss Function
### MSE就是预测值与目标值之差的平方的平均值
```pycon
import torch.nn as nn
mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)
# tensor(1.3748, grad_fn=<MseLossBackward0>)
```
## categorical cross-entropy loss
### 通常用于多类分类设置，其中输出被解释为类隶属度概率的预测
```pycon
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
```

## Binary Cross-Entropy
### 
pass 


#### repeat_interleave
```pycon
import torch
# 创建一个包含[1, 2, 3]的张量
x = torch.tensor([1, 2, 3])
# 使用repeat_interleave方法将每个元素重复扩展3次
y = x.repeat_interleave(3)
print(y)
# tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
```

#### repeat
torch.repeat方法用于沿指定的维度对张量进行复制和扩展。它会在指定的维度上重复复制张量的元素
```pycon
import torch
# 创建一个包含[[1, 2], [3, 4]]的2维张量
x = torch.tensor([[1, 2], [3, 4]])
# 使用torch.repeat方法在维度0上重复复制2次
y = x.repeat(2, 1)
print(y)
# tensor([[1, 2],[3, 4],[1, 2],[3, 4]])
```

#### torch.bmm
用于执行批量矩阵乘法（batch matrix multiplication）操作。它接受三个输入张量，进行矩阵相乘操作，并返回结果张量。
torch.bmm(mat1, mat2, *, out=None) -> Tensor
参数：
mat1：形状为(batch_size, n, m)的张量，表示批量的左操作数矩阵。
mat2：形状为(batch_size, m, p)的张量，表示批量的右操作数矩阵。
out（可选）：输出张量，用于指定计算结果的存储位置。
返回值：
形状为(batch_size, n, p)的张量，表示批量矩阵乘法的结果。
```pycon
import torch
# 创建两个批量矩阵
mat1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
mat2 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# 执行批量矩阵乘法
result = torch.bmm(mat1, mat2)
print(result)
'''
tensor([[[  7,  10],
         [ 15,  22]],
        [[ 67,  78],
         [ 91, 106]]])
'''
```

#### torch.unsqueeze
用于在指定的维度上插入一个长度为1的维度。它可以用来改变张量的形状，增加维度的数量。这在进行某些操作或与其他张量进行广播操作时非常有用。
参数：
input：输入张量。
dim：要插入新维度的位置。
返回值：
插入了新维度的张量。
```pycon
import torch
# 创建一个形状为(3,)的张量
x = torch.tensor([1, 2, 3])
# 在维度0上插入一个新维度
y = torch.unsqueeze(x, 0)
print(y.shape)
print(y)
'''
torch.Size([1, 3])
tensor([[1, 2, 3]])
'''
```

#### torch.transpose 
`torch.transpose` 是 PyTorch 中用于对张量进行转置操作的函数。它可以改变张量的维度顺序，实现维度的转置。

函数签名如下：
```python
torch.transpose(input, dim0, dim1) -> Tensor
```

参数说明：
- `input`：要转置的输入张量。
- `dim0`：要交换的第一个维度。
- `dim1`：要交换的第二个维度。

返回值：
- 返回一个新的张量，其维度顺序是在输入张量的基础上进行了转置。

下面是一个示例：

```python
import torch

# 创建一个大小为(2, 3)的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 对张量进行转置
y = torch.transpose(x, 0, 1)

print("原始张量：")
print(x)

print("转置后的张量：")
print(y)
```
输出结果：
```
原始张量：
tensor([[1, 2, 3],
        [4, 5, 6]])   
转置后的张量：
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```
在上述示例中，我们创建了一个大小为(2, 3)的张量`x`。然后，使用`torch.transpose`函数将第一个维度和第二个维度进行转置，得到了转置后的张量`y`。可以看到，原始张量中的行变为了列，列变为了行。
你可以根据需要在`torch.transpose`函数中指定要交换的维度，从而进行不同的转置操作。

### torch.cat
`torch.cat` 是 PyTorch 中用于连接（拼接）张量的函数。它可以将多个张量沿指定的维度进行拼接。

函数签名如下：
```python
torch.cat(tensors, dim=0, *, out=None) -> Tensor
```
参数说明：
- `tensors`：要拼接的张量序列，可以是一个张量列表或元组。
- `dim`：指定拼接的维度，默认为 0，表示沿着第一个维度进行拼接。
- `out`：可选参数，输出张量。
示例用法：
```python
import torch
# 创建两个张量
x1 = torch.tensor([[1, 2], [3, 4]])
x2 = torch.tensor([[5, 6]])
# 沿着第一个维度拼接
result = torch.cat((x1, x2), dim=0)
print(result)
```
输出：
```
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```
在上述示例中，我们创建了两个张量 `x1` 和 `x2`，然后使用 `torch.cat` 函数将它们沿着第一个维度拼接成一个新的张量 `result`。
你可以根据实际需要选择要拼接的张量和拼接的维度。注意，要拼接的张量在指定维度上的大小必须是一致的，除了沿着该维度进行拼接外，其他维度的大小应该是相同的。


#### torch.max
`torch.max` 是一个 PyTorch 函数，用于在张量中找到指定维度上的最大值。它返回指定维度上的最大值以及对应的索引。

函数签名如下：
```python
torch.max(input, dim=None, keepdim=False, *, out=None) -> (Tensor, LongTensor)
```

参数说明：
- `input`：输入张量。
- `dim`：指定的维度，用于在该维度上寻找最大值。如果未指定，则在整个张量上寻找最大值。
- `keepdim`：一个布尔值，指示是否保持输出张量的维度和输入张量相同。
- `out`：可选参数，输出张量以保存结果。
返回值：
- 一个包含最大值的张量。
- 一个包含最大值索引的长整型张量。
示例用法：
```python
import torch
# 创建一个张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 在指定维度上寻找最大值
max_values, max_indices = torch.max(x, dim=1)
print("Max Values:", max_values)
print("Max Indices:", max_indices)
```
输出：
```
Max Values: tensor([3, 6])
Max Indices: tensor([2, 2])
```
在上述示例中，我们创建了一个二维张量 `x`。然后，我们使用 `torch.max` 函数在第一个维度上寻找最大值。返回的 `max_values` 张量包含了每行的最大值，而 `max_indices` 张量包含了每行最大值的索引。
你可以根据需要选择要在哪个维度上寻找最大值，以及是否保持输出张量的维度与输入张量相同。

#### getattr

#### torch one_hot
```pycon
import torch
import torch.nn.functional as F
# Define a tensor with integer values
tensor = torch.tensor([1, 2, 0, 1, 2])
# Convert the tensor to one-hot representation
one_hot = F.one_hot(tensor)
print(one_hot)
```
tensor([[0, 1, 0],[0, 0, 1],[1, 0, 0],[0, 1, 0],[0, 0, 1]])

#### torch contiguous()
```pycon
import torch
# 创建一个非连续的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.t()  # 转置操作，导致存储方式不连续
print(y.is_contiguous())  # False，y张量的存储方式不连续
# 使用contiguous()方法使张量连续
y = y.contiguous()
print(y.is_contiguous())  # True，y张量的存储方式已变为连续
```

#### CROSSENTROPYLOSS
```pycon
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
```

#### optim.lr_scheduler.StepLR
在上述例子中，优化器使用随机梯度下降（SGD）算法，初始学习率为 0.1。学习率调度器每经过一个训练步骤即一个 epoch，学习率会乘以 0.95 进行衰减。因此，第一个 epoch 结束后，学习率变为 0.1 * 0.95 = 0.095，第二个 epoch 结束后，学习率变为 0.095 * 0.95 = 0.09025，以此类推。
通过使用学习率调度器，可以在训练过程中逐渐降低学习率，从而使模型在接近最优解时更加稳定地收敛。
##### 如何选择步长
步长的选择应该基于数据集的大小和训练的速度。例如，对于较大的数据集，可以选择较大的步长，而对于较小的数据集，可以选择较小的步长。通常，步长的值在几个 epochs 到几十个 epochs 之间。
##### 选择衰减因子
衰减因子决定学习率在每次更新时的缩放比例。较小的衰减因子会使学习率缓慢地衰减，而较大的衰减因子会使学习率迅速地衰减。一般来说，衰减因子的选择应该基于模型的复杂性和数据集的特点。对于复杂的模型和大型数据集，较小的衰减因子可能更适合，而对于简单的模型和小型数据集，较大的衰减因子可能更适合。一般来说，衰减因子的值在0到1之间。
```pycon
import torch
from torch import optim
from torch.optim import lr_scheduler
# 创建一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 创建一个学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
# 模拟训练过程
for epoch in range(10):
    # 在每个训练步骤之前更新学习率
    scheduler.step()
    # 进行模型训练
    train(model, optimizer)
```
### 词表
词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。 现在，让我们构建一个字典，通常也叫做词表（vocabulary）， 
用来将字符串类型的词元映射到从 开始的数字索引中。 我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称
之为语料（corpus）。 然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。 另外，
语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：
填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。

### 困惑度perplexity
困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。 我们看看一些案例。
在最好的情况下，模型总是完美地估计标签词元的概率为1。 在这种情况下，模型的困惑度为1。
在最坏的情况下，模型总是预测标签词元的概率为0。 在这种情况下，困惑度是正无穷大。
在基线上，该模型的预测是词表的所有可用词元上的均匀分布。 在这种情况下，困惑度等于词表中唯一词元的数量。 
事实上，如果我们在没有任何压缩的情况下存储序列， 这将是我们能做的最好的编码方式。 因此，这种方式提供了一个重要的上限， 
而任何实际模型都必须超越这个上限。

### BLEU（bilingual evaluation understudy)
最先是用于评估机器翻译的结果， 但现在它已经被广泛用于测量许多应用的输出序列的质量。 原则上说，对于预测序列中的任意
元语法（n-grams），BLEU的评估都是这个元语法是否出现在标签序列中。

### 束搜索（beam search）
束搜索（beam search）是贪心搜索的一个改进版本。它有一个超参数，名为束宽（beam size)
在时间步，我们选择具有最高条件概率的个词元。这个词元将分别是个候选输出序列的第一个词元。在随后的每个时间步，基于上一时间步的个候选输出序列， 
我们将继续从个可能的选择中挑出具有最高条件概率的个候选输出序列。

#### 为何独热向量是一个糟糕的选择
我们使用独热向量来表示词（字符就是单词）。假设词典中不同词的数量（词典大小）为 ，每个词对应一个从 到 的不同整数（索引）。
为了得到索引为 的任意词的独热向量表示，我们创建了一个全为0的长度为 的向量，并将位置 的元素设置为1。这样，每个词都被表示为
一个长度为 的向量，可以直接由神经网络使用。 虽然独热向量很容易构建，但它们通常不是一个好的选择。一个主要原因是独热向量不能
准确表达不同词之间的相似度，比如我们经常使用的“余弦相似度”。

### 负采样
在自然语言处理（NLP）领域中，负采样（Negative Sampling）是一种训练词嵌入（Word Embedding）模型的技术。它的目的是通过选择一些负样本，来提高模型的效率和性能。 
简单来说，负采样是通过从大量的可能负样本中随机选择一小部分作为训练样本，而不使用所有的负样本。通常，在训练词嵌入模型时，我们希望模型能够将目标词与其上下文中的其他
词区分开来，即将正样本（目标词与上下文词配对）的相似度提高，将负样本（目标词与随机选择的其他词配对）的相似度降低。
```pycon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义数据集类
class WordContextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 定义负采样模型
class NegativeSamplingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(NegativeSamplingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target, context):
        target_embed = self.embedding(target)
        context_embed = self.embedding(context)
        return target_embed, context_embed
    
# 设置超参数
vocab_size = 10000
embedding_dim = 100
learning_rate = 0.001
num_epochs = 10
batch_size = 64
# 构建示例数据集
data = [(1, 2), (3, 4), (5, 6), ...]  # (target, context) pairs
dataset = WordContextDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 初始化模型和优化器
model = NegativeSamplingModel(vocab_size, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
for epoch in range(num_epochs):
    for batch in dataloader:
        targets, contexts = batch
        optimizer.zero_grad()
        target_embed, context_embed = model(targets, contexts)
        loss = criterion(target_embed, context_embed)
        loss.backward()
        optimizer.step()
# 使用训练好的模型进行预测
target = torch.tensor([1])
context = torch.tensor([2])
target_embed, context_embed = model(target, context)
```
### 下采样
文本数据通常有“the”“a”和“in”等高频词：它们在非常大的语料库中甚至可能出现数十亿次。然而，这些词经常在上下文窗口中与许多不同的词
共同出现，提供的有用信息很少。例如，考虑上下文窗口中的词“chip”：直观地说，它与低频单词“intel”的共现比与高频单词“a”的共现在训练
中更有用。此外，大量（高频）单词的训练速度很慢。因此，当训练词嵌入模型时，可以对高频单词进行下采样 (Mikolov et al., 2013)。
具体地说，数据集中的每个词将有概率地被丢弃。词频越大求起的概率就越大

### 