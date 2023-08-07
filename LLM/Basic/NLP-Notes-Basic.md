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