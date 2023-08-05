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
