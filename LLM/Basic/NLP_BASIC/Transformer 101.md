# Transformer 101
## 起源与发展
- GPT (the Generative Pretrained Transformer)；
- BERT (Bidirectional Encoder Representations from Transformers)。
通过将 Transformer 结构与无监督学习相结合，我们不再需要对每一个任务都从头开始训练模型，并且几乎在所有 NLP 任务上都远远超过先前的最强基准。
虽然新的 Transformer 模型层出不穷，它们采用不同的预训练目标在不同的数据集上进行训练，但是依然可以按模型结构将它们大致分为三类：
纯 Encoder 模型（例如 BERT），又称自编码 (auto-encoding) Transformer 模型；
纯 Decoder 模型（例如 GPT），又称自回归 (auto-regressive) Transformer 模型；
Encoder-Decoder 模型（例如 BART、T5），又称 Seq2Seq (sequence-to-sequence) Transformer 模型。

Transformer 模型本质上都是预训练语言模型，大都采用自监督学习 (Self-supervised learning) 的方式在大量生语料上进行训练，也就是说，训练这些 Transformer 模型完全不需要人工标注数据。
自监督学习是一种训练目标可以根据模型的输入自动计算的训练方法。
* 基于句子的前N个词来预测下一个词，因为输出依赖于过去和当前的输入，因此该任务被称为因果语言建模 (causal language modeling)；
* 基于上下文（周围的词语）来预测句子中被遮盖掉的词语 (masked word)，因此该任务被称为遮盖语言建模 (masked language modeling)。

即使用自己的任务语料对模型进行“二次训练”，通过微调参数使模型适用于新任务。

这种迁移学习的好处是：

预训练时模型很可能已经见过与我们任务类似的数据集，通过微调可以激发出模型在预训练过程中获得的知识，将基于海量数据获得的统计理解能力应用于我们的任务；
由于模型已经在大量数据上进行过预训练，微调时只需要很少的数据量就可以达到不错的性能；
换句话说，在自己任务上获得优秀性能所需的时间和计算成本都可以很小。

纯 Encoder 模型：适用于只需要理解输入语义的任务，例如句子分类、命名实体识别；
纯 Decoder 模型：适用于生成式任务，例如文本生成；
Encoder-Decoder 模型或 Seq2Seq 模型：适用于需要基于输入的生成式任务，例如翻译、摘要。

## 模型与分词器
### 分词器
由于神经网络模型不能直接处理文本，因此我们需要先将文本转换为数字，这个过程被称为编码 (Encoding)，其包含两个步骤：

使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens； 将所有的 token 映射到对应的 token ID。
#### 分词策略
按词切分 (Word-based)：这种策略的问题是会将文本中所有出现过的独立片段都作为不同的 token，从而产生巨大的词表。而实际上很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果给它们赋予不同的编号就无法表示出这种关联性。
当遇到不在词表中的词时，分词器会使用一个专门的[UNK]token 来表示它是 unknown 的。显然，如果分词结果中包含很多[UNK]就意味着丢失了很多文本信息，因此一个好的分词策略，应该尽可能不出现 unknown token。

按字符切分 (Character-based):
1. 按子词切分 (Subword)
高频词直接保留，低频词被切分为更有意义的子词。例如 “annoyingly” 是一个低频词，可以切分为 “annoying” 和 “ly”，这两个子词不仅出现频率更高，而且词义也得以保留。下图展示了对 “Let’s do tokenization!“ 按子词切分的结果：
bpe_subword 可以看到，“tokenization” 被切分为了 “token” 和 “ization”，不仅保留了语义，而且只用两个 token 就表示了一个长词。这种策略只用一个较小的词表就可以覆盖绝大部分文本，基本不会产生 unknown token。尤其对于土耳其语等黏着语，几乎所有的复杂长词都可以通过串联多个子词构成。