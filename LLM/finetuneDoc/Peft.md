# PEFT
## 目前支持的一些高效微调方法如下：
LoRA: LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
Prefix Tuning: Prefix-Tuning: Optimizing Continuous Prompts for Generation 和 P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
P-Tuning: GPT Understands, Too
Prompt Tuning: The Power of Scale for Parameter-Efficient Prompt Tuning
AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
(IA)3(IA)^3(IA)3 : Infused Adapter by Inhibiting and Amplifying Inner Activations

### P-Tuning 
P-Tuning（论文：GPT Understands, Too），该方法将 Prompt 转换为可以学习的 Embedding 层，
并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。
相比Prefix Tuning，P-Tuning加入的可微的virtual token，但仅限于输入层，没有在每一层都加；
另外，virtual token的位置也不一定是前缀，插入的位置是可选的。这里的出发点实际是把传统人工设计模版
中的真实token替换成可微的virtual token。
#### v2
之前的Prompt Tuning和P-Tuning等方法存在两个主要的问题：
第一，缺乏模型参数规模和任务通用性。

缺乏规模通用性：Prompt Tuning论文中表明当模型规模超过100亿个参数时，提示优化可以与全量微调相媲美。但是对于那些较小的模型（从100M到1B），提示优化和全量微调的表现有很大差异，这大大限制了提示优化的适用性。
缺乏任务普遍性：尽管Prompt Tuning和P-tuning在一些 NLU 基准测试中表现出优势，但提示调优对硬序列标记任务（即序列标注）的有效性尚未得到验证。

第二，缺少深度提示优化，在Prompt Tuning和P-tuning中，连续提示只被插入transformer第一层的输入embedding序列中，在接下来的transformer层中，插入连续提示的位置的embedding是由之前的transformer层计算出来的，这可能导致两个可能的优化挑战。

由于序列长度的限制，可调参数的数量是有限的。
输入embedding对模型预测只有相对间接的影响。

考虑到这些问题，作者提出了P-Tuning v2，它利用深度提示优化（如：Prefix Tuning），对Prompt Tuning和P-Tuning进行改进，作为一个跨规模和NLU任务的通用解决方案。

Prefix Tuning（论文：Prefix-Tuning: Optimizing Continuous Prompts for Generation），在输入token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而PLM中的其他部分参数固定。
针对不同的模型结构，需要构造不同的Prefix。

针对自回归架构模型：在句子前面添加前缀，得到 z = [PREFIX; x; y]，合适的上文能够在固定 LM 的情况下去引导生成下文（比如：GPT3的上下文学习）。
针对编码器-解码器架构模型：Encoder和Decoder都增加了前缀，得到 z = [PREFIX; x; PREFIX0; y]。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。

### Prompt Tuning
Prompt Tuning（论文：The Power of Scale for Parameter-Efficient Prompt Tuning），
该方法可以看作是 Prefix Tuning 的简化版本，它给每个任务定义了自己的Prompt，然后拼接到数据上作为输入，
但只在输入层加入prompt tokens，并且不需要加入 MLP 进行调整来解决难训练的问题。

### Perfix Tuning
Prefix Tuning（论文：Prefix-Tuning: Optimizing Continuous Prompts for Generation），在输入token之前构造一段任务相关的virtual tokens作为Prefix；然后，在训练的时候只更新Prefix部分的参数，而 PLM 中的其他部分参数固定。
针对不同的模型结构，需要构造不同的 Prefix。
针对自回归架构模型：在句子前面添加前缀，得到 z = [PREFIX; x; y]，合适的上文能够在固定 LM 的情况下去引导生成下文（比如：GPT3的上下文学习）。
针对编码器-解码器架构模型：Encoder和Decoder都增加了前缀，得到 z = [PREFIX; x; PREFIX0; y]。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。
除此之外，通过消融实验证实，只调整embedding层的表现力不够，将导致性能显著下降，因此，在每层都加了prompt的参数，改动较大。

### AdaLoRA
在NLP领域，对于下游任务进行大型预训练语言模型的微调已经成为一种重要的做法。一般而言，我们会采用对原有的预训练模型进行全量微调的方法来适配下游任务，但这种方法存在两个问题。

训练阶段。对于预训练模型进行微调的时候，为了更新权重参数，需要大量的显存来存储参数的梯度和优化器信息，在当今预训练模型的参数变得越来越大的情况下，针对下游任务微调门槛变得越来越高。
推理阶段。由于我们训练的时候是对于模型参数进行全量的更新，所以多个下游任务需要为每个任务维护一个大型模型的独立副本，这样就导致我们在实际应用的时候浪费了不必要的存储。

为了解决这些问题，研究者提出了两个主要研究方向，以减少微调参数的数量，同时保持甚至提高预训练语言模型的性能。

方向一：添加小型网络模块：将小型网络模块添加到PLMs中，保持基础模型保持不变的情况下仅针对每个任务微调这些模块
，可以用于所有任务。这样，只需引入和更新少量任务特定的参数，就可以适配下游的任务，大大提高了预训练模型的实用
性。如：Adapter tuning、Prefix tuning、Prompt Tuning等，这类方法虽然大大减少了内存消耗。但是这些方法
存在一些问题，比如：Adapter tuning引入了推理延时；Prefix tuning或Prompt tuning直接优化Prefix和Prompt
是非单调的，比较难收敛，并且消耗了输入的token。
方向二：下游任务增量更新：对预训练权重的增量更新进行建模，而无需修改模型架构，即W=W0+△W。
比如：Diff pruning、LoRA等， 此类方法可以达到与完全微调几乎相当的性能，但是也存在一些问题，
比如：Diff pruning需要底层实现来加速非结构化稀疏矩阵的计算，不能直接使用现有的框架，训练过程中需要存储完整
的∆W矩阵，相比于全量微调并没有降低计算成本。 LoRA则需要预先指定每个增量矩阵的本征秩 r 相同，忽略了在微调
预训练模型时，权重矩阵的重要性在不同模块和层之间存在显著差异，并且只训练了Attention，没有训练FFN，事实上FFN更重要。

 
基于以上问题进行总结：

第一，我们不能预先指定矩阵的秩，需要动态更新增量矩阵的R，因为权重矩阵的重要性在不同模块和层之间存在显著差异。
第二，需要找到更加重要的矩阵，分配更多的参数，裁剪不重要的矩阵。找到重要的矩阵，可以提升模型效果；而裁剪不重要的矩阵，可以降低参数计算量，降低模型效果差的风险。

AdaLoRA（论文：ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING），是对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵。具体做法如下：

调整增量矩分配。AdaLoRA将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。
以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。

在训练损失中添加了额外的惩罚项，以规范奇异矩阵P和Q的正交性，从而避免SVD的大量计算并稳定训练。

通过实验证明，AdaLoRA 实现了在所有预算、所有数据集上与现有方法相比，性能更好或相当的水平。 例如，当参数预算为 0.3M 时，AdaLoRA 在RTE数据集上，比表现最佳的基线（Baseline）高 1.8%。



### QLoRA
QLoRA（论文： QLORA: Efficient Finetuning of Quantized LLMs），使用一种新颖的高精度技术将预训
练模型量化为 4 bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调。
QLORA 有一种低精度存储数据类型（4 bit），还有一种计算数据类型（BFloat16）。实际上，这意味着无论何时使用
QLoRA 权重张量，我们都会将张量反量化为 BFloat16，然后执行 16 位矩阵乘法。QLoRA提出了两种技术实现高保真
4 bit微调——4 bit NormalFloat(NF4) 量化和双量化。此外，还引入了分页优化器，以防止梯度检查点期间的
内存峰值，从而导致内存不足的错误，这些错误在过去使得大型模型难以在单台机器上进行微调。具体说明如下：
4bit NormalFloat（NF4）：对于正态分布权重而言，一种信息理论上最优的新数据类型，该数据类型对正态分布
数据产生比 4 bit整数和 4bit 浮点数更好的实证结果。
双量化：对第一次量化后的那些常量再进行一次量化，减少存储空间。
分页优化器：使用NVIDIA统一内存特性，该特性可以在在GPU偶尔OOM的情况下，进行CPU和GPU之间自动分页到分页的
传输，以实现无错误的 GPU 处理。该功能的工作方式类似于 CPU 内存和磁盘之间的常规内存分页。使用此功能为优化器
状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新步骤需要时将其
加载回 GPU 内存。


### IA3
IA3（论文：Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning），
通过学习向量来对激活层加权进行缩放，从而获得更强的性能，同时仅引入相对少量的新参数，如下图左边所示，它的诞生背景是为了改进 LoRA。

与 LoRA 类似，IA3 具有许多相同的优点：

IA3 通过大幅减少可训练参数的数量，使微调更加高效。对于 T0 模型，使用 IA3 只有大约 0.01% 的可训练参数，而使用 LoRA 有 > 0.1% 的可训练参数。
原始的预训练权重保持冻结状态，这意味着您可以拥有多个轻量级、便携式 IA3 模型，用于在其之上构建的各种下游任务。
使用 IA3 微调的模型的性能与完全微调的模型的性能相当。
IA3 不会增加任何推理延迟，因为适配器（adapter）权重可以与基础模型合并。

原则上，IA3 可以应用于神经网络中权重矩阵的任何子集，以减少可训练参数的数量。 根据作者的实现，IA3 权重被添加
到 Transformer 模型的 key， value 和 feedforward 层。 给定注入 IA3 参数的目标层，可训练参数的数量可以
根据权重矩阵的大小确定。
