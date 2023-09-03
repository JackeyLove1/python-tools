from collections import defaultdict, Counter

# 初始化文本数据和词汇表
text = "low lower newest widest how are lo lower"
words = text.split()
# 添加单词结束符 '_'
words = [word + '_' for word in words]
print("words:", words)
# 创建初始词汇表，计算每个单词的频率
vocab = Counter(words)
print("vocab:", vocab)
# 定义函数来获取最常见的词片对
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        print(f"word:{word}, freq:{freq}, symbols:{symbols}")
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

# 定义函数来合并词汇表中的最常见词片对
def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# 执行 BPE 算法
num_merges = 10  # 设置合并次数
for i in range(num_merges):
    pairs = get_stats(vocab)
    print("pairs:", pairs)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"Step {i + 1}: Merged {best_pair} => New vocab: {vocab}")
# 重新执行 BPE 算法代码

# 初始化
vocab = Counter(words)

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"Step {i + 1}: Merged {best_pair} => New vocab: {vocab}")