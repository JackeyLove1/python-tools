import sentencepiece as spm
import os
txt_path = "data"
# merge txt
file_list = [os.path.join(txt_path, file) for file in os.listdir(txt_path) if file.endswith("txt")]
print(f"file_list:{file_list}")

# write
with open('data/corpus.txt', 'w') as merged_file:
    # 遍历每个文件
    for file_name in file_list:
        with open(file_name, 'r') as current_file:
            for line in current_file:
                merged_file.write(line)
        merged_file.write('\n')

# train
spm.SentencePieceTrainer.train(
    input='data/corpus.txt',
    model_prefix='tokenizer',
    vocab_size=50000,
    # user_defined_symbols=['foo', 'bar'],
    character_coverage=1.0,
    model_type="bpe",
)
