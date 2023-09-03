import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=./data/botchan.txt --model_prefix=m --vocab_size=2000')

sp = spm.SentencePieceProcessor()
sp.load('m.model')

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
print(sp.decode_ids([209, 31, 9, 375, 586]))

# returns vocab size
print(sp.get_piece_size())

# id <=> piece conversion
print(sp.id_to_piece(209))
print(sp.piece_to_id('▁This'))

# returns 0 for unknown tokens (we can change the id for UNK)
print(sp.piece_to_id('__MUST_BE_UNKNOWN__'))

# , ,  are defined by default. Their ids are (0, 1, 2)
#  and  are defined as 'control' symbol.
for id in range(3):
    print(sp.id_to_piece(id), sp.is_control(id))

print('bos=', sp.bos_id())
print('eos=', sp.eos_id())
print('unk=', sp.unk_id())
print('pad=', sp.pad_id())  # disabled by default


print(sp.encode_as_ids('Hello world'))

# Prepend or append bos/eos ids.
print([sp.bos_id()] + sp.encode_as_ids('Hello world') + [sp.eos_id()])

spm.SentencePieceTrainer.train('--input=./data/botchan.txt --model_prefix=m_bpe --vocab_size=2000 --model_type=bpe')
sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load('m_bpe.model')
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids("This is a test"))

print('*** BPE ***')
print(sp_bpe.encode_as_pieces('thisisatesthelloworld'))
print(sp_bpe.nbest_encode_as_pieces('hello world', 5))  # returns an empty list.

spm.SentencePieceTrainer.train('--input=./data/botchan.txt --model_prefix=m_unigram --vocab_size=2000 --model_type=unigram')
sp_unigram = spm.SentencePieceProcessor()
sp_unigram.load('m_unigram.model')

print('*** Unigram ***')
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids("This is a test"))
print(sp_unigram.encode_as_pieces('thisisatesthelloworld'))
print(sp_unigram.nbest_encode_as_pieces('thisisatesthelloworld', 5))
