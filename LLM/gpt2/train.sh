#!/bin/bash
python3.10 data/calcn/create_data.py
python3.10 -m data.calcn.prepare_sft --input=small/small_sft_train.txt --output=small/small_sft_train.bin
python3.10 -u train_sft_calcn.py config/sft_train_small_calcn.py \
  --out_dir=out/calcn/small \
  --out_ckpt=ckpt_small.pt \
  --data_dir=data/calcn/small \
  --device=cpu \
  --compile=False
# torchrun --standalone --nproc_per_node=8 train_sft_calcn.py