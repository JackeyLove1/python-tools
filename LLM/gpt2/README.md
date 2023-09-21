准备数据：
```shell
python3.10 -m data.calcn.prepare_sft --input=small/small_sft_train.txt --output=small/small_sft_train.bin
```

sft训练：
```shell
# 使用small config，在sft训练集上训练，产出的模型保存到本地
python3.10 -u train_sft_calcn.py config/sft_train_small_calcn.py \
  --out_dir=out/calcn/small \
  --out_ckpt=ckpt_small.pt \
  --data_dir=data/calcn/small \
  --device=cpu \
  --compile=False 
```