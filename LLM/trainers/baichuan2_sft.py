import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json
import deepspeed
import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments

'''
hostfile=""
deepspeed --hostfile=$hostfile fine-tune.py  \
    --report_to "none" \
    --data_path "data/belle_chat_ramdon_10k.json" \
    --model_name_or_path "baichuan-inc/Baichuan2-7B-Base" \
    --output_dir "output" \
    --model_max_length 512 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 False \
    --tf32 False \
    --use_lora 
'''
@dataclass
class ModelArguments:
    model_name_or_path : Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/data.json", metadata={"help": "path to training data"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="cached")
    optim: str = field(default="adamw_torch")
    model_max_legth: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }
    )
    output_dir : str = field(default="output")
    use_lora: bool = field(default=True)
    lora_r : int = field(default=64)
    lora_alpha : int = field(default=16)
    lora_dropout : float = field(default=0.1)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path,
            tokenizer : transformers.LlamaTokenizer,
            model_max_length,
            user_tokens=[195],
            assistant_tokens=[196],
    ):
        super().__init__(SupervisedDataset, self)
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    '''
    {
  "id": "77771",
  "conversations": [
   {
    "from": "human",
    "value": "写一篇关于气候变化对海洋生态的影响的文章。\n"
   },
   {
    "from": "gpt",
    "value": "好的，以下是你要求的文章：\n气候变化对海洋生态的影响\n气候变化对海洋生态产生了广泛的影响。全球变暖导致海洋温度上升，这可能会对许多水生生物的生存、繁殖和迁移造成巨大的影响。另外，海洋酸化也是一个问题，因为大气中二氧化碳的增加导致海水中的pH值下降，使得许多生物的外骨骼变得脆弱，从而影响它们的生存和繁殖。\n此外，气候变化还会引起海平面上升，这不仅对海岸线和沿海居民产生影响，还会对许多海洋生物的栖息地造成影响。一些岛屿上的陆地生物面临几乎完全消失的风险，而另一些生物范围则会扩大到更广泛的领域，这可能会导致新的竞争和生态平衡的改变。\n总之，气候变化并不只是一个地球气温的问题，它对海洋生态系统产生了广泛和深远的影响。\n"
   },
   {
    "from": "human",
    "value": "从刚刚的文章中提取出气候变化对海洋生物繁殖的影响。\n"
   },
   {
    "from": "gpt",
    "value": "文章已经提到过气候变化会对海洋生物的生存、繁殖和迁移造成影响。例如，海洋酸化会使得许多生物的外骨骼变得脆弱，进而影响它们的生存和繁殖。此外，海洋温度上升也可能会对很多生物的生存和繁殖产生负面影响。"
   }
  ]
 },
    '''
    def preprocessing(self, example):
        input_ids = []
        labels = []

        for message in example['conversations']:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)

            if from_ == "human":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[:self.model_max_length]
        labels = input_ids[:self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids" : input_ids,
            "labels" : labels,
            "attention_mask" : attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model_args, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()