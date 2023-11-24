from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class CustomizedArguments:
    """Some arguments self defined"""
    max_seq_length: int = field(default=1000,  metadata={"help" : "input length"})
    train_file : str = field(default='data/train.csv', metadata={"help" : "train file"})
    model_name_or_path : str = field(default="MBZUAI/LaMini-GPT-1.5B", metadata={"help" : "model name"})
    eval_file : Optional[str] = field(default="", metadata={"help" : "eval file"})

@dataclass
class QLoRAArguments:
    """QLora arguments"""
    model_name_or_path : str = field(default='MBZUAI/LaMini-GPT-1.5B', metadata={"help" : "model name"})
    train_file : str = field(default='data/train.csv', metadata={"help" : "train file"})
    eval_file : Optional[str] = field(default="", metadata={"help" : "eval file"})
    max_seq_length : int = field(default=512, metadata={"help" : "max length"})
    task_type : Optional[str] = field(default="", metadata={"help" : "task type: [sft, pretrain]"})
    lora_rank : Optional[int] = field(default=64, metadata={"help" : "lora rank"})
    lora_alpha : Optional[int] = field(default=16, metadata={"help" : "lora alpha"})
    lora_dropout : Optional[float] = field(default=0.05, metadata={"help" : "lora dropout"})

def example_test():
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    args, training_args = parser.parse_json_file("test.json")
    print(parser.print_help())
    print(args.model_name_or_path)
    print(args.lora_rank)
    print(training_args.dataloader_num_workers)


example_test()