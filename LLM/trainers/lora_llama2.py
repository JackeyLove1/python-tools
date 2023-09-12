'''
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12 tensorboard
apt-get -qq install poppler-utils tesseract-ocr
pip install -q unstructured["local-inference"]==0.7.4 pillow
'''

import os
import torch
import transformers
from typing import Optional, Dict
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import deepspeed
from transformers import LlamaTokenizer
# Used for multi-gpu
local_rank = -1
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
weight_decay = 0.001
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
max_seq_length = None
# The model that you want to train from the Hugging Face hub
model_name = "guardrail/llama-2-7b-guanaco-instruct-sharded"
# Fine-tuned model name
new_model = "llama-2-7b-guanaco-dolly-mini"
# The instruction dataset to use
dataset_name = "databricks/databricks-dolly-15k"
# Activate 4-bit precision base model loading
use_4bit = True
# Activate nested quantization for 4-bit base models
use_nested_quant = False
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Number of training epochs
num_train_epochs = 2
# Enable fp16 training, (bf16 to True with an A100)
fp16 = False
# Enable bf16 training
bf16 = False
# Use packing dataset creating
packing = False
# Enable gradient checkpointing
gradient_checkpointing = True
# Optimizer to use, original is paged_adamw_32bit
optim = "paged_adamw_32bit"
# Learning rate schedule (constant a bit better than cosine, and has advantage for analysis)
lr_scheduler_type = "cosine"
# Number of optimizer update steps, 10K original, 20 for demo purposes
max_steps = -1
# Fraction of steps to do a warmup for
warmup_ratio = 0.03
# Group sequences into batches with same length (saves memory and speeds up training considerably)
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 10
# Log every X updates steps
logging_steps = 1
# The output directory where the model predictions and checkpoints will be written
output_dir = "./results"
# Load the entire model on the GPU 0
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# Visualize training
report_to = "tensorboard"
# Tensorboard logs
tb_log_dir = "./results/logs"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=model_name, metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default=dataset_name, metadata={"help": "the dataset name"}
    )


def load_model(model_name):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config


model, tokenizer, peft_config = load_model(model_name)


def format_dolly(sample):
    instruction = f"<s>[INST] {sample['instruction']}"
    context = f"Here's some context: {sample['context']}" if len(sample["context"]) > 0 else None
    response = f" [/INST] {sample['response']}"
    # join all the parts together
    prompt = "".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


# apply prompt template per sample
# dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
# Shuffle the dataset
# dataset_shuffled = dataset.shuffle(seed=42)
# Select the first 50 rows from the shuffled dataset, comment if you want 15k
# dataset = dataset_shuffled.select(range(50))
# dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
dataset_shuffled = dataset.shuffle(seed=42)
# Select the first 50 rows from the shuffled dataset, comment if you want 15k
dataset = dataset_shuffled.select(range(100))
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
            self,
            data,
            tokenizer : transformers.LlamaTokenizer,
            model_max_length,
    ):
        super().__init__(SupervisedDataset, self)
        self.data = data
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def preprocessing(self, item):
        data = self.tokenizer.tokenize(item['text'],)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = []

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

# prompt = "how long does an American football match REALLY last, if you substract all the downtime?"
#
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)


trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, max_length=max_seq_length,
    ),
)

trainer.train()
trainer.model.save_pretrained(output_dir)
