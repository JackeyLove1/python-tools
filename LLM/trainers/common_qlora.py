'''
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12 tensorboard
apt-get -qq install poppler-utils tesseract-ocr
pip install -q unstructured["local-inference"]==0.7.4 pillow
'''

import os
import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict
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
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Shuffle the dataset
dataset_shuffled = dataset.shuffle(seed=42)

# Select the first 50 rows from the shuffled dataset, comment if you want 15k
dataset = dataset_shuffled.select(range(50))

dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
dataset_shuffled = dataset.shuffle(seed=42)

# Select the first 50 rows from the shuffled dataset, comment if you want 15k
dataset = dataset_shuffled.select(range(100))

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


prompt = "how long does an American football match REALLY last, if you substract all the downtime?"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

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

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
trainer.model.save_pretrained(output_dir)
