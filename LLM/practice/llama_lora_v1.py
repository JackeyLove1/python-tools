# !pip3 install -U transformers torch trl peft accelerate bitsandbytes vllm triton

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer

tqdm.pandas()


# Define and parse arguments.
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = "meta-llama/Llama-2-7b-hf"
    dataset_name: Optional[str] = "timdettmers/openassistant-guanaco"
    dataset_text_field: Optional[str] = "text"
    log_with: Optional[str] = None
    learning_rate: Optional[float] = 1.41e-5
    batch_size: Optional[int] = 4
    seq_length: Optional[int] = 512
    gradient_accumulation_steps: Optional[int] = 2
    load_in_8bit: Optional[bool] = False
    load_in_4bit: Optional[bool] = True
    use_peft: Optional[bool] = True
    trust_remote_code: Optional[bool] = True
    output_dir: Optional[str] = "output"
    peft_lora_r: Optional[int] = 32
    peft_lora_alpha: Optional[int] = 16
    logging_steps: Optional[int] = 1
    use_auth_token: Optional[bool] = True
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = -1
    save_steps: Optional[int] = 100
    save_total_limit: Optional[int] = 10
    push_to_hub: Optional[bool] = False
    hub_model_id: Optional[str] = None
    train_samples: Optional[int] = 1000


# parser = HfArgumentParser(ScriptArguments)
script_args = ScriptArguments()

print(script_args.model_name)

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # This means: fit the entire model on the GPU:0
    device_map = {"": 0}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)

# Step 2: Load the dataset
dataset = load_dataset(script_args.dataset_name, split="train")

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

'''
# sample train
import random
random.seed(0)
train_datasets = dataset.filter(lambda x : random.random() < 0.05)
'''

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
