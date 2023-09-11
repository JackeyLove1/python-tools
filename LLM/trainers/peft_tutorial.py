# !pip uninstall tensorflow tensorflow-io accelerate transformers -y
# !pip install --no-deps tensorflow-io
# !pip install -U tensorflow bitsandbytes datasets loralib transformers accelerate peft

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# Post-processing
import torch
import torch.nn as nn

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def postprocessing_base_model(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)


postprocessing_base_model(model)

from peft import get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r = 8,
    lora_alpha= 32,
    lora_dropout=0.1,
    bias="none",
    target_modules=['q_proj', 'v_proj']
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Train with transformers
import transformers
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

trainer = transformers.Trainer(
    model = model,
    train_dataset = data['train'],
    args = transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs',
        report_to='none'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()