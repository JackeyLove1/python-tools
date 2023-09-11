from dataclasses import dataclass, field
from typing import Optional
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
import torch
import torch.nn as nn

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    train_args_file: Optional[str] = field(default="llama2.json", metadata={"help" : "the train arguments"})
    seed : Optional[int] = field(default=0, metadata={"help" : "random seed"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args()

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 100)
        self.linear1 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 2)
    def forward(self, x):
        x = self.emb(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
def find_all_linear_names(model):
    """
    find all linears and add adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        print(f"name:{name}, modules:{module}")
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

models = TestModel()
print(find_all_linear_names(models))

def init_running():
    pass