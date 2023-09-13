import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import torch.nn as nn

# TODO: use arguments parse
#使用QLoRA引入的 NF4量化数据类型以节约显存
model_name_or_path ="baichuan-inc/Baichuan2-7B-Base" #远程：'Qwen/Qwen-7b-Chat'

bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

tokenizer = AutoTokenizer.from_pretrained(
   model_name_or_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

import random

def get_messages(conversation):
    select = random.choice
    messages, history = [], []
    for t in conversation:
        history.append((select(t[0]), select(t[-1])))

    for prompt, response in history:
        pair = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        messages.extend(pair)
    return messages


from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from torchkeras.chat import ChatLLM
llm = ChatLLM(model,tokenizer)

class MyDataset(Dataset):
    def __init__(self, conv, size=8
                 ):
        self.conv = conv
        self.index_list = list(range(size))
        self.size = size

    def __len__(self):
        return self.size

    def get(self, index):
        idx = self.index_list[index]
        messages = get_messages(self.conv)
        return messages

    def __getitem__(self, index):
        messages = self.get(index)
        input_ids, labels = llm.build_inputs_labels(messages, multi_rounds=True)  # 支持多轮
        return {'input_ids': input_ids, 'labels': labels}

who_are_you = ['请介绍一下你自己。','你是谁呀？','你是？',]
i_am = ['我叫梦中情炉，是一个三好炼丹炉：好看，好用，好改。我的英文名字叫做torchkeras，是一个pytorch模型训练模版工具。']
where_you_from = ['你多大了？','你是谁开发的呀？','你从哪里来呀']
i_from = ['我在2020年诞生于github星球，是一个有毅力的吃货设计和开发的。']
what_you_can = ['你能干什么','你有什么作用呀？','你能帮助我干什么']
i_can = ['我能够帮助你以最优雅的方式训练各种类型的pytorch模型，并且训练过程中会自动展示一个非常美丽的训练过程图表。']
conversation = [(who_are_you,i_am),(where_you_from,i_from),(what_you_can,i_can)]
print(conversation)
ds_train = ds_val = MyDataset(conversation)

# 如果pad为None，需要处理一下
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id


def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]

        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

import torch
dl_train = torch.utils.data.DataLoader(ds_train,batch_size=2,
                                       pin_memory=True,shuffle=False,
                                       collate_fn = data_collator)

dl_val = torch.utils.data.DataLoader(ds_val,batch_size=2,
                                    pin_memory=True,shuffle=False,
                                     collate_fn = data_collator)

from peft import get_peft_config, get_peft_model, TaskType
model.supports_gradient_checkpointing = True  #
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

import bitsandbytes as bnb
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

lora_modules = find_all_linear_names(model)
print(lora_modules)

from peft import AdaLoraConfig
peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=16,
    lora_alpha=16, lora_dropout=0.08,
    target_modules= lora_modules
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True
peft_model.print_trainable_parameters()

from torchkeras import KerasModel
from accelerate import Accelerator


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net.forward(**batch)[0]

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


# 仅仅保存QLora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path, 'adapter_model.bin')), strict=False)
    self.from_scratch = False


KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt

optimizer = bnb.optim.adamw.AdamW(peft_model.parameters(),
                                  lr=6e-03,is_paged=True)  #'paged_adamw'
keras_model = KerasModel(peft_model,loss_fn =None,
        optimizer=optimizer)

ckpt_path = 'qwen7b_multirounds'

keras_model.from_scratch=False

# keras_model.load_ckpt(ckpt_path) #支持加载微调后的权重继续训练(断点续训)
keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=100,patience=15,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path
               )

