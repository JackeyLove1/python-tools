import torch
import torch.nn as nn
import math

A = 1000
B = 100
C = 20


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Embedding(A, B)
        self.linear2 = nn.Linear(B, C)

    def forward(self, x):
        out_main = self.linear2(self.linear1(x))
        out = out_main
        return out


# 创建模型实例
model = MyModel()



class LoraLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, adapter_name="default"):
        super().__init__()
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.adapter_name = adapter_name
        self.weight = None

    def update_layer(self):
        self.r[self.adapter_name] = 16
        self.lora_alpha[self.adapter_name] = 16
        lora_dropout_layer = nn.Dropout(p=0.1)
        self.lora_dropout.update(nn.ModuleDict({self.adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A.update(nn.ModuleDict({self.adapter_name: nn.Linear(self.in_features, r, bias=False)}))
        self.lora_B.update(nn.ModuleDict({self.adapter_name: nn.Linear(r, self.out_features, bias=False)}))
        self.scaling[self.adapter_name] = 1
        self.reset_lora_parameters(self.adapter_name)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)


for k, v in model.named_modules():
    print(f"{k} : {v}, type:{type(v)}")
    if isinstance(v, nn.Linear):
        new_module = LoraLayer(v.in_features, v.out_features)
        setattr(model, k, new_module)
        
print(model)
