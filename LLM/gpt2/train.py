import os
from transformers import set_seed
from contextlib import nullcontext
import numpy as np
import time
import torch
import torch.nn as nn
from model import GPT, Config
from tqdm import tqdm

batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337

# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use device:", device)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
profile = False  # use pytorch profiler, or just simple benchmarking?

set_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if real_data:
    dataset = "shakespeare_char"
    data_dir = os.path.join("data", dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')


    def get_batch(split):
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        # (batch_size, block_size)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
        if torch.cuda.is_available():
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
gptconf = Config(
    block_size=block_size,
    n_layer=1,
    n_head=12,
    n_embd=768,
    dropout=0,
    bias=bias,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95),
                                       device_type=device_type)
if compile and torch.cuda.is_available():
    print("Compiling model ... ")
    model = torch.compile(model)

if profile:
    # TODO
    pass
else:
    for stage, num_steps in enumerate([10, 20]):
        t0 = time.time()
        for k in range(num_steps):
            X, Y = get_batch("train")
            with ctx:
                logits, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        dt = time.time() - t0
        if stage == 1:
            print(f"time per iteration: {dt / num_steps * 1000:.4f}ms")
