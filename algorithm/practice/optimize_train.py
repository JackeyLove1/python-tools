import gc

import torch
import torch.optim

num_epoch = 10
optimizer = None
dataloader = None
model = None
loss_func = None

for epoch in range(num_epoch):
    for i, sample in enumerate(dataloader):
        inputs, labels = sample
        optimizer.zero_grad()
        outputs = optimizer(inputs)
        loss = loss_func(labels, outputs)
        loss.backward()
        optimizer.step()

# use AMP
# use gradient step
# use gradient clip
# use gc
optimizer = None
scaler = torch.cuda.amp.GradScaler()
NUM_ACCUMULATION_STEPS = 10
for epoch in range(num_epoch):
    for idx, sample in enumerate(dataloader):
        inputs, labels = sample
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_func(labels, outputs)
        scaler.scale(loss).backward()
        loss = loss / NUM_ACCUMULATION_STEPS
        if ((idx + 1) % NUM_ACCUMULATION_STEPS) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        torch.cuda.empty_cache()
        _ = gc.collect()