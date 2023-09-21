"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_calcn.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_calcn.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_calcn.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_calcn.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer_calcn import CalcNTokenizer
from reward_calcn import reward as hit_reward
from hdfs_io import hcopy
from hdfs_io import hput
from tqdm import tqdm


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
out_ckpt = 'ckpt.pt'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
data_dir = 'data/calcn'
sft_train = 'small_sft_train'
sft_val = 'small_sft_val'  # temp, the same with training
hit_ratio_val = 'small_rl_train'  # temp, prompt only dataset
hit_ratio_val_size = 16  # 16 prompts
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_prompt_len = 32
max_resp_len = 136
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.1 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# deduced from configs
block_size = max_prompt_len + 1 + max_resp_len  # 169

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    if not out_dir.startswith('hdfs:'):  # only mkdir for valid local path
        os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# import ipdb
# ipdb.set_trace()

# make the calc-n task specific tonkenizer
tokenizer = CalcNTokenizer()

# local or hdfs file
def maybe_download_to_local(fullpath):
    if fullpath.startswith('hdfs:'):
        # download to local and return the local path
        print(f'downloading {fullpath} to local ./')
        hcopy(fullpath, './')
        full_dir, basename = os.path.split(fullpath)
        return f'./{basename}'
    # simply return the local path
    return fullpath

# poor man's data loader
train_data = np.memmap(maybe_download_to_local(os.path.join(data_dir, f'{sft_train}.bin')),
                       dtype=np.uint16, mode='r')
val_data = np.memmap(maybe_download_to_local(os.path.join(data_dir, f'{sft_val}.bin')),
                     dtype=np.uint16, mode='r')  # temp
val_hit_ratio_data = np.memmap(maybe_download_to_local(os.path.join(data_dir, f'{hit_ratio_val}.bin')),
                               dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # get a random batch of sequences, each sequence is of length block_size
    n_row = len(data) // block_size
    # the caller should ensure the last row data is a sentinel, which will never be sampled
    i_row = torch.randint(n_row - 1, (batch_size,))
    ix = i_row * block_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # make the mask, 1 for the response, 0 otherwise
    ix_sep = (x == tokenizer.sep_token_id).nonzero()
    ix_eos = (x == tokenizer.eos_token_id).nonzero()
    mask = torch.zeros_like(x, dtype=torch.float32)
    for i in range(batch_size):
        i_start = ix_sep[i][1] - 1  # for x: including [SEP] as the start
        i_end = ix_eos[i][1]  # for x: exclude the [EOS] on tail
        mask[i][i_start+1:i_end] = 1

    # Example.
    # prompt: p0, p1, p2; response: r0, r1, r2, r3; concatenated sequence: p0, p1, p2, [SEP], r0, r1, r2, r3, [EOS]
    # masked x: [SEP], r0, r1, r2, r3
    # masked y: r0, r1, r2, r3, [EOS]

    if device_type == 'cuda':
        # pin arrays x,y, mask, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), \
                     mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

    return x, y, mask

def get_val_hit_ratio_batch():
    data = val_hit_ratio_data

    i_row = torch.arange(hit_ratio_val_size)  # get all data
    ix = i_row * max_prompt_len
    x = torch.stack([torch.from_numpy((data[i:i+max_prompt_len]).astype(np.int64)) for i in ix])  # (bs, max_prompt_len)

    # append [SEP] on tail
    sep_tensor = torch.ones_like(x[:, -1]) * tokenizer.sep_token_id  # (bs,)
    sep_tensor = sep_tensor.view(-1, 1)  # (bs, 1)
    x = torch.concat([x, sep_tensor], dim=1)  # (bs, max_prompt_len + 1)

    if device_type == 'cuda':
        # pin arrays x,y, mask, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args['vocab_size'] = tokenizer.vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f'Unknown init_from {init_from}')
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, mask=mask, ignore_index=tokenizer.pad_token_id)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def calc_val_hit_ratio():
    pad_token_str = tokenizer.decode(tokenizer.pad_token_id)
    model.eval()
    X = get_val_hit_ratio_batch()
    total_count = X.size(0)
    hit_count = 0
    local_batch_size = batch_size
    if batch_size < total_count:
        n_batch_indexes = np.arange(0, total_count, batch_size)
    else:
        n_batch_indexes = np.ndarray([0])
        local_batch_size = total_count
    debug_pos, debug_neg = 1, 3
    for left in n_batch_indexes:
        mini_batch_tensor = model.generate(X[left:left+local_batch_size,:], max_resp_len, temperature=1.0, top_k=None)
        # full sequence: (1, max_prompt_len+1+max_resp_len)
        mini_batch_size = mini_batch_tensor.size(0)
        for ind in range(mini_batch_size):
            sequence_str = tokenizer.decode(mini_batch_tensor[ind].tolist())
            hit = hit_reward(sequence_str, partial_correct_reward=0.0, bad_format_reward=0.0)
            if hit > 0.999999:
                hit_count += 1
                if debug_pos > 0:
                    debug_pos -= 1
                    print(f'eval debug pos, reward: {hit}, seq: {sequence_str}')
            else:
                if debug_neg > 0:
                    debug_neg -= 1
                    print(f'eval debug neg, reward: {hit}, seq: {sequence_str}')
    model.train()
    return hit_count, total_count

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, mask = get_batch('train') # fetch the very first batch
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # eval mode: only calculate the hit_ratio, then exit
    if local_iter_num == 0 and eval_only:
        val_hit_count, val_total_count = calc_val_hit_ratio()
        val_hit_ratio = val_hit_count/val_total_count
        print(f"val hit ratio {val_hit_ratio:.4f} ({val_hit_count}/{val_total_count})")
        print('eval_only, exiting...')
        break

    t_batch_start = time.time()
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num != 0 and iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_hit_count, val_total_count = calc_val_hit_ratio()
        val_hit_ratio = val_hit_count/val_total_count
        dt = time.time() - t_batch_start
        print(f"validation: step {iter_num}: time {dt*1000:.2f}ms, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"val hit ratio {val_hit_ratio:.4f} ({val_hit_count}/{val_total_count})")
        if wandb_log:
            wandb.log({
                "val/loss": losses['val'],
                "val/hit_ratio": val_hit_ratio,
                "val/time": dt*1000
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if out_dir.startswith('hdfs:'):
                    print(f"saving checkpoint to local ./{out_ckpt}")
                    torch.save(checkpoint, f'./{out_ckpt}')
                    print(f"saving checkpoint to remote {out_dir}")
                    hput(f'./{out_ckpt}', out_dir)
                else:
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, f'{out_ckpt}'))

    t_train_start = time.time()
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, mask=mask, ignore_index=tokenizer.pad_token_id)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if wandb_log:
        wandb.log({
            "train/grad_norm": total_norm
        }, step=iter_num)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if iter_num % log_interval == 0 and master_process:
        dt = time.time() - t_train_start
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms/step, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "train/lr": lr,
                "train/loss": lossf,
                "train/mfu": running_mfu*100,  # convert to percentage
                "train/time": dt*1000
            }, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
