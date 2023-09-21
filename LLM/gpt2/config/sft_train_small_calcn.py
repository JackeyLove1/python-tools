# sft training for a small calcn model
# good for debugging and playing on macbooks and such

out_dir = 'out/calcn/small'
out_ckpt = 'ckpt.pt'
eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = 'calcn_small'
wandb_run_name = 'test_small'

data_dir = 'data/calcn/small'
sft_train = 'small_sft_train'
sft_val = 'small_sft_train'  # the same with training, for metrics of loss
hit_ratio_val = 'small_rl_train'  # prompt only dataset, for metrics of hit_ratio
hit_ratio_val_size = 16  # 16 prompts
gradient_accumulation_steps = 1
batch_size = 20
max_prompt_len = 32
max_resp_len = 136

# small GPT, 0.79M
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 3000
lr_decay_iters = 300  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False  # do not torch compile the model
