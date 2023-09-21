# sft trainining for data/v1.3
import datetime

out_dir = 'data-v1.3-calcn'
eval_interval = 500
eval_iters = 2
log_interval = 100  # don't print too often

always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = 'nanoGPT'
wandb_run_name = 'calcmn_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 'run' + str(time.time())'

data_dir = 'data/calcn'
sft_train = 'v1.3/sft_train'
sft_val = 'v1.3/sft_train'  # temp
hit_ratio_val = 'v1.3/rl_val'
hit_ratio_val_size = 512  # as the name rl_train_512 suggests
gradient_accumulation_steps = 1
batch_size = 128
max_prompt_len = 32
max_resp_len = 136

# 85M GPT model
n_head = 16
n_layer = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

learning_rate = 5e-4  # 
max_iters = 19423 # ~ 2 epoches
lr_decay_iters = 19423  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = True  # do not torch compile the model
