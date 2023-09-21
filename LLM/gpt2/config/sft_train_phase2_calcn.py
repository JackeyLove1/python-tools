# sft training for phase2
import datetime

out_dir = 'out/calcn/phase2'
out_ckpt = 'ckpt.pt'
eval_interval = 500
eval_iters = 2
log_interval = 100  # don't print too often

always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = 'calcn_phase2'
wandb_run_name = 'yourname_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 'run' + str(time.time())'

data_dir = 'data/calcn/phase2'
sft_train = 'sft_train'
sft_val = 'sft_train'  # the same with training, for metrics of loss
hit_ratio_val = 'rl_val'  # prompt only dataset, for metrics of hit_ratio
hit_ratio_val_size = 512  # TODO: confirm
gradient_accumulation_steps = 1
batch_size = 128
max_prompt_len = 32
max_resp_len = 136

# "big" GPT, 10.6M
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # fine to set it zero

learning_rate = 5e-4  # 
max_iters = 19423 # ~ 2 epoches TODO: confirm it
lr_decay_iters = 19423  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# gpu machine
device = 'cuda'  # run on cpu only
compile = True  # do not torch compile the model
