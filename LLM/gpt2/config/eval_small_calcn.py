# eval for a small calcn model
# good for debugging and playing on macbooks and such

out_dir = 'out/calcn/small'
out_ckpt = 'ckpt.pt'
eval_iters = 200
eval_only = True

wandb_log = False  # override via command line if you like
wandb_project = 'calcn_small'
wandb_run_name = 'test_small'

data_dir = 'data/calcn/small'
sft_train = 'small_sft_train'
sft_val = 'small_sft_train'  # temp, the same with training
hit_ratio_val = 'small_rl_train'  # temp, prompt only dataset
hit_ratio_val_size = 16  # 16 prompts
gradient_accumulation_steps = 1
batch_size = 4
max_prompt_len = 32
max_resp_len = 136

# small GPT, 0.79M
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False  # do not torch compile the model
