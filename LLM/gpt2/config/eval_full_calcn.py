# eval for for full data; defaults to data/v1.3

out_dir = 'data-v1.3-calcn'
eval_iters = 2
eval_only = True

wandb_log = False  # override via command line if you like
wandb_project = 'calcn-test'
wandb_run_name = 'eval0812a'

data_dir = 'data/calcn'
sft_train = 'v1.3/sft_train'
sft_val = 'v1.3/sft_train'  # temp
hit_ratio_val = 'v1.3/rl_train_512'
hit_ratio_val_size = 512  # as the name rl_train_512 suggests
gradient_accumulation_steps = 1
batch_size = 4
max_prompt_len = 32
max_resp_len = 136

# 85M GPT model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False  # do not torch compile the model
