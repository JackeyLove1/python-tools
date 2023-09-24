#!/bin/bash
python3 create_data.py
torchrun --standalone --nproc_per_node=8 train_operation.py
