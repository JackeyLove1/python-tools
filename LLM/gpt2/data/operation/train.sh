#!/bin/bash
torchrun --standalone --nproc_per_node=4 train_operation.py
