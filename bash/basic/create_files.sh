#!/bin/bash
num_files=$1
for i in $(seq 1 "$num_files")
do
  touch "file${i}.txt"
done

# for i in {1..10};do touch "file${i}.txt";done