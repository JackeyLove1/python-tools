import os

start_dir = "fine-tune"
for root, dirs, files in os.walk(start_dir):
    print("root:", root)
    for file in files:
        print("file:", file)
