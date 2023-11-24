#TODO: dump file meta and support breakpoint download
import os
import shutil

# Define the source and destination directories
source_dir = "/mnt/sync/sync/pypi/web/packages/"
dest_dir = "/path/to/destination/directory"
package_nums = 10000
# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

hash_map = {}
current_nums = 0

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    global current_nums
    if current_nums > package_nums:
        break
    for file in files:
        # Check if the file is a .gz file
        if file.endswith(".gz"):
            # Calculate hash of the file name
            package_name = hash(file.split("-")[0])
            if package_name in hash_map:
                continue
            else:
                hash_map[package_name] = 1
            # Construct full file path
            file_path = os.path.join(root, file)
            # Copy the file to the destination directory
            shutil.copy(file_path, dest_dir)
            current_nums += 1
            print("Succeed to download file:%s, current nums: %d".format(file_path, current_nums))
            if current_nums > package_nums:
                break

print("Copying of .gz files completed.")
