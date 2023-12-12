import glob
import os

def get_dataset_directories(base_directory, num_processes):
    directories = []
    for process_id in range(num_processes):
        # Find directories matching the pattern "{process_id}_{idx}"
        matched_dirs = glob.glob(os.path.join(base_directory, f"{process_id}_*"))
        # print(matched_dirs)
        directories.extend(matched_dirs)
    return directories

from datasets import concatenate_datasets, load_from_disk

def load_and_concatenate_datasets(base_directory, num_processes):
    all_datasets = []
    directories = get_dataset_directories(base_directory, num_processes)
    for directory in directories:
        # Load all .arrow files from each directory
        dataset_files = glob.glob(os.path.join(directory, "*.arrow"))
        # for file in dataset_files:
        print(directory)
        try:
            dataset = load_from_disk(directory, keep_in_memory=True)
            all_datasets.append(dataset)
        except:
            continue

    concatenated_dataset = concatenate_datasets(all_datasets)
    return concatenated_dataset
def save_concatenated_dataset(concatenated_dataset, output_directory):
    concatenated_dataset.save_to_disk(output_directory)
base_directory= "/table_efs/users/rabonagy/cs236_final/datasets/"
num_processes = 50  # The number of processes you used
# directories = get_dataset_directories(base_directory, num_processes)
# Concatenate all datasets
concatenated_dataset = load_and_concatenate_datasets(base_directory, num_processes)

# concatenated_dataset.save_to_disk('/table_efs/users/rabonagy/cs236_final/train_dataset', num_proc=32)