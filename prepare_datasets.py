from datasets import Dataset
from PIL import Image
import json
import random
import webdataset as wds
import io
from multiprocessing import Process
import boto3
from pudb.remote import set_trace
def pick_task(src):
    for sample in src:
        # print(sample.keys())
        # Load and post-process the selected task if it exists in the sample
        # print(sample.keys())
        if 'layoutanalysis.json' in sample:
            # print(json.loads(sample[task])['ground_truth'])
            

            image = sample['screenshot.png']
            # print('iterating')
            # print(json.loads(sample['annotation.json'])['downloaded_css_files_count'])
            if json.loads(sample['annotation.json'])['css_files_count']>0 and json.loads(sample['annotation.json'])['downloaded_css_files_count']/json.loads(sample['annotation.json'])['css_files_count']>0.5:
                image = Image.open(io.BytesIO(image)).resize((512, 512), Image.BILINEAR)
                # print('yielding')
                annotation = json.loads(sample['annotation.json'])
                original_words_data = annotation['original_words_data']
                words = []
                # original_words_data = sample[0]['original_words_data']
                for word in original_words_data:
                    if word.get('type')=='text':
                        words.append({'content':word['content'], 'x':word['x'],'y':word['y'],'width':word['width'],'height':word['height']})
                title = original_words_data[0]['content']
                yield {'image':image, 'layout':sample['layoutanalysis.json'], 'words':words, 'title':title}
            else:
                continue
        else:
            # If the selected task is not in the sample, skip it
            continue
def build_wds_dataset(urls, split):
    urls = [f"pipe:aws s3 cp {url} -" for url in urls]
    pipeline = [wds.ResampledShards(urls, nshards=len(urls)),wds.tarfile_to_samples(handler = wds.warn_and_continue)]
    if split=='train':
        pipeline.append(
            wds.shuffle(bufsize=1000, initial=0,handler = wds.warn_and_continue)
        )
    pipeline.append(
        pick_task
    )
    if split=='train':
        dataset = wds.DataPipeline(*pipeline).with_epoch(20000).with_length(20000)
    else:
        dataset = wds.DataPipeline(*pipeline).with_epoch(16).with_length(16)
    return dataset
dataset_name_or_paths = ["downloaded_bytes/wds_debug/CC-MAIN-2023-06/20000_30000/","downloaded_bytes/wds_debug/CC-MAIN-2023-06/10000_20000/","downloaded_bytes/wds_debug/CC-MAIN-2023-06/0_10000/","downloaded_bytes/wds_debug/CC-MAIN-2022-40/0_0/"]
s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')
urls = []


for prefix in dataset_name_or_paths:

    bucket='rabonagy-bedrock'

    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('.tar'):
                if f"s3://{bucket}/{key}"!='s3://rabonagy-bedrock/downloaded_bytes/wds_debug/CC-MAIN-2023-06/0_10000/machine_16-00006.tar' :
                    urls.append(f"s3://{bucket}/{key}")
# Turn the code below to multiprocessing
def process_dataset_subset(urls, split, process_id):
    dataset = build_wds_dataset(urls, split)
    data_for_hf = []
    for idx, item in enumerate(dataset):
        # if idx % 100 == 0:
        #     print(f"Process {process_id}, Index: {idx}")
        data_for_hf.append(item)
        # Optionally, save intermediate results to disk
        if idx % 1000 == 0 and idx > 0:
            # set_trace(term_size=(120,30))
            data_dict = {key: [dic[key] for dic in data_for_hf] for key in data_for_hf[0]}
            hf_dataset = Dataset.from_dict(data_dict)
            hf_dataset.save_to_disk(f"/table_efs/users/rabonagy/cs236_final/simplified_datasets/{process_id}_{idx}")
            print(f"Saving Process {process_id}, Index: {idx}")
            del data_for_hf
            data_for_hf = []
    # Save the last chunk of data
    if data_for_hf:
        data_dict = {key: [dic[key] for dic in data_for_hf] for key in data_for_hf[0]}
        hf_dataset = Dataset.from_dict(data_dict)
        hf_dataset.save_to_disk(f"/table_efs/users/rabonagy/cs236_final/simplified_datasets/{process_id}_final")


def parallel_process(urls, split, num_processes):
    # Split the URLs into approximately equal chunks
    chunk_size = len(urls) // num_processes
    processes = []

    for i in range(num_processes):
        start = i * chunk_size
        end = None if i == num_processes - 1 else (i + 1) * chunk_size
        process_urls = urls[start:end]
        p = Process(target=process_dataset_subset, args=(process_urls, split, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() 
num_processes = 96  # For example, use 8 processes

# Call the function to process the dataset in parallel
parallel_process(urls, 'train', num_processes)