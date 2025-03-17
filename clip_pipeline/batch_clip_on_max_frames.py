from itertools import product
import os
import torch
import numpy as np
import argparse
from clip_client import Client
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Initialize the argparse
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--csv_file", type=str, required=True)
args_parser.add_argument("--word_list", type=str, required=False)
args_parser.add_argument("--existing_image_embeddings_path", type=str, required=False)
args_parser.add_argument("--server_url", type=str, default="grpc://0.0.0.0:51000")
args_parser.add_argument("--save_root_dir", type=str, required=False)
args_parser.add_argument("--frame_batch_size", type=int, default=512)
args_parser.add_argument("--clip_batch_size", type=int, default=256)
args_parser.add_argument("--prefetch", type=int, default=100)

args = args_parser.parse_args()

# Ensure that torch is using the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the CLIP client
client = Client(args.server_url)

# Setup directories
csv_file = pd.read_csv(args.csv_file)#.head(50) --used head(50) for testing
utterances_df = csv_file.drop_duplicates(subset=["utterance_no"], keep="first")
save_root_dir = args.save_root_dir or "./"
frame_batch_size = args.frame_batch_size
clip_batch_size = args.clip_batch_size
prefetch = args.prefetch
if args.word_list:
    word_df = pd.read_csv(args.word_list)
    words = word_df["object"].dropna().tolist()

print("Starting batch processing for max frames of all preprocessed videos:")

def save_embedding(embedding, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, file_name)
    # Convert tensor to numpy before saving
    np_embedding = embedding.cpu().numpy()
    np.save(path, np_embedding)
    return path

# Collect unique texts
texts = set()
# Collect all max frame paths with utterance information
frame_paths = []

# if non-empty word list has been provided, we calculate alignment at the word level instead of utterance level
if words:
    texts = set(words)
    frame_paths = []
    for utterance_no, word in product(utterances_df['utterance_no'].unique(), words):
        max_frame = utterances_df.loc[utterances_df['utterance_no'] == utterance_no, 'max_frame'].values[0]
        frame_paths.append((utterance_no, word, max_frame))
else:
    texts = set(utterances_df['text'])
    frame_paths = []
    for utterance_no, text in zip(utterances_df['utterance_no'], utterances_df['text']):
        max_frame = utterances_df.loc[utterances_df['utterance_no'] == utterance_no, 'max_frame'].values[0]
        frame_paths.append((utterance_no, text, max_frame))

# Convert set back to list for ordering
texts = list(texts)
# remove empty strings
# TODO: don't re-calculate existing embeddings, we only have 50 for the VEDI dataset so not a big deal here
texts = [text for text in texts if text]
# Get all unique text embeddings at once
print(f"Processing texts, clip_batch_size={clip_batch_size}...")
text_embeddings = torch.tensor(client.encode(texts, batch_size=clip_batch_size, show_progress=True, prefetch=prefetch)).to(device)
text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)

# Create a mapping of text to its embedding
text_to_embedding = {text: embedding for text, embedding in zip(texts, text_embeddings)}

if(len(frame_paths) == 0):
    raise ValueError(f"No frames found:{frame_paths}")

if args.existing_image_embeddings_path:
    base_image_embedding_path = args.existing_image_embeddings_path

print(f"Processing frames in batches, frame_batch_size={frame_batch_size}, clip_batch_size={clip_batch_size}...")
# Process frame paths in batches and calculate dot products
results = []
for i in range(0, len(frame_paths), frame_batch_size):
    batch_frame_paths = frame_paths[i:i+frame_batch_size]
    paths = [path for _, _, path in batch_frame_paths]
    if not os.path.exists(paths[0]):
        raise ValueError(f"Frame path not found : {paths[0]}")
    
    # skipping image embedding calc if all of the 
    if base_image_embedding_path is not None:
        # TODO: what if we have an incomplete embedding set? probbaly unlikely if this is step 2 after extracting frames, should also rename base_image_embedding_path
        batch_image_embeddings = torch.tensor(client.encode(paths, batch_size=clip_batch_size, show_progress=True, prefetch=prefetch)).to(device)
        batch_image_embeddings /= torch.norm(batch_image_embeddings, dim=1, keepdim=True)

    for j, (utterance_no, text, frame_path) in enumerate(batch_frame_paths):
        if not text:
            continue
        text_embedding = text_to_embedding[text]
        subject_id = utterance_no.split("_")[0]
        short_utterance_no = utterance_no.split("_")[-1]
        video_name = utterance_no.rsplit('_', 1)[0]
        if base_image_embedding_path is not None:
            image_embedding_path = Path(f'{base_image_embedding_path}/{subject_id}/all_embeddings/image_embeddings/{video_name}/{short_utterance_no}/{os.path.splitext(os.path.basename(frame_path))[0]}_image_embedding.npy')
            image_embedding = torch.tensor(np.load(image_embedding_path)).to(device)
        else:
            image_embedding = batch_image_embeddings[j]
        dot_product = torch.dot(text_embedding, image_embedding).item()
        # making sure we're saving a single embedding file for each word
        if words:
            text_embedding_path = Path(f'{save_root_dir}/word_embeddings/{text}_text_embedding.npy')
            if not os.path.exists(text_embedding_path):
                text_embedding_path = save_embedding(text_embedding, Path(f'{save_root_dir}/word_embeddings'), f'{text}_text_embedding.npy')
        else:
            text_embedding_path = save_embedding(text_embedding, os.path.join(save_root_dir, f'{subject_id}/all_embeddings/{video_name}'), f'{short_utterance_no}_text_embedding.npy')
        
        if base_image_embedding_path is None:
            image_embedding_path = save_embedding(image_embedding, os.path.join(save_root_dir, f'{subject_id}/all_embeddings/{video_name}/{short_utterance_no}'), f'{os.path.splitext(os.path.basename(frame_path))[0]}_image_embedding.npy')
        
        original_utterance_row = utterances_df.loc[utterances_df['utterance_no'] == utterance_no].iloc[0]
        results.append({
            "utterance_no": utterance_no,
            "subject_id": original_utterance_row["subject_id"],
            "video_name": original_utterance_row["video_name"],
            "text": original_utterance_row["text"],
            "max_frame": frame_path,
            "keyword_used": text,
            "keyword_found": original_utterance_row["keyword_found"],
            "total_keywords_in_utterance": original_utterance_row["total_keywords_in_utterance"],
            "dot_product": dot_product,
            "original_max_dot_product": original_utterance_row["max_dot_product"],
            **original_utterance_row[['months_old', 'date_cleaned_mdy',]].to_dict(),
            "text_embedding_path": text_embedding_path,
            "image_embedding_path": image_embedding_path,
        })

# Extract all unique utterance_no values
utterance_nos = set(res["utterance_no"] for res in results)

# using the convergent logit scale from CLIP
logit_scale = 100

print("Calculating softmaxes")
# Compute softmax for each utterance_no group
for utterance_no in tqdm(utterance_nos):
    # Get all results for the given utterance_no
    utterance_results = [res for res in results if res["utterance_no"] == utterance_no]
    
    # Extract dot_product values
    dot_products = torch.tensor([res["dot_product"] for res in utterance_results])
    scaled_logits = logit_scale * dot_products
    # Compute softmax using PyTorch's softmax function 
    softmax_values = scaled_logits.softmax(dim=-1).detach().cpu().numpy()
    # Assign softmax values back to the corresponding results
    for i, res in enumerate(utterance_results):
        res["softmax"] = softmax_values[i]

# Save results
output_dir = os.path.join(save_root_dir, "all_result_csv_files")
print(output_dir)
os.makedirs(output_dir, exist_ok=True)

df_results = pd.DataFrame(results)
# Save the DataFrame to a CSV file
csv_file_path = os.path.join(output_dir, 'clip_max_frame_results.csv')
df_results.to_csv(csv_file_path, index=False)

print(f"Max frame processing completed!")