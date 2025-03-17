import os
import json
from pathlib import Path
cwd_path = os.getcwd()
config_path = Path(f"{cwd_path}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)

# Extracting data from parent directory
os.chdir("..")
data_folder = Path(f"{os.getcwd()}/data/babyview/adult_english_only")
word_list = Path(f"{data_folder}/vedi_words_with_aoa.csv")
max_frames = Path(f"{data_folder}/max_and_shuffled_baseline_vedi_words_only.csv")
save_dir = config_data.get("save_folder")
frame_batch_size = config_data.get("frame_batch_size")
clip_batch_size = config_data.get("clip_batch_size")
prefetch = config_data.get("prefetch")
existing_image_embeddings_path = config_data.get("old_save_folder")

command = (f"python3 {cwd_path}/batch_clip_on_max_frames.py --csv_file {max_frames} "
f"--word_list {word_list} --save_root_dir {save_dir} --frame_batch_size {frame_batch_size} "
f"--clip_batch_size {clip_batch_size} --prefetch {prefetch} --existing_image_embeddings_path {existing_image_embeddings_path}")
print(command)
os.system(command)