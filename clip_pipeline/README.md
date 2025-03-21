# Environment

- CPU:AMD EPYC 9334 32-Core Processor
- GPU: NVIDIA A40,  Driver Version: 535.104.12 
- Memory: 756GB
- System: Ubuntu 20.04.6
- Pytorch 2.2.2, CUDA 12.1, Python3.10

# Installation

```
conda create --name torch python=3.10
conda activate torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ffmpeg
pip install numpy pandas tqdm opencv-python pillow
pip install matplotlib
pip install clip-server
pip install clip-client
pip install nvidia-pyindex 
pip install "clip_server[tensorrt]"
pip install --upgrade transformers accelerate
```
## [Optional] Install flash-attn to accelerate the inference of transformers
```
conda install nvidia/label/cuda-12.1.0::cuda
export CUDA_HOME=$(python -c "import os; print(os.path.dirname(os.path.dirname(os.path.dirname(os.__file__))))")
pip install flash-attn --no-build-isolation
```

## CLIP on all videos

### Start CLIP Server(TensorRT accelerated)

```
cd ~/workspace/clip-alignment/babyview_exps
conda activate torch
python -m clip_server ./tensorrt-flow_replica_4.yml
```



### Demo: Run CLIP on single video

#### Step1: Extract all the frames from the specifc video

```
cd ~/workspace/clip-alignment/babyview_exps
conda activate torch
python extract_all_frames.py --video_file /data/yinzi/babyview/Babyview_Main/00230001/00230001_GX010002_10.30.2023-11.05.2023_11.05.2023-11:00am.MP4 --csv_file /data/yinzi/babyview/transcripts/Babyview_Main/00230001/00230001_GX010002_10.30.2023-11.05.2023_11.05.2023-11:00am.csv --save_root_dir=/home/yinzi/workspace/output_root

```

#### Step2: Run CLIP on all the frames

```
cd ~/workspace/clip-alignment/babyview_exps
conda activate torch
python batch_clip_on_all_frames.py --video_file /data/yinzi/babyview/Babyview_Main/00230001/00230001_GX010002_10.30.2023-11.05.2023_11.05.2023-11:00am.MP4 --csv_file /data/yinzi/babyview/transcripts/Babyview_Main/00230001/00230001_GX010002_10.30.2023-11.05.2023_11.05.2023-11:00am.csv --save_root_dir=/home/yinzi/workspace/output_root --frame_batch_size 512 --clip_batch_size 256 --prefetch 100
```


### Run CLIP on all videos(Batch mode)

```
cd ~/workspace/clip-alignment/babyview_exps
conda activate torch
python all_videos_clip_batch.py --babyview_folder /data/yinzi/babyview/Babyview_Main --babyview_transcript_folder /data/yinzi/babyview/transcripts/Babyview_Main  --output_root_dir /data/yinzi/babyview/all_clip_results 
```

