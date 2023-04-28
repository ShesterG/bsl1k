"""
Demo on a single video input.
Example usage:
    python demo.py
    python demo.py --topk 3 --confidence 0
"""
import os
import sys
import math
import pickle as pkl
import gzip
import shutil
import argparse
from pathlib import Path

import cv2
import wget
import numpy as np
import pandas as pd
import torch
import scipy.special
from pathlib import Path
from beartype import beartype
from zsvision.zs_utils import BlockTimer
from tqdm import tqdm

sys.path.append("..")
import models
from utils.misc import to_torch
from utils.imutils import im_to_numpy, im_to_torch, resize_generic
from utils.transforms import color_normalize

last_path = None
cap = None
cap_height = None
cap_width = None
cap_fps = None

@beartype
def load_rgb_video(start, end, video_path: str, fps: int, lookback: int, chunk_size: int = 10000):
    """
    Load frames of a video using cv2 (fetch from provided URL if file is not found
    at given location).
    """
    
    global last_path, cap, cap_width, cap_height, cap_fps
    if video_path != last_path:
      cap = cv2.VideoCapture(str(video_path))
      cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      cap_fps = cap.get(cv2.CAP_PROP_FPS)
    last_path = video_path

    rgb = []    
    start_framenum = int(start * cap_fps)
    end_framenum = int(end * cap_fps)
    number_of_frames = int((end - start) * cap_fps)
    frame_count = 0
    
    # Set the start time
    start_time = int(start*1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    # Read frames until the end time is reached
    #keep = True
    

    for fra in range(number_of_frames): 
        ret, frame = cap.read()
        if not ret:
            continue

        # BGR (OpenCV) to RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
                
    # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
    rgb_out = torch.stack(rgb).permute(1, 0, 2, 3)
    print(f"Loaded video {video_path} with {rgb_out.shape[1]} frames [{cap_height}hx{cap_width}w] res. "
        f"at {cap_fps}")
    yield rgb_out


@beartype
def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3), std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
        )
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb


@beartype
def fetch_from_url(url: str, dest_path: str):
    dest_path = Path(dest_path)
    if not dest_path.exists():
        try:
            print(f"Missing file at {dest_path}, downloading from {url} to {dest_path}")
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            wget.download(url, str(dest_path))
            assert dest_path.exists()
        except IOError as IOE:
            print(f"{IOE} (was not able to download file to {dest_path} please try to "
                  "download the file manually via the link on the README")
            raise IOE


@beartype
def load_model(
        checkpoint_path: str,
        checkpoint_url: str,
        num_classes: int,
        num_in_frames: int,
) -> torch.nn.Module:
    """Load pre-trained I3D checkpoint, put in eval mode.  Download checkpoint
    from url if not found locally.
    """
    fetch_from_url(url=checkpoint_url, dest_path=checkpoint_path)
    model = models.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@beartype
def sliding_windows(
        rgb: torch.Tensor,
        num_in_frames: int,
        stride: int,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)

model = None
def get_video_feature(
    start,
    end,
    video_path: str = '/net/cephfs/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_256/srf.2020-03-13_trim_2m.mp4',
    checkpoint_path: str = '/content/gbucketafrisign/I3DFT/models/bsl5k.pth.tar',
    checkpoint_url: str = 'https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar',
    fps: int = 30,
    num_classes: int = 5383,
    num_in_frames: int = 64,
    batch_size: int = 8,
    stride: int = 8,
):
    global model
    if model is None:
        with BlockTimer("Loading model"):
            model = load_model(
                checkpoint_path=checkpoint_path,
                checkpoint_url=checkpoint_url,
                num_classes=num_classes,
                num_in_frames=num_in_frames,
            )

    rgb_orig_gen = load_rgb_video(
        start=start,
        end=end,      
        video_path=video_path,
        fps=fps,
        lookback=(num_in_frames-stride),
    )
    feature_l = []
    for rgb_orig in rgb_orig_gen:
        with BlockTimer("Working on chunk"):
            # Prepare: resize/crop/normalize
            rgb_input = prepare_input(rgb_orig)
            print(rgb_input.size())
            # Sliding window
            rgb_slides, t_mid = sliding_windows(
                rgb=rgb_input,
                stride=stride,
                num_in_frames=num_in_frames,
            )
            print(rgb_slides.shape)
            # Number of windows/clips
            num_clips = rgb_slides.shape[0]
            # Group the clips into batches
            num_batches = math.ceil(num_clips / batch_size)
            raw_scores = np.empty((0, num_classes), dtype=float)
            features = []
            for b in range(num_batches):
                inp = rgb_slides[b * batch_size : (b + 1) * batch_size]
                # Forward pass
                out = model(inp)
                features.append(out['embds'].view(-1, out['embds'].shape[1]).cpu().detach().numpy())
            feature = np.concatenate(features)
            feature_l.append(feature)
            print(feature.shape)
    feature = np.concatenate(feature_l)
    print(feature.shape)

    return feature

if __name__ == "__main__":
    
    #ONLY EDIT THIS  
    bucket = '/content/gbucketafrisign/'
    model_feature_path = '/content/gbucketafrisign/I3Dfeatures/bsl5k/' 
    checkpoint_path = bucket+'I3DFT/models/bsl5k.pth.tar'
    index_path='/content/gbucketafrisign/JWSign/JWSign_lcv.dict'
    
    with gzip.open(index_path, "rb") as f:
        index = pkl.load(f)
    shester = 0

    language_to_test = ['ASL']
    #for la in index:
    for la in language_to_test:
      for vid in index[la]:
        video_path = bucket+'videos/'+la+'/'+vid+'.mp4'
        #cap = cv2.VideoCapture(video_path)
        for ver in index[la][vid]:
          feature_path = model_feature_path + la + '/'
          ver_dict = index[la][vid][ver] 
          feature_path_v = feature_path + ver_dict['verse_unique'] + ".npy"
          """
          if Path(feature_path_v).exists():
            print(f"verse exist") 
            continue
          """  
          print(f"new verse started")        
          start = float(ver_dict['verse_start'])
          end = float(ver_dict['verse_end'])
          feature = get_video_feature(start=start,end=end, video_path=video_path)          
          Path(feature_path).mkdir(exist_ok=True, parents=True)         
          np.save(feature_path_v, feature)
          shester = shester + 1
          print(f"verse {shester} done.")
        #cap.release()  
