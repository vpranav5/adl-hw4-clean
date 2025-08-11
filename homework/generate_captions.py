from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info, generate_qa_pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

# Name: Pranav Teja Varanasi
# UT EID: ptv247

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # find ego kart
    ego = None
    for k in karts:
        if k.get("is_center_kart") or k["instance_id"] == 0:
            ego = k
            break
    
    if ego is None:
        return []  
    
    captions = []
    
    # first 3 captions
    captions.append(f"The track is {track_name}.")
    captions.append(f"There are {len(karts)} karts in the scenario.")
    
    captions.append(f"{ego['kart_name']} is the ego car.")
    
    # 4: relative position captions
    MARGIN = 6
    
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        
        position_parts = []
        if dy <= -MARGIN:
            position_parts.append("in front")
        elif dy >= MARGIN:
            position_parts.append("behind")
        
        if dx <= -MARGIN:
            position_parts.append("to the left")
        elif dx >= MARGIN:
            position_parts.append("to the right")
        
        if position_parts:
            position = " and ".join(position_parts)
            captions.append(f"{k['kart_name']} is {position} of the ego car.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate(split: str = "train", output_file: str = None, num_views: int = None):
  
    split_dir = Path("data") / split
    
    if output_file is None:
        output_file = split_dir / "all_captions.json"
    else:
        output_file = Path(output_file)
    
    all_caption_pairs = []
    info_files = sorted(split_dir.glob("*_info.json"))
    
    
    # loop all files
    for info_path in info_files:
        for view_index in range(num_views):
            # check for img
            base_name = info_path.stem.replace("_info", "")
            image_path = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            
            if not image_path.exists():
                continue
            
            # generate captions
            captions = generate_caption(str(info_path), view_index)
            
            # skip if no captions
            if not captions:
                continue
            
            # get caption pairs
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            
            for caption in captions:
                all_caption_pairs.append({
                    "image_file": image_file,
                    "caption": caption
                })
            
    # save all captions
    with open(output_file, "w") as f:
        json.dump(all_caption_pairs, f, indent=2)
    
    return all_caption_pairs

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
