import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Name: Pranav  Varanasi
# UT EID: ptv247

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    # verify view index
    if view_index >= len(info["detections"]):
        return []
    
    detections = info["detections"][view_index]
    kart_names = info.get("karts", [])  

    # scale by image dims
    scale_x = img_width / ORIGINAL_WIDTH  
    scale_y = img_height / ORIGINAL_HEIGHT 

    karts = []
    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        # if not 1 its not a kart
        if int(class_id) != 1:  
            continue

        # scale by dims
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        width = x2_scaled - x1_scaled
        height = y2_scaled - y1_scaled

        # get rid of to small bounding boxes
        if width < min_box_size or height < min_box_size:
            continue
        
        # check out of bounds
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        track_id_int = int(track_id)
        
        # look up kart name from dict
        if 0 <= track_id_int < len(kart_names):
            kart_name = kart_names[track_id_int]
        else:
            kart_name = f"kart_{track_id_int}"
        
        karts.append({
            "instance_id": track_id_int,
            "kart_name": kart_name,
            "center": (center_x, center_y)
        })

    # ego kart is closest to center
    if karts:
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # use euclid distance formula
        closest_kart = min(karts, key=lambda k: 
            ((k["center"][0] - img_center_x)**2 + (k["center"][1] - img_center_y)**2)**0.5)
        
        # identify center karts
        for kart in karts:
            kart["is_center_kart"] = (kart["instance_id"] == closest_kart["instance_id"])

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "Unknown") # should be word track not track_name


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # find ego kart
    ego_kart = None
    for kart in karts:
        if kart.get("is_center_kart", False):
            ego_kart = kart
            break
    
    # skip view if no ego kart
    if ego_kart is None:
        return [] 

    # get image path
    split_name = Path(info_path).parent.name
    base_name = Path(info_path).stem.replace("_info", "")
    image_file = f"{split_name}/{base_name}_{view_index:02d}_im.jpg"

    qa_pairs = []
    
    # 1: what kart is the ego car?
    qa_pairs.append({
        "image_file": image_file,
        "question": "What kart is the ego car?",
        "answer": ego_kart["kart_name"]
    })
    
    # 2: how many karts are there in the scenario?
    qa_pairs.append({
        "image_file": image_file,
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })
    
    # 3: what track is this?
    qa_pairs.append({
        "image_file": image_file,
        "question": "What track is this?",
        "answer": track_name
    })

    # Get center of ego kart
    ego_center_x, ego_center_y = ego_kart["center"]

    # 4: Relative questions
    for kart in karts:
        if kart["instance_id"] == ego_kart["instance_id"]:
            continue
            
        kart_center_x, kart_center_y = kart["center"]
        dx = kart_center_x - ego_center_x
        dy = kart_center_y - ego_center_y
        
        # left/right question
        if dx <= 0: 
            lr_answer = "left"
        else:
            lr_answer = "right"
            
        qa_pairs.append({
            "image_file": image_file,
            "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
            "answer": lr_answer
        })
        
        # front/back question  
        if dy <= 0:
            fb_answer = "front"
        else:
            fb_answer = "back"
            
        qa_pairs.append({
            "image_file": image_file,
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": fb_answer
        })
    
    # 4: relative positions from ego
    for kart in karts:
        if kart["instance_id"] == ego_kart["instance_id"]:
            continue  
            
        kart_center_x, kart_center_y = kart["center"]
        dx = kart_center_x - ego_center_x
        dy = kart_center_y - ego_center_y
        
        # get rel positions
        position_parts = []
        
        if dy < 0:
            position_parts.append("front")
        elif dy > 0:
            position_parts.append("back")
            
        if dx < 0:
            position_parts.append("left")  
        elif dx > 0:
            position_parts.append("right")
        
        # if found a close enough position, add
        if position_parts: 
            qa_pairs.append({
                "image_file": image_file,
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": " and ".join(position_parts)
            })

    # 5: count karts
    left_count = right_count = front_count = back_count = 0
    
    for kart in karts:
        if kart["instance_id"] == ego_kart["instance_id"]:
            continue
            
        kart_center_x, kart_center_y = kart["center"]
        dx = kart_center_x - ego_center_x
        dy = kart_center_y - ego_center_y
        
        # get karts in each direction
        if dy < 0:
            front_count += 1
        elif dy > 0:
            back_count += 1
            
        if dx < 0:
            left_count += 1
        elif dx > 0:
            right_count += 1
    
    # save the counting question
    for direction, count in [("left", left_count), ("right", right_count), 
                           ("front", front_count), ("back", back_count)]:
        qa_pairs.append({
            "image_file": image_file,
            "question": f"How many karts are to the {direction} of the ego car?",
            "answer": str(count)
        })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


# Generate funciton to save data
def generate(split: str = "train", output_file: str = None, num_views: int = 5):
    """
    Generate and save QA pairs for all files in a dataset split.
    """
    split_dir = Path("data") / split
    
    # json output dir
    if output_file is None:
        output_file = split_dir / "all_qa_pairs.json"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_qa_pairs = []
    info_files = sorted(split_dir.glob("*_info.json"))
       
    for info_path in info_files:
        for view_index in range(num_views):
            # check if file exists
            base_name = info_path.stem.replace("_info", "")
            image_path = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            
            if not image_path.exists():
                continue 
            
            qa_pairs = generate_qa_pairs(str(info_path), view_index)
            
            # if no pairs, skip
            if len(qa_pairs) == 0:
                continue
            
            all_qa_pairs.extend(qa_pairs)
    
    # save pairs out
    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
     

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate, 
    })


if __name__ == "__main__":
    main()
