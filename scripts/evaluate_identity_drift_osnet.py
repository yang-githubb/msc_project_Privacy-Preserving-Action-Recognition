import os
import torch
from torchreid.utils import FeatureExtractor
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
import cv2
from pathlib import Path
import mediapipe as mp

# 1. Load OSNet pre-trained on Market1501
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 2. Initialize MediaPipe Pose for precise person segmentation
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

# 3. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def create_person_mask_with_pose(image):
    """Create a precise mask of the person using MediaPipe pose estimation"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(image_rgb)
    
    if results.pose_landmarks is None:
        return None, None
    
    # Get segmentation mask
    if results.segmentation_mask is not None:
        # Convert segmentation mask to binary mask
        mask = results.segmentation_mask > 0.1
        mask = mask.astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours to get bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (main person)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding to ensure we capture the full person
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return mask, (x, y, w, h)
    
    return None, None

def extract_embeddings_from_frames(frames_dir, max_frames=None):
    """Extract embeddings from frames using pose estimation for precise person cropping"""
    frames_dir = Path(frames_dir)
    frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    embeddings = []
    frame_names = []
    
    for frame_path in frame_files:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
            
        # Get person mask and bounding box using pose estimation
        mask, bbox = create_person_mask_with_pose(img)
        
        if mask is None or bbox is None:
            continue
            
        x, y, w, h = bbox
        
        # Crop the person region
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
            
        # Apply the mask to remove background
        mask_crop = mask[y:y+h, x:x+w]
        
        # Create a 4-channel image (RGBA) with transparency
        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
        crop_rgba[:, :, 3] = mask_crop  # Set alpha channel to mask
        
        # Convert to PIL and preprocess (PIL will handle the alpha channel)
        crop_pil = Image.fromarray(crop_rgba, 'RGBA')
        
        # Convert RGBA to RGB by compositing on white background
        # This ensures no background artifacts
        white_bg = Image.new('RGB', crop_pil.size, (255, 255, 255))
        white_bg.paste(crop_pil, mask=crop_pil.split()[-1])  # Use alpha channel as mask
        crop_pil = white_bg
        
        tensor = preprocess(crop_pil).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            emb = extractor(tensor)
        embeddings.append(emb.squeeze(0).cpu().numpy())
        frame_names.append(frame_path.name)
    
    return np.stack(embeddings), frame_names

def extract_embeddings(crop_dir):
    """Original function for processing pre-cropped images"""
    crops = sorted([f for f in os.listdir(crop_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    embeddings = []
    for fname in crops:
        img = Image.open(os.path.join(crop_dir, fname)).convert('RGB')
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = extractor(tensor)
        embeddings.append(emb.squeeze(0).cpu().numpy())
    return np.stack(embeddings), crops

def compute_consecutive_cosine_sim(embeddings):
    from numpy import dot
    from numpy.linalg import norm
    sims = []
    for i in range(len(embeddings)-1):
        a, b = embeddings[i], embeddings[i+1]
        sim = dot(a, b) / (norm(a) * norm(b))
        sims.append(sim)
    return np.array(sims)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate identity drift using OSNet with pose-based person cropping.")
    parser.add_argument('--crop_dir', type=str, help='Directory with pre-cropped person images')
    parser.add_argument('--frames_dir', type=str, help='Directory with full frames to detect and crop persons using pose estimation')
    parser.add_argument('--max_frames', type=int, default=None, help='Max number of frames to process')
    args = parser.parse_args()

    if args.frames_dir:
        # Use frame-based person detection with pose estimation for precise cropping
        embeddings, frame_names = extract_embeddings_from_frames(args.frames_dir, args.max_frames)
        print(f"Processed {len(frame_names)} frames with pose-based person cropping.")
    elif args.crop_dir:
        # Use pre-cropped images (original functionality)
        embeddings, crops = extract_embeddings(args.crop_dir)
        print(f"Processed {len(crops)} pre-cropped images.")
    else:
        print("Error: Must specify either --crop_dir or --frames_dir")
        exit(1)

    sims = compute_consecutive_cosine_sim(embeddings)
    print(f"Evaluated {len(sims)} consecutive frame pairs from {len(embeddings)} crops.")
    print(f"Mean cosine similarity: {sims.mean():.4f}")
    print(f"Variance: {sims.var():.4f}")
    print(f"Min: {sims.min():.4f}, Max: {sims.max():.4f}") 