"""
FDH Dataset Downloader and Preprocessor
Downloads and prepares FDH dataset subset for video anonymization
"""
import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict, Optional, Any

class FDHDownloader:
    """Downloader for FDH dataset subset"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.FDH_DATA_DIR
        self.images_dir = config.FDH_IMAGES_DIR
        self.annotations_dir = config.FDH_ANNOTATIONS_DIR
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def download_fdh_subset(self, num_images: int = 10000, random_seed: int = 42):
        """
        Download a subset of FDH dataset
        
        Args:
            num_images: Number of images to download (max 50K for 8-week timeline)
            random_seed: Random seed for reproducible subset selection
        """
        print(f"Downloading FDH dataset subset ({num_images} images)...")
        
        # Set random seed for reproducible selection
        random.seed(random_seed)
        
        # For demonstration, we'll create synthetic data
        # In practice, you would download from the actual FDH dataset
        self._create_synthetic_fdh_subset(num_images)
        
        print(f"FDH subset created successfully in {self.data_dir}")
    
    def _create_synthetic_fdh_subset(self, num_images: int):
        """Create synthetic FDH subset for development"""
        print("Creating synthetic FDH subset for development...")
        
        # Create synthetic image metadata
        image_metadata = []
        
        for i in tqdm(range(num_images), desc="Creating synthetic data"):
            # Generate synthetic image info
            image_info = {
                "id": f"fdh_{i:06d}",
                "filename": f"fdh_{i:06d}.jpg",
                "width": 288,
                "height": 160,
                "keypoints": self._generate_synthetic_keypoints(),
                "segmentation": self._generate_synthetic_segmentation(),
                "bbox": self._generate_synthetic_bbox()
            }
            
            image_metadata.append(image_info)
            
            # Save annotation
            annotation_path = self.annotations_dir / f"fdh_{i:06d}.json"
            with open(annotation_path, 'w') as f:
                json.dump(image_info, f, indent=2)
        
        # Save metadata index
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "total_images": num_images,
                "images": image_metadata
            }, f, indent=2)
        
        print(f"Created {num_images} synthetic FDH images")
    
    def _generate_synthetic_keypoints(self) -> List[List[float]]:
        """Generate synthetic 17 keypoints"""
        keypoints = []
        
        # COCO format: [x, y, visibility]
        # 17 keypoints: nose, left_eye, right_eye, left_ear, right_ear,
        # left_shoulder, right_shoulder, left_elbow, right_elbow,
        # left_wrist, right_wrist, left_hip, right_hip,
        # left_knee, right_knee, left_ankle, right_ankle
        
        # Generate realistic keypoint positions
        base_x, base_y = 144, 80  # Center of image
        
        for i in range(17):
            # Add some randomness to keypoint positions
            x = base_x + random.uniform(-50, 50)
            y = base_y + random.uniform(-40, 40)
            visibility = random.choice([0, 1, 2])  # 0=invisible, 1=occluded, 2=visible
            
            keypoints.append([x, y, visibility])
        
        return keypoints
    
    def _generate_synthetic_segmentation(self) -> List[List[float]]:
        """Generate synthetic segmentation polygon"""
        # Simple rectangular segmentation
        return [
            [100, 60], [188, 60], [188, 140], [100, 140]
        ]
    
    def _generate_synthetic_bbox(self) -> List[float]:
        """Generate synthetic bounding box [x, y, width, height]"""
        return [100, 60, 88, 80]
    
    def create_video_sequences(self, sequence_length: int = 16, num_sequences: int = 100):
        """
        Create synthetic video sequences from FDH images
        
        Args:
            sequence_length: Number of frames in each sequence
            num_sequences: Number of video sequences to create
        """
        print(f"Creating {num_sequences} video sequences of length {sequence_length}...")
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            print("Metadata not found. Please run download_fdh_subset first.")
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create video sequences
        sequences = []
        total_images = len(metadata["images"])
        
        for i in range(num_sequences):
            # Select random starting point
            start_idx = random.randint(0, total_images - sequence_length)
            
            sequence = {
                "sequence_id": f"seq_{i:04d}",
                "length": sequence_length,
                "frames": []
            }
            
            # Add frames to sequence
            for j in range(sequence_length):
                frame_idx = start_idx + j
                frame_info = metadata["images"][frame_idx]
                
                sequence["frames"].append({
                    "image_id": frame_info["id"],
                    "filename": frame_info["filename"],
                    "keypoints": frame_info["keypoints"],
                    "segmentation": frame_info["segmentation"],
                    "bbox": frame_info["bbox"]
                })
            
            sequences.append(sequence)
        
        # Save sequences metadata
        sequences_path = self.data_dir / "video_sequences.json"
        with open(sequences_path, 'w') as f:
            json.dump({
                "total_sequences": num_sequences,
                "sequence_length": sequence_length,
                "sequences": sequences
            }, f, indent=2)
        
        print(f"Created {num_sequences} video sequences")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the downloaded dataset"""
        info = {
            "data_dir": str(self.data_dir),
            "images_dir": str(self.images_dir),
            "annotations_dir": str(self.annotations_dir)
        }
        
        # Count files
        if self.images_dir.exists():
            image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
            info["num_images"] = len(image_files)
        
        if self.annotations_dir.exists():
            annotation_files = list(self.annotations_dir.glob("*.json"))
            info["num_annotations"] = len(annotation_files)
        
        # Check for sequences
        sequences_path = self.data_dir / "video_sequences.json"
        if sequences_path.exists():
            with open(sequences_path, 'r') as f:
                sequences_data = json.load(f)
                info["num_sequences"] = sequences_data["total_sequences"]
                info["sequence_length"] = sequences_data["sequence_length"]
        
        return info

def main():
    """Main function for dataset download"""
    from config import Config
    
    config = Config()
    config.create_directories()
    
    downloader = FDHDownloader(config)
    
    # Download FDH subset (10K images for 8-week timeline)
    downloader.download_fdh_subset(num_images=10000)
    
    # Create video sequences
    downloader.create_video_sequences(sequence_length=16, num_sequences=100)
    
    # Print dataset info
    info = downloader.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 