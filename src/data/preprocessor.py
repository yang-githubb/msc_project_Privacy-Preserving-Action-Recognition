"""
Data Preprocessor for Video Anonymization
Handles pose extraction, temporal consistency, and data preparation
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2
from PIL import Image

class VideoPreprocessor:
    """Preprocessor for video sequences with pose data"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.FDH_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.sequence_length = config.SEQUENCE_LENGTH
        self.image_size = config.IMAGE_SIZE
        
        # Create processed data directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_sequences(self, sequences_file: str = "video_sequences.json"):
        """
        Preprocess video sequences for training
        
        Args:
            sequences_file: Path to sequences metadata file
        """
        print("Preprocessing video sequences...")
        
        sequences_path = self.data_dir / sequences_file
        if not sequences_path.exists():
            print(f"Sequences file not found: {sequences_path}")
            return
        
        # Load sequences
        with open(sequences_path, 'r') as f:
            sequences_data = json.load(f)
        
        sequences = sequences_data["sequences"]
        processed_sequences = []
        
        for sequence in tqdm(sequences, desc="Processing sequences"):
            processed_seq = self._process_single_sequence(sequence)
            if processed_seq is not None:
                processed_sequences.append(processed_seq)
        
        # Save processed sequences
        processed_path = self.processed_dir / "processed_sequences.json"
        with open(processed_path, 'w') as f:
            json.dump({
                "total_sequences": len(processed_sequences),
                "sequence_length": self.sequence_length,
                "image_size": self.image_size,
                "sequences": processed_sequences
            }, f, indent=2)
        
        print(f"Processed {len(processed_sequences)} sequences")
    
    def _process_single_sequence(self, sequence: Dict) -> Optional[Dict]:
        """Process a single video sequence"""
        try:
            frames = sequence["frames"]
            
            # Extract pose data for each frame
            pose_data = []
            temporal_features = []
            
            for frame in frames:
                # Extract keypoints
                keypoints = np.array(frame["keypoints"])
                
                # Normalize keypoints to [0, 1]
                keypoints_normalized = self._normalize_keypoints(keypoints)
                
                # Extract pose features
                pose_features = self._extract_pose_features(keypoints_normalized)
                
                pose_data.append(pose_features)
                
                # Extract temporal features (simple motion vectors)
                if len(pose_data) > 1:
                    motion = pose_features - pose_data[-2]
                    temporal_features.append(motion)
                else:
                    temporal_features.append(np.zeros_like(pose_features))
            
            # Convert to numpy arrays
            pose_data = np.array(pose_data)
            temporal_features = np.array(temporal_features)
            
            # Ensure sequence length
            if len(pose_data) != self.sequence_length:
                # Pad or truncate to match sequence length
                pose_data = self._pad_sequence(pose_data, self.sequence_length)
                temporal_features = self._pad_sequence(temporal_features, self.sequence_length)
            
            return {
                "sequence_id": sequence["sequence_id"],
                "pose_data": pose_data.tolist(),
                "temporal_features": temporal_features.tolist(),
                "metadata": {
                    "original_length": len(frames),
                    "processed_length": self.sequence_length
                }
            }
            
        except Exception as e:
            print(f"Error processing sequence {sequence.get('sequence_id', 'unknown')}: {e}")
            return None
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints to [0, 1] range"""
        # Extract x, y coordinates (first two elements of each keypoint)
        coords = keypoints[:, :2]
        
        # Normalize to [0, 1] based on image dimensions
        coords[:, 0] /= self.image_size[0]  # x / width
        coords[:, 1] /= self.image_size[1]  # y / height
        
        # Clip to [0, 1]
        coords = np.clip(coords, 0, 1)
        
        return coords
    
    def _extract_pose_features(self, keypoints: np.ndarray) -> np.ndarray:
        """Extract pose features from keypoints"""
        # Flatten keypoints
        features = keypoints.flatten()
        
        # Add additional pose features
        # 1. Center of mass
        center_x = np.mean(keypoints[:, 0])
        center_y = np.mean(keypoints[:, 1])
        
        # 2. Pose scale (distance from center)
        distances = np.sqrt((keypoints[:, 0] - center_x)**2 + (keypoints[:, 1] - center_y)**2)
        scale = np.mean(distances)
        
        # 3. Pose orientation (principal component)
        if len(keypoints) > 1:
            # Simple orientation based on shoulder line
            if len(keypoints) >= 6:  # Has shoulder keypoints
                left_shoulder = keypoints[5]  # left_shoulder
                right_shoulder = keypoints[6]  # right_shoulder
                orientation = np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                       right_shoulder[0] - left_shoulder[0])
            else:
                orientation = 0.0
        else:
            orientation = 0.0
        
        # Combine all features
        additional_features = np.array([center_x, center_y, scale, orientation])
        features = np.concatenate([features, additional_features])
        
        return features
    
    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate sequence to target length"""
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        elif current_length > target_length:
            # Truncate
            return sequence[:target_length]
        else:
            # Pad with last frame
            padding = np.tile(sequence[-1:], (target_length - current_length, 1))
            return np.concatenate([sequence, padding])
    
    def create_training_batches(self, batch_size: int = 8) -> List[Dict]:
        """
        Create training batches from processed sequences
        
        Args:
            batch_size: Number of sequences per batch
            
        Returns:
            List of batch dictionaries
        """
        print("Creating training batches...")
        
        processed_path = self.processed_dir / "processed_sequences.json"
        if not processed_path.exists():
            print("Processed sequences not found. Run preprocess_sequences first.")
            return []
        
        # Load processed sequences
        with open(processed_path, 'r') as f:
            data = json.load(f)
        
        sequences = data["sequences"]
        
        # Create batches
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Prepare batch data
            batch_pose_data = []
            batch_temporal_data = []
            batch_ids = []
            
            for seq in batch_sequences:
                batch_pose_data.append(seq["pose_data"])
                batch_temporal_data.append(seq["temporal_features"])
                batch_ids.append(seq["sequence_id"])
            
            # Convert to numpy arrays
            batch_pose_data = np.array(batch_pose_data)
            batch_temporal_data = np.array(batch_temporal_data)
            
            batches.append({
                "pose_data": batch_pose_data,
                "temporal_data": batch_temporal_data,
                "sequence_ids": batch_ids,
                "batch_size": len(batch_sequences)
            })
        
        # Save batch metadata
        batch_metadata = {
            "total_batches": len(batches),
            "batch_size": batch_size,
            "total_sequences": len(sequences)
        }
        
        batch_metadata_path = self.processed_dir / "batch_metadata.json"
        with open(batch_metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        print(f"Created {len(batches)} training batches")
        return batches
    
    def get_preprocessing_stats(self) -> Dict:
        """Get statistics about preprocessing"""
        stats = {
            "processed_dir": str(self.processed_dir),
            "sequence_length": self.sequence_length,
            "image_size": self.image_size
        }
        
        # Check processed sequences
        processed_path = self.processed_dir / "processed_sequences.json"
        if processed_path.exists():
            with open(processed_path, 'r') as f:
                data = json.load(f)
                stats["num_processed_sequences"] = data["total_sequences"]
        
        # Check batch metadata
        batch_metadata_path = self.processed_dir / "batch_metadata.json"
        if batch_metadata_path.exists():
            with open(batch_metadata_path, 'r') as f:
                batch_data = json.load(f)
                stats.update(batch_data)
        
        return stats

def main():
    """Main function for data preprocessing"""
    from config import Config
    
    config = Config()
    config.create_directories()
    
    preprocessor = VideoPreprocessor(config)
    
    # Preprocess sequences
    preprocessor.preprocess_sequences()
    
    # Create training batches
    batches = preprocessor.create_training_batches(batch_size=8)
    
    # Print preprocessing stats
    stats = preprocessor.get_preprocessing_stats()
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 