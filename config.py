"""
Configuration file for Video Anonymization Project
"""
import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # FDH Dataset settings
    FDH_DATA_DIR = DATA_DIR / "fdh"
    FDH_IMAGES_DIR = FDH_DATA_DIR / "images"
    FDH_ANNOTATIONS_DIR = FDH_DATA_DIR / "annotations"
    
    # Video settings
    VIDEO_HEIGHT = 160
    VIDEO_WIDTH = 288
    FPS = 30
    
    # Model settings
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Training settings
    NUM_EPOCHS = 100
    SAVE_INTERVAL = 10
    LOG_INTERVAL = 100
    
    # GAN settings
    LATENT_DIM = 512
    GENERATOR_FEATURES = 64
    DISCRIMINATOR_FEATURES = 64
    
    # Temporal consistency settings
    TEMPORAL_WINDOW_SIZE = 5
    CONSISTENCY_LAMBDA = 10.0
    
    # Action recognition settings
    ACTION_NUM_CLASSES = 101  # UCF-101 or similar
    ACTION_SEQUENCE_LENGTH = 16
    
    # Privacy evaluation settings
    FACE_RECOGNITION_THRESHOLD = 0.6
    
    # Device settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR, 
                        cls.FDH_DATA_DIR, cls.FDH_IMAGES_DIR, cls.FDH_ANNOTATIONS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True) 