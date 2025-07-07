# Privacy-Preserving Video Anonymization with Pose-Conditioned GANs

**MSc Project: Pose-Conditioned GANs with Temporal Consistency for Privacy-Preserving Video Anonymization**

## Project Overview

This project implements a privacy-preserving video anonymization system using pose-conditioned generative adversarial networks (GANs) with temporal consistency. The system extracts pose information from videos and generates anonymized versions while preserving action recognition capabilities.

## Project Structure

```
s2737744/
├── DensePose/                    # DensePose implementation (Facebook Research)
├── kinetics-dataset/             # Kinetics dataset processing scripts
├── vid2vid/                      # Vid2Vid framework for video generation
├── src/
│   └── data/
│       ├── pose_extractor.py     # MediaPipe pose extraction (current)
│       └── kinetics_pose_pipeline.py  # Complete pipeline
├── data/
│   └── kinetics_processed/       # Processed Kinetics data
│       ├── frames/               # Extracted video frames
│       ├── pose_data/            # Extracted pose keypoints
│       ├── metadata.json         # Video metadata
│       └── summary.json          # Processing summary
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Components

### 1. DensePose Integration
- **Location**: `DensePose/` directory
- **Purpose**: Extract dense pose UV maps for privacy-preserving representation
- **Status**: Ready for integration (needs Detectron2 setup)

### 2. Kinetics Dataset Processing
- **Location**: `kinetics-dataset/` directory
- **Purpose**: Process Kinetics-400 videos for training and evaluation
- **Status**: Working pipeline for frame extraction

### 3. Vid2Vid Framework
- **Location**: `vid2vid/` directory
- **Purpose**: Generate anonymized videos from pose information
- **Status**: Ready for training with pose maps

### 4. Pose Extraction Pipeline
- **Current**: MediaPipe pose keypoints (`src/data/pose_extractor.py`)
- **Target**: DensePose UV maps for richer pose representation
- **Status**: Working with MediaPipe, ready to switch to DensePose

## Current Status

✅ **Completed:**
- Kinetics video frame extraction
- MediaPipe pose keypoint extraction
- Pose map generation for Vid2Vid
- Basic pipeline integration

🔄 **In Progress:**
- Switching from MediaPipe to DensePose
- Detectron2 installation and setup
- DensePose UV map extraction

📋 **Next Steps:**
1. Install Detectron2 with DensePose support
2. Extract DensePose UV maps from Kinetics frames
3. Train Vid2Vid model with DensePose data
4. Evaluate privacy preservation and action recognition
5. Implement temporal consistency mechanisms

## Privacy Preservation Approach

This project achieves privacy preservation by:

1. **Pose Extraction**: Converting video frames to pose representations (keypoints or UV maps)
2. **Identity Removal**: Eliminating facial features, clothing, and background details
3. **Action Preservation**: Maintaining body posture and movement information
4. **Video Generation**: Creating anonymized videos using Vid2Vid from pose data

## Usage

### Current Pipeline (MediaPipe)
```bash
# Extract pose keypoints from existing frames
python src/data/pose_extractor.py --input_dir data/kinetics_processed/frames --output_dir data/kinetics_processed/pose_data --save_visualizations
```

### Target Pipeline (DensePose)
```bash
# TODO: DensePose extraction commands will be added here
```

## Dependencies

- Python 3.8+
- PyTorch 2.4.1
- MediaPipe (current pose extraction)
- Detectron2 + DensePose (target pose extraction)
- Vid2Vid framework
- OpenCV, NumPy, etc.

## Research Contributions

This project contributes to the field of privacy-preserving computer vision by:

1. **Novel Integration**: Combining DensePose with Vid2Vid for video anonymization
2. **Temporal Consistency**: Ensuring smooth transitions in generated videos
3. **Action Recognition Evaluation**: Measuring utility preservation
4. **Privacy Metrics**: Quantifying privacy protection levels
5. **Kinetics Dataset**: Using large-scale action recognition dataset

## License

This project is for academic research purposes. 