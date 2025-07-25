# Re-Identification (Re-ID) Evaluation Pipeline

This folder contains scripts for evaluating person re-identification performance and identity leakage after anonymization.

## Pipeline Overview

The re-ID evaluation pipeline consists of several stages:

1. **Data Preparation**: Extract and organize person crops from videos
2. **Feature Extraction**: Use pre-trained Re-ID models to extract identity features
3. **Privacy Evaluation**: Measure identity leakage after anonymization
4. **Identity Drift Analysis**: Analyze temporal consistency of identities

## Scripts

### Data Preparation

#### `extract_person_crops_yolov5.py`
- **Purpose**: Extract person crops from video frames using YOLOv5 detection
- **Input**: Video frames or video files
- **Output**: Cropped person images for Re-ID evaluation
- **Usage**: 
  ```bash
  python extract_person_crops_yolov5.py --input_dir /path/to/frames --output_dir /path/to/crops
  ```

#### `organize_reid_dataset.py`
- **Purpose**: Organize extracted crops into Re-ID dataset format
- **Input**: Cropped person images
- **Output**: Organized dataset with proper ID/camera structure
- **Usage**:
  ```bash
  python organize_reid_dataset.py --input_dir /path/to/crops --output_dir /path/to/reid_dataset
  ```

### Privacy Evaluation

#### `evaluate_reid_osnet.py` ⭐ **MAIN SCRIPT**
- **Purpose**: Evaluate identity leakage after anonymization using OSNet
- **Setup**: 
  - Query set: ANONYMIZED frames
  - Gallery set: ORIGINAL frames
- **Metrics**: Rank-1 accuracy and mAP
- **Privacy Interpretation**:
  - **Low Rank-1 (< 25%)** = Good privacy (identity obfuscated) ✅
  - **High Rank-1 (≥ 25%)** = Poor privacy (identity preserved) ⚠️
- **Usage**:
  ```bash
  python evaluate_reid_osnet.py --query_dir /path/to/anonymized --gallery_dir /path/to/original
  ```

### Identity Drift Analysis

#### `evaluate_identity_drift_osnet.py`
- **Purpose**: Analyze identity drift over time using OSNet features
- **Metrics**: Cosine similarity, Euclidean distance, Manhattan distance, Pearson correlation
- **Output**: Temporal consistency analysis and plots
- **Usage**:
  ```bash
  python evaluate_identity_drift_osnet.py --frames_dir /path/to/frames --plot_path drift_analysis.png
  ```

#### `evaluate_identity_drift_agw.py`
- **Purpose**: Analyze identity drift using AGW (Attribute-Guided Network) features
- **Similar to**: `evaluate_identity_drift_osnet.py` but uses AGW model
- **Usage**:
  ```bash
  python evaluate_identity_drift_agw.py --frames_dir /path/to/frames --plot_path drift_analysis_agw.png
  ```

## Evaluation Workflow

### 1. Privacy Assessment
```bash
# Step 1: Extract person crops from original and anonymized videos
python extract_person_crops_yolov5.py --input_dir original_frames --output_dir original_crops
python extract_person_crops_yolov5.py --input_dir anonymized_frames --output_dir anonymized_crops

# Step 2: Organize into Re-ID format
python organize_reid_dataset.py --input_dir original_crops --output_dir reid_gallery
python organize_reid_dataset.py --input_dir anonymized_crops --output_dir reid_query

# Step 3: Evaluate privacy (identity leakage)
python evaluate_reid_osnet.py --query_dir reid_query --gallery_dir reid_gallery
```

### 2. Identity Drift Analysis
```bash
# Analyze temporal consistency of identities
python evaluate_identity_drift_osnet.py --frames_dir anonymized_frames --plot_path identity_drift.png
```

## Expected Results

### Privacy Evaluation Results
- **Rank-1 Accuracy**: Percentage of anonymized queries correctly matched to original identities
- **mAP**: Mean Average Precision across all ranks
- **Privacy Level**: 
  - 🟢 EXCELLENT (< 10%)
  - 🟡 GOOD (10-25%)
  - 🟠 MODERATE (25-50%)
  - 🔴 POOR (> 50%)

### Identity Drift Results
- **Cosine Similarity**: Higher values indicate more consistent identities
- **Euclidean/Manhattan Distance**: Lower values indicate more consistent identities
- **LPIPS/SSIM**: Image-based consistency metrics

## Dependencies

- `torchreid`: For OSNet and AGW models
- `torch`: PyTorch framework
- `opencv-python`: Image processing
- `numpy`: Numerical computations
- `matplotlib`: Plotting results
- `lpips`: Learned Perceptual Image Patch Similarity
- `mediapipe`: Pose detection for person segmentation

## Notes

- The main privacy evaluation script (`evaluate_reid_osnet.py`) is designed to measure how well anonymization prevents re-identification
- Low Rank-1 accuracy indicates successful identity obfuscation
- High Rank-1 accuracy suggests the anonymization method needs improvement
- Identity drift analysis helps understand temporal consistency of the anonymization 