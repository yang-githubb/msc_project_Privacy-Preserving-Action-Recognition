# Privacy-Preserving Video Anonymization with DeepPrivacy2

**MSc Project: Pose-Conditioned GANs with Temporal Consistency for Privacy-Preserving Video Anonymization**

## Project Overview

This project implements a comprehensive privacy-preserving video anonymization system using DeepPrivacy2, a state-of-the-art pose-conditioned generative adversarial network (GAN). The system generates anonymized videos while preserving action recognition capabilities and evaluates privacy protection through re-identification metrics.

## 🎯 Key Features

- **DeepPrivacy2 Integration**: Advanced pose-conditioned GAN for high-quality anonymization
- **Privacy Evaluation**: Comprehensive re-identification assessment pipeline
- **Action Recognition**: Maintains video utility for downstream tasks
- **Temporal Consistency**: Ensures smooth transitions in anonymized videos
- **Multi-Modal Evaluation**: Face detection, identity drift, and video quality metrics

## 📁 Project Structure

```
s2737744/
├── deep_privacy2/                # DeepPrivacy2 implementation
├── detectron2/                   # Detectron2 for pose detection
├── mmaction2/                    # MMAction2 for action recognition
├── scripts/
│   ├── re-id/                    # Re-identification evaluation pipeline
│   │   ├── evaluate_reid_osnet.py        # Main privacy evaluation script
│   │   ├── evaluate_identity_drift_osnet.py  # Identity drift analysis
│   │   ├── evaluate_identity_drift_agw.py    # AGW-based drift analysis
│   │   ├── extract_person_crops_yolov5.py    # Person crop extraction
│   │   ├── organize_reid_dataset.py          # Dataset organization
│   │   ├── visualize_identity_drift.py       # Drift visualization
│   │   └── README.md                         # Pipeline documentation
│   ├── evaluation_face_iden.py   # Face identification evaluation
│   ├── evaluation_vid_quality.py # Video quality assessment
│   ├── visualize_embedding_trajectory.py  # Embedding trajectory visualization
│   └── extract_frames.py         # Video frame extraction
├── data/                         # Dataset storage
├── config.py                     # Configuration settings
└── README.md                     # This file
```

## 🔐 Privacy Evaluation Pipeline

The project includes a comprehensive re-identification evaluation pipeline located in `scripts/re-id/`:

### Main Privacy Assessment
- **`evaluate_reid_osnet.py`**: Evaluates identity leakage after anonymization
  - Query set: ANONYMIZED frames
  - Gallery set: ORIGINAL frames
  - Metrics: Rank-1 accuracy and mAP
  - **Low Rank-1 (< 25%)** = Good privacy ✅
  - **High Rank-1 (≥ 25%)** = Poor privacy ⚠️

### Identity Drift Analysis
- **`evaluate_identity_drift_osnet.py`**: Temporal consistency analysis using OSNet
- **`evaluate_identity_drift_agw.py`**: Temporal consistency using AGW features
- **`visualize_identity_drift.py`**: Drift visualization and plotting

### Data Preparation
- **`extract_person_crops_yolov5.py`**: Person detection and cropping
- **`organize_reid_dataset.py`**: Dataset organization for Re-ID evaluation

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd s2737744

# Install dependencies (see individual component READMEs)
# - DeepPrivacy2: Follow deep_privacy2/README.md
# - Detectron2: Follow detectron2/README.md  
# - MMAction2: Follow mmaction2/README.md
```

### 2. Privacy Evaluation
```bash
# Extract person crops from original and anonymized videos
python scripts/re-id/extract_person_crops_yolov5.py --input_dir original_frames --output_dir original_crops
python scripts/re-id/extract_person_crops_yolov5.py --input_dir anonymized_frames --output_dir anonymized_crops

# Organize into Re-ID format
python scripts/re-id/organize_reid_dataset.py --input_dir original_crops --output_dir reid_gallery
python scripts/re-id/organize_reid_dataset.py --input_dir anonymized_crops --output_dir reid_query

# Evaluate privacy (identity leakage)
python scripts/re-id/evaluate_reid_osnet.py --query_dir reid_query --gallery_dir reid_gallery
```

### 3. Identity Drift Analysis
```bash
# Analyze temporal consistency
python scripts/re-id/evaluate_identity_drift_osnet.py --frames_dir anonymized_frames --plot_path identity_drift.png
```

## 📊 Evaluation Metrics

### Privacy Metrics
- **Rank-1 Accuracy**: Percentage of anonymized queries correctly matched to original identities
- **mAP**: Mean Average Precision across all ranks
- **Privacy Levels**:
  - 🟢 EXCELLENT (< 10%)
  - 🟡 GOOD (10-25%)
  - 🟠 MODERATE (25-50%)
  - 🔴 POOR (> 50%)

### Quality Metrics
- **Face Detection Rate**: Measures privacy preservation (lower = better)
- **LPIPS**: Learned Perceptual Image Patch Similarity for identity drift
- **SSIM**: Structural Similarity Index for temporal consistency
- **Action Recognition Accuracy**: Measures utility preservation

## 🔧 Core Components

### DeepPrivacy2
- **Purpose**: Pose-conditioned GAN for video anonymization
- **Features**: Temporal consistency, multi-person support
- **Status**: ✅ Implemented and functional

### Re-Identification Evaluation
- **Purpose**: Measure identity leakage after anonymization
- **Models**: OSNet, AGW (Attribute-Guided Network)
- **Status**: ✅ Complete pipeline with comprehensive metrics

### Action Recognition
- **Purpose**: Evaluate utility preservation
- **Framework**: MMAction2
- **Status**: ✅ Integrated for downstream task evaluation

## 📈 Research Contributions

This project contributes to privacy-preserving computer vision through:

1. **Comprehensive Privacy Evaluation**: Novel re-identification assessment pipeline
2. **Temporal Consistency**: Ensuring smooth anonymized video generation
3. **Multi-Modal Metrics**: Face detection, identity drift, and quality assessment
4. **Utility Preservation**: Action recognition evaluation
5. **Real-World Applicability**: Practical implementation with DeepPrivacy2

## 🎓 Academic Context

This work addresses the critical challenge of balancing privacy protection with data utility in video analysis. By implementing and evaluating pose-based anonymization, we contribute to:

- **Privacy-Preserving Computer Vision**: Novel evaluation methodologies
- **Video Anonymization**: Practical implementation with state-of-the-art GANs
- **Re-Identification Research**: Comprehensive privacy assessment frameworks

## 📝 License

This project is for academic research purposes. Please refer to individual component licenses for specific terms.

## 🤝 Acknowledgments

- DeepPrivacy2: [Original Implementation](https://github.com/hukkelas/deep_privacy2)
- Detectron2: Facebook Research
- MMAction2: OpenMMLab
- OSNet: [TorchReID](https://github.com/KaiyangZhou/deep-person-reid) 