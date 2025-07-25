from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix, evaluate_rank
from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms

# 1. Load OSNet pre-trained on Market1501
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 2. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_images_from_folder(folder):
    images, ids = [], []
    for fname in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        images.append(preprocess(img).unsqueeze(0))
        ids.append(int(fname.split('_')[0]))  # Person ID
    return torch.cat(images), ids

def extract_camids(filenames):
    # Assumes filenames like 0001_c1.jpg or 0001_c2_001.jpg
    camids = []
    for fname in filenames:
        # Find '_cX' in the filename
        parts = fname.split('_')
        for part in parts:
            if part.startswith('c') and part[1].isdigit():
                camids.append(int(part[1]))
                break
        else:
            camids.append(0)  # Default camid if not found
    return camids

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Identity Leakage after Anonymization using OSNet Re-ID.")
    parser.add_argument('--query_dir', type=str, default='datasets/ucf_reid_eval/query', 
                       help='Directory with ANONYMIZED query images')
    parser.add_argument('--gallery_dir', type=str, default='datasets/ucf_reid_eval/gallery', 
                       help='Directory with ORIGINAL gallery images')
    args = parser.parse_args()

    print("🔐 Privacy Evaluation: Identity Leakage Assessment")
    print("=" * 60)
    print("Setup:")
    print("  • Query set: ANONYMIZED frames")
    print("  • Gallery set: ORIGINAL frames")
    print("  • Goal: Measure if anonymization prevents re-identification")
    print("  • Low Rank-1 accuracy = Good privacy (identity obfuscated)")
    print("  • High Rank-1 accuracy = Poor privacy (identity preserved)")
    print()

    query_filenames = sorted(os.listdir(args.query_dir))
    gallery_filenames = sorted(os.listdir(args.gallery_dir))

    print(f"📁 Loading {len(query_filenames)} anonymized query images from: {args.query_dir}")
    print(f"📁 Loading {len(gallery_filenames)} original gallery images from: {args.gallery_dir}")
    print()

    query_imgs, query_ids = load_images_from_folder(args.query_dir)
    gallery_imgs, gallery_ids = load_images_from_folder(args.gallery_dir)

    q_camids = extract_camids(query_filenames)
    g_camids = extract_camids(gallery_filenames)

    # 4. Extract features
    print("🧠 Extracting OSNet features...")
    query_feats = extractor(query_imgs)
    gallery_feats = extractor(gallery_imgs)

    # 5. Compute cosine distances and evaluate
    print("🔍 Computing similarity matrix and evaluating re-identification...")
    distmat = compute_distance_matrix(query_feats, gallery_feats, metric='cosine')
    cmc, mAP = evaluate_rank(distmat, query_ids, gallery_ids, q_camids, g_camids)

    # 6. Show results with privacy interpretation
    print("\n" + "=" * 60)
    print("🔐 PRIVACY EVALUATION RESULTS")
    print("=" * 60)
    
    rank1_acc = cmc[0] * 100
    map_score = mAP * 100
    
    print(f"📊 Rank-1 Accuracy: {rank1_acc:.2f}%")
    print(f"📊 mAP Score: {map_score:.2f}%")
    print()
    
    # Privacy interpretation
    print("🔒 PRIVACY ASSESSMENT:")
    if rank1_acc < 10:
        privacy_level = "EXCELLENT"
        privacy_emoji = "🟢"
    elif rank1_acc < 25:
        privacy_level = "GOOD"
        privacy_emoji = "🟡"
    elif rank1_acc < 50:
        privacy_level = "MODERATE"
        privacy_emoji = "🟠"
    else:
        privacy_level = "POOR"
        privacy_emoji = "🔴"
    
    print(f"{privacy_emoji} Privacy Level: {privacy_level}")
    print(f"   • {rank1_acc:.1f}% of anonymized queries were correctly matched to original identities")
    print(f"   • This indicates the level of identity leakage after anonymization")
    print()
    
    if rank1_acc < 25:
        print("✅ GOOD NEWS: Anonymization appears to be working well!")
        print("   The low Rank-1 accuracy suggests identities are being effectively obfuscated.")
    else:
        print("⚠️  CONCERN: Anonymization may not be sufficient!")
        print("   The high Rank-1 accuracy suggests identities are still recognizable.")
        print("   Consider strengthening the anonymization method.")
    
    print("\n" + "=" * 60)