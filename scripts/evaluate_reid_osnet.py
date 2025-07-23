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

query_filenames = sorted(os.listdir('/work/tc067/tc067/s2737744/datasets/ucf_reid_eval/query'))      # original
gallery_filenames = sorted(os.listdir('/work/tc067/tc067/s2737744/datasets/ucf_reid_eval/gallery'))  # anonymized

query_imgs, query_ids = load_images_from_folder('/work/tc067/tc067/s2737744/datasets/ucf_reid_eval/query')
gallery_imgs, gallery_ids = load_images_from_folder('/work/tc067/tc067/s2737744/datasets/ucf_reid_eval/gallery')

q_camids = extract_camids(query_filenames)
g_camids = extract_camids(gallery_filenames)

# 4. Extract features
query_feats = extractor(query_imgs)
gallery_feats = extractor(gallery_imgs)

# 5. Compute cosine distances and evaluate
distmat = compute_distance_matrix(query_feats, gallery_feats, metric='cosine')
cmc, mAP = evaluate_rank(distmat, query_ids, gallery_ids, q_camids, g_camids)

# 6. Show results
print(f'🔐 Re-ID Evaluation (OSNet):')
print(f'Rank-1 Accuracy: {cmc[0]*100:.2f}%')
print(f'mAP: {mAP*100:.2f}%')