import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup face detector and embedder ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Utility: Load face embeddings from video ---
def extract_face_embeddings(video_path, max_faces=10):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    frame_count = 0
    while frame_count < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        face = detector(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = embedder(face).cpu().numpy()
                embeddings.append(emb[0])
            frame_count += 1
    cap.release()
    return np.array(embeddings)

# --- Compute pairwise cosine similarity ---
def compute_identity_similarity(original_video, anonymized_video):
    print(f"🔍 Extracting faces from original: {original_video}")
    emb_orig = extract_face_embeddings(original_video)
    print(f"🔍 Extracting faces from anonymized: {anonymized_video}")
    emb_anon = extract_face_embeddings(anonymized_video)

    if len(emb_orig) == 0 or len(emb_anon) == 0:
        print("❌ Not enough faces detected for similarity comparison.")
        return None

    sim_matrix = cosine_similarity(emb_orig, emb_anon)
    max_sim = np.max(sim_matrix)
    avg_sim = np.mean(sim_matrix)

    print("\n📊 Identity Leakage Evaluation:")
    print(f"🔁 Max Cosine Similarity: {max_sim:.3f} (lower = better privacy)")
    print(f"🔁 Avg Cosine Similarity: {avg_sim:.3f} (lower = better privacy)")

    return max_sim, avg_sim

# --- Run this on your video pair ---
if __name__ == "__main__":
    original_path = "/work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101/Archery/v_Archery_g01_c01.avi"
    anonymized_path = "/work/tc067/tc067/s2737744/output/ucf101_anonymized/Archery/v_Archery_g01_c01_anonymized.mp4"

    compute_identity_similarity(original_path, anonymized_path)
