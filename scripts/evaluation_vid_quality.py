import cv2
import torch
import lpips
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from facenet_pytorch import MTCNN

# ======== Setup ========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_model = lpips.LPIPS(net='alex').to(device)
face_detector = MTCNN(keep_all=False, device=device)
resize_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# ======== Load video frames ========
def load_video_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

# ======== Face detection rate ========
def face_detection_rate(frames):
    detected = 0
    for f in frames:
        img = Image.fromarray(f)
        face = face_detector(img)
        if face is not None:
            detected += 1
    return detected / len(frames)

# ======== LPIPS for identity drift ========
def compute_lpips(frames):
    scores = []
    for i in range(len(frames) - 1):
        img1 = resize_256(Image.fromarray(frames[i])).unsqueeze(0).to(device)
        img2 = resize_256(Image.fromarray(frames[i + 1])).unsqueeze(0).to(device)
        with torch.no_grad():
            d = lpips_model(img1, img2).item()
        scores.append(d)
    return np.mean(scores)

# ======== SSIM for temporal consistency ========
def compute_ssim(frames):
    scores = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        s, _ = ssim(gray1, gray2, full=True)
        scores.append(s)
    return np.mean(scores)

# ======== Main evaluation function ========
def evaluate_single_video(anonymized_path):
    frames = load_video_frames(anonymized_path)
    if len(frames) < 2:
        print("❌ Not enough frames for evaluation.")
        return

    print(f"\n🔍 Evaluating: {anonymized_path}")
    fdr = face_detection_rate(frames)
    lp = compute_lpips(frames)
    sm = compute_ssim(frames)

    print(f"\n📊 Video Quality Evaluation:")
    print(f"🧠 Face Detection Rate      : {fdr:.3f} (lower = better privacy)")
    print(f"🔄 LPIPS (identity drift)   : {lp:.3f} (lower = better privacy)")
    print(f"🎞️ SSIM (temporal consistency): {sm:.3f} (higher = better quality)")

# ======== Example usage ========
if __name__ == "__main__":
    anonymized_video_path = "/work/tc067/tc067/s2737744/output/ucf101_anonymized/Archery/v_Archery_g01_c01_anonymized.mp4"
    evaluate_single_video(anonymized_video_path)
