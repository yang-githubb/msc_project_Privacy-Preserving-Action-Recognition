import os
import torch
from torchreid.utils import FeatureExtractor
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
import cv2
from pathlib import Path
import mediapipe as mp
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import structural_similarity as ssim

def get_agw_extractor(device='cuda'):
    return FeatureExtractor(
        model_name='pcb_p6',
        model_path=None,
        device=device if torch.cuda.is_available() else 'cpu'
    )

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def create_person_mask_with_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks is None:
        return None, None
    if results.segmentation_mask is not None:
        mask = results.segmentation_mask > 0.1
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return mask, (x, y, w, h)
    return None, None

def extract_embeddings_from_frames(frames_dir, extractor, max_frames=None, output_dir=None):
    frames_dir = Path(frames_dir)
    frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    if max_frames:
        frame_files = frame_files[:max_frames]
    embeddings = []
    frame_names = []
    for frame_path in frame_files:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        mask, bbox = create_person_mask_with_pose(img)
        if mask is None or bbox is None:
            continue
        x, y, w, h = bbox
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        mask_crop = mask[y:y+h, x:x+w]
        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
        crop_rgba[:, :, 3] = mask_crop
        crop_pil = Image.fromarray(crop_rgba)
        white_bg = Image.new('RGB', crop_pil.size, (255, 255, 255))
        white_bg.paste(crop_pil, mask=crop_pil.split()[-1])
        crop_pil = white_bg
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            crop_pil.save(os.path.join(output_dir, f"cropped_{frame_path.name}"))
        tensor = preprocess(crop_pil).unsqueeze(0)
        with torch.no_grad():
            emb = extractor(tensor)
        embeddings.append(emb.squeeze(0).cpu().numpy())
        frame_names.append(frame_path.name)
    return np.stack(embeddings), frame_names

def compute_consecutive_cosine_sim(embeddings):
    from numpy import dot
    from numpy.linalg import norm
    sims = []
    for i in range(len(embeddings)-1):
        a, b = embeddings[i], embeddings[i+1]
        sim = dot(a, b) / (norm(a) * norm(b))
        sims.append(sim)
    return np.array(sims)

def compute_consecutive_euclidean(embeddings):
    from numpy.linalg import norm
    dists = []
    for i in range(len(embeddings)-1):
        a, b = embeddings[i], embeddings[i+1]
        dists.append(norm(a - b))
    return np.array(dists)

def compute_consecutive_manhattan(embeddings):
    dists = []
    for i in range(len(embeddings)-1):
        a, b = embeddings[i], embeddings[i+1]
        dists.append(np.sum(np.abs(a - b)))
    return np.array(dists)

def compute_consecutive_pearson(embeddings):
    corrs = []
    for i in range(len(embeddings)-1):
        a, b = embeddings[i], embeddings[i+1]
        if np.std(a) == 0 or np.std(b) == 0:
            corrs.append(0)
        else:
            corrs.append(np.corrcoef(a, b)[0, 1])
    return np.array(corrs)

def compute_consecutive_lpips(frames, lpips_model):
    scores = []
    for i in range(len(frames) - 1):
        img1 = frames[i]
        img2 = frames[i + 1]
        img1_t = torch.from_numpy(cv2.resize(img1, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_t = torch.from_numpy(cv2.resize(img2, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            d = lpips_model(img1_t, img2_t).item()
        scores.append(d)
    return np.array(scores)

def compute_consecutive_ssim(frames):
    scores = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        s, _ = ssim(gray1, gray2, full=True)
        scores.append(s)
    return np.array(scores)

def plot_identity_drift_all(sims, eucl, manh, pear, lpips_scores, ssim_scores, plot_path):
    plt.figure(figsize=(14, 8))
    x = range(1, len(sims)+1)
    plt.plot(x, sims, marker='o', label='Cosine Similarity')
    plt.plot(x, eucl, marker='x', label='Euclidean Distance')
    plt.plot(x, manh, marker='s', label='Manhattan Distance')
    plt.plot(x, pear, marker='^', label='Pearson Correlation')
    if lpips_scores is not None:
        plt.plot(x, lpips_scores, marker='d', label='LPIPS (image)')
    if ssim_scores is not None:
        plt.plot(x, ssim_scores, marker='*', label='SSIM (image)')
    plt.title('Frame-to-frame Identity Drift Metrics (AGW)')
    plt.xlabel('Frame index')
    plt.ylabel('Metric value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved identity drift plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate identity drift using AGW (Torchreid) with pose-based person cropping.")
    parser.add_argument('--crop_dir', type=str, default='datasets/crops', help='Directory with pre-cropped person images')
    parser.add_argument('--frames_dir', type=str, default='datasets/frames', help='Directory with full frames to detect and crop persons using pose estimation')
    parser.add_argument('--output_dir', type=str, default='datasets/cropped_pose_outputs', help='Directory to save pose-based cropped person images')
    parser.add_argument('--max_frames', type=int, default=None, help='Max number of frames to process')
    parser.add_argument('--plot_path', type=str, default='identity_drift_plot_agw.png', help='Path to save the identity drift plot')
    parser.add_argument('--lpips_net', type=str, default='alex', help='LPIPS backbone: alex, vgg, or squeeze')
    parser.add_argument('--frames_dir_orig', type=str, default=None, help='Directory with original frames for cross-video comparison')
    parser.add_argument('--frames_dir_anon', type=str, default=None, help='Directory with anonymized frames for cross-video comparison')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = get_agw_extractor(device)
    lpips_model = lpips.LPIPS(net=args.lpips_net)

    # Cross-video comparison mode
    if args.frames_dir_orig and args.frames_dir_anon:
        print("Comparing original and anonymized videos for privacy/re-ID (AGW)...")
        def get_embeddings_and_names(frames_dir, max_frames=None):
            frames_dir = Path(frames_dir)
            frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
            if max_frames:
                frame_files = frame_files[:max_frames]
            embeddings = []
            names = []
            for frame_path in frame_files:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                mask, bbox = create_person_mask_with_pose(img)
                if mask is None or bbox is None:
                    continue
                x, y, w, h = bbox
                crop = img[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                mask_crop = mask[y:y+h, x:x+w]
                crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
                crop_rgba[:, :, 3] = mask_crop
                crop_pil = Image.fromarray(crop_rgba)
                white_bg = Image.new('RGB', crop_pil.size, (255, 255, 255))
                white_bg.paste(crop_pil, mask=crop_pil.split()[-1])
                crop_pil = white_bg
                tensor = preprocess(crop_pil).unsqueeze(0)
                with torch.no_grad():
                    emb = extractor(tensor)
                embeddings.append(emb.squeeze(0).cpu().numpy())
                names.append(frame_path.name)
            return np.stack(embeddings), names
        emb_orig, names_orig = get_embeddings_and_names(args.frames_dir_orig, args.max_frames)
        emb_anon, names_anon = get_embeddings_and_names(args.frames_dir_anon, args.max_frames)
        name_to_idx_anon = {n: i for i, n in enumerate(names_anon)}
        matched_orig = []
        matched_anon = []
        matched_names = []
        for i, n in enumerate(names_orig):
            if n in name_to_idx_anon:
                matched_orig.append(emb_orig[i])
                matched_anon.append(emb_anon[name_to_idx_anon[n]])
                matched_names.append(n)
        matched_orig = np.stack(matched_orig)
        matched_anon = np.stack(matched_anon)
        from numpy import dot
        from numpy.linalg import norm
        cosine = np.array([dot(a, b) / (norm(a) * norm(b)) for a, b in zip(matched_orig, matched_anon)])
        eucl = np.array([norm(a - b) for a, b in zip(matched_orig, matched_anon)])
        manh = np.array([np.sum(np.abs(a - b)) for a, b in zip(matched_orig, matched_anon)])
        pear = np.array([np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0 for a, b in zip(matched_orig, matched_anon)])
        print(f"Compared {len(matched_names)} matched frames between original and anonymized videos.")
        print(f"Cosine Similarity: Mean={cosine.mean():.4f}, Var={cosine.var():.4f}, Min={cosine.min():.4f}, Max={cosine.max():.4f}")
        print(f"Euclidean Distance: Mean={eucl.mean():.4f}, Var={eucl.var():.4f}, Min={eucl.min():.4f}, Max={eucl.max():.4f}")
        print(f"Manhattan Distance: Mean={manh.mean():.4f}, Var={manh.var():.4f}, Min={manh.min():.4f}, Max={manh.max():.4f}")
        print(f"Pearson Correlation: Mean={pear.mean():.4f}, Var={pear.var():.4f}, Min={pear.min():.4f}, Max={pear.max():.4f}")
        plt.figure(figsize=(12, 6))
        x = range(1, len(matched_names)+1)
        plt.plot(x, cosine, marker='o', label='Cosine Similarity (orig vs anon)')
        plt.plot(x, eucl, marker='x', label='Euclidean Distance (orig vs anon)')
        plt.plot(x, manh, marker='s', label='Manhattan Distance (orig vs anon)')
        plt.plot(x, pear, marker='^', label='Pearson Correlation (orig vs anon)')
        plt.title('Original vs Anonymized Embedding Similarity (AGW)')
        plt.xlabel('Frame index')
        plt.ylabel('Metric value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.plot_path)
        print(f"Saved original vs anonymized comparison plot to {args.plot_path}")
        exit(0)

    # Load frames for image-based metrics
    if args.frames_dir:
        frame_dir = Path(args.frames_dir)
        frame_files = sorted([f for f in frame_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        if args.max_frames:
            frame_files = frame_files[:args.max_frames]
        if args.frames_dir:
            embeddings, frame_names = extract_embeddings_from_frames(args.frames_dir, extractor, args.max_frames, args.output_dir)
            print(f"Processed {len(frame_names)} frames with pose-based person cropping.")
    elif args.crop_dir:
        crop_dir = Path(args.crop_dir)
        crop_files = sorted([f for f in crop_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        if args.max_frames:
            crop_files = crop_files[:args.max_frames]
        if args.crop_dir:
            embeddings, frame_names = extract_embeddings_from_frames(args.crop_dir, extractor, args.max_frames, None)
            print(f"Processed {len(frame_names)} pre-cropped images.")
    else:
        print("Error: Must specify either --crop_dir or --frames_dir")
        exit(1)

    # Filter frames to only those used for embeddings (by filename, in order)
    frame_name_set = set(frame_names)
    filtered_frames = []
    for f in (frame_files if args.frames_dir else crop_files):
        fname = f.name if hasattr(f, 'name') else os.path.basename(f)
        if fname in frame_name_set:
            filtered_frames.append(cv2.imread(str(f)))
    # Ensure order matches frame_names
    name_to_frame = {f.name if hasattr(f, 'name') else os.path.basename(f): cv2.imread(str(f)) for f in (frame_files if args.frames_dir else crop_files) if (f.name if hasattr(f, 'name') else os.path.basename(f)) in frame_name_set}
    filtered_frames = [name_to_frame[n] for n in frame_names if n in name_to_frame]

    sims = compute_consecutive_cosine_sim(embeddings)
    eucl = compute_consecutive_euclidean(embeddings)
    manh = compute_consecutive_manhattan(embeddings)
    pear = compute_consecutive_pearson(embeddings)
    lpips_scores = compute_consecutive_lpips(filtered_frames, lpips_model)
    ssim_scores = compute_consecutive_ssim(filtered_frames)
    print(f"Evaluated {len(sims)} consecutive frame pairs from {len(embeddings)} crops.")
    print(f"Cosine Similarity: Mean={sims.mean():.4f}, Var={sims.var():.4f}, Min={sims.min():.4f}, Max={sims.max():.4f}")
    print(f"Euclidean Distance: Mean={eucl.mean():.4f}, Var={eucl.var():.4f}, Min={eucl.min():.4f}, Max={eucl.max():.4f}")
    print(f"Manhattan Distance: Mean={manh.mean():.4f}, Var={manh.var():.4f}, Min={manh.min():.4f}, Max={manh.max():.4f}")
    print(f"Pearson Correlation: Mean={pear.mean():.4f}, Var={pear.var():.4f}, Min={pear.min():.4f}, Max={pear.max():.4f}")
    print(f"LPIPS: Mean={lpips_scores.mean():.4f}, Var={lpips_scores.var():.4f}, Min={lpips_scores.min():.4f}, Max={lpips_scores.max():.4f}")
    print(f"SSIM: Mean={ssim_scores.mean():.4f}, Var={ssim_scores.var():.4f}, Min={ssim_scores.min():.4f}, Max={ssim_scores.max():.4f}")
    plot_identity_drift_all(sims, eucl, manh, pear, lpips_scores, ssim_scores, args.plot_path)

if __name__ == "__main__":
    main() 