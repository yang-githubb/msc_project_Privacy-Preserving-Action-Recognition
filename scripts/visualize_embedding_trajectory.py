import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms
from torchreid.utils import FeatureExtractor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mediapipe as mp
import torch

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

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

def extract_embeddings_from_frames(frames_dir, extractor, preprocess, max_frames=None, cropped_dir=None):
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
        if cropped_dir is not None:
            os.makedirs(cropped_dir, exist_ok=True)
            crop_pil.save(os.path.join(cropped_dir, frame_path.name))
        tensor = preprocess(crop_pil).unsqueeze(0)
        with torch.no_grad():
            emb = extractor(tensor)
        embeddings.append(emb.squeeze(0).cpu().numpy())
        frame_names.append(frame_path.name)
    return np.stack(embeddings), frame_names

def plot_trajectory(embeddings, method, output_plot, label=None, color=None):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError('method must be pca or tsne')
    reduced = reducer.fit_transform(embeddings)
    plt.plot(reduced[:, 0], reduced[:, 1], marker='o', label=label, color=color)
    plt.scatter(reduced[0, 0], reduced[0, 1], c='green', s=80, label='Start' if label is None else f'{label} Start', zorder=5)
    plt.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=80, label='End' if label is None else f'{label} End', zorder=5)
    plt.title(f'Embedding Trajectory ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f'Saved {method.upper()} trajectory plot to {output_plot}')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize embedding trajectory using PCA/t-SNE.')
    parser.add_argument('--frames_dir', type=str, default=None, help='Directory with frames to extract embeddings from')
    parser.add_argument('--frames_dir_orig', type=str, default=None, help='Directory with original frames for comparison')
    parser.add_argument('--frames_dir_anon', type=str, default=None, help='Directory with anonymized frames for comparison')
    parser.add_argument('--model_name', type=str, default='osnet_x1_0', help='Torchreid model name')
    parser.add_argument('--output_plot', type=str, default='embedding_trajectory.png', help='Output plot file')
    parser.add_argument('--max_frames', type=int, default=None, help='Max number of frames to process')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'], help='Dimensionality reduction method')
    parser.add_argument('--cropped_dir', type=str, default=None, help='Directory to save cropped pose images (optional)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=None,
        device=device
    )
    preprocess = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    if args.frames_dir:
        embeddings, _ = extract_embeddings_from_frames(args.frames_dir, extractor, preprocess, args.max_frames, args.cropped_dir)
        plot_trajectory(embeddings, args.method, args.output_plot)
    elif args.frames_dir_orig and args.frames_dir_anon:
        emb_orig, _ = extract_embeddings_from_frames(args.frames_dir_orig, extractor, preprocess, args.max_frames, args.cropped_dir)
        emb_anon, _ = extract_embeddings_from_frames(args.frames_dir_anon, extractor, preprocess, args.max_frames, args.cropped_dir)
        plt.figure(figsize=(10, 8))
        plot_trajectory(emb_orig, args.method, args.output_plot.replace('.png', '_orig.png'), label='Original', color='blue')
        plot_trajectory(emb_anon, args.method, args.output_plot.replace('.png', '_anon.png'), label='Anonymized', color='orange')
        # Overlay both on a single plot
        reducer = PCA(n_components=2) if args.method == 'pca' else TSNE(n_components=2, random_state=42)
        all_emb = np.concatenate([emb_orig, emb_anon], axis=0)
        reduced = reducer.fit_transform(all_emb)
        n = len(emb_orig)
        plt.plot(reduced[:n, 0], reduced[:n, 1], marker='o', label='Original', color='blue')
        plt.plot(reduced[n:, 0], reduced[n:, 1], marker='o', label='Anonymized', color='orange')
        plt.scatter(reduced[0, 0], reduced[0, 1], c='green', s=80, label='Orig Start', zorder=5)
        plt.scatter(reduced[n-1, 0], reduced[n-1, 1], c='red', s=80, label='Orig End', zorder=5)
        plt.scatter(reduced[n, 0], reduced[n, 1], c='green', s=80, label='Anon Start', zorder=5)
        plt.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=80, label='Anon End', zorder=5)
        plt.title(f'Embedding Trajectory Comparison ({args.method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f'Saved overlay {args.method.upper()} trajectory plot to {args.output_plot}')
        plt.close()
    else:
        print('Please specify either --frames_dir or both --frames_dir_orig and --frames_dir_anon')
        exit(1)

if __name__ == '__main__':
    main() 