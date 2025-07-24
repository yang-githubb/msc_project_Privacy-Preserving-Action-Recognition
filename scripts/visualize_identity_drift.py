import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

FIXED_SIZE = (128, 256)  # width, height

def load_images(crop_dir):
    crops = sorted([f for f in os.listdir(crop_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    images = [Image.open(os.path.join(crop_dir, f)).convert('RGB').resize(FIXED_SIZE, Image.BILINEAR) for f in crops]
    return images, crops

def plot_cosine_similarity(images, out_png):
    # Convert images to numpy arrays
    arrs = [np.asarray(img).astype(np.float32).flatten() for img in images]
    sims = []
    for i in range(len(arrs)-1):
        a, b = arrs[i], arrs[i+1]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(sim)
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, len(sims)+1), sims, marker='o')
    plt.title('Frame-to-frame Cosine Similarity (pixel space)')
    plt.xlabel('Frame index')
    plt.ylabel('Cosine similarity')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved similarity plot to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize frame-to-frame cosine similarity in a crop sequence.")
    parser.add_argument('--crop_dir', type=str, required=True, help='Directory with crops')
    parser.add_argument('--out_png', type=str, default='drift_visualization.png', help='Output PNG plot file')
    args = parser.parse_args()

    images, crops = load_images(args.crop_dir)
    plot_cosine_similarity(images, args.out_png) 