import os
import glob
import numpy as np
import open3d as o3d
import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# === ALIKED & LightGlue imports ===
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from lightglue import viz2d

from collections import defaultdict

def collect_triplets(left_dir, mid_dir, right_dir):
    """
    Returns a list of (idx, left_path, mid_path, right_path) for every frame idx
    that appears in all three folders.
    """
    # Helper to build map: frame_idx → full path
    def build_map(folder):
        d = {}
        for fn in os.listdir(folder):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # extract "001" from "rgb_frame_001.png"
            name, _ = os.path.splitext(fn)
            # assumes format "rgb_frame_<idx>"
            parts = name.split('_')
            idx = parts[-1]
            d[idx] = os.path.join(folder, fn)
        return d

    left_map  = build_map(left_dir)
    mid_map   = build_map(mid_dir)
    right_map = build_map(right_dir)

    triplets = []
    for idx in sorted(left_map.keys()):
        if idx in mid_map and idx in right_map:
            triplets.append((idx, left_map[idx], mid_map[idx], right_map[idx]))
    return triplets

# === Step 1: Extract ALIKED keypoints & descriptors ===
def detect_aliked(img_fnames, feature_dir, device, max_num_keypoints=4096, resize=1024):
    extractor = ALIKED(max_num_keypoints=max_num_keypoints,
                       detection_threshold=0.05,
                       resize=resize).eval().to(device)
    os.makedirs(feature_dir, exist_ok=True)
    kp_h5 = os.path.join(feature_dir, 'keypoints.h5')
    desc_h5 = os.path.join(feature_dir, 'descriptors.h5')
    with h5py.File(kp_h5, 'w') as f_kp, h5py.File(desc_h5, 'w') as f_desc:
        for img_path in tqdm(img_fnames, desc='ALIKED Extractor'):
            folder   = os.path.basename(os.path.dirname(img_path))
            basename = os.path.basename(img_path)
            key      = f"{folder}_{basename}"
            with torch.inference_mode():
                img = load_image(img_path)
                feats = extractor.extract(img.to(device))
                kpts = feats['keypoints'].cpu().numpy()
                descs = feats['descriptors'].cpu().numpy()
            f_kp[key] = kpts
            f_desc[key] = descs

# === Step 2: Run LightGlue matching on shortlisted or all pairs ===
def match_triplet(left_path, mid_path, right_path, feature_dir, device):
    # Initialize extractor + matcher
    matcher = LightGlue(features='aliked').eval().to(device)

    kp_h5 = os.path.join(feature_dir, 'keypoints.h5')
    desc_h5 = os.path.join(feature_dir, 'descriptors.h5')
    match_h5 = os.path.join(feature_dir, 'matches.h5')

    def make_key(img_path):
        folder   = os.path.basename(os.path.dirname(img_path))
        basename = os.path.basename(img_path)
        return f"{folder}_{basename}"

    keyL = make_key(left_path)   # e.g. "left_angle_rgb_frame_001.png"
    keyM = make_key(mid_path)    # e.g. "middle_angle_rgb_frame_001.png"
    keyR = make_key(right_path)  # e.g. "right_angle_rgb_frame_001.png"

    # now match pairs
    with h5py.File(kp_h5, 'r') as f_kp, \
         h5py.File(desc_h5, 'r') as f_desc, \
         h5py.File(match_h5, 'w') as f_match:
        def match_pair_and_store(key0, key1, path0, path1):
            # load features
            kpts0 = torch.from_numpy(f_kp[key0][...]).to(device)
            kpts1 = torch.from_numpy(f_kp[key1][...]).to(device)
            desc0 = torch.from_numpy(f_desc[key0][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)

            feats0 = {'keypoints': kpts0, 'descriptors': desc0}
            feats1 = {'keypoints': kpts1, 'descriptors': desc1}

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [
                rbd(x) for x in [feats0, feats1, matches01]
            ]
            matches = matches01['matches']            
            
            grp = f_match.require_group(key0)
            grp.create_dataset(key1, data=matches.cpu().numpy())

            # Visualize matches
            img0 = load_image(path0)  # Tensor (3, H, W), float32
            img1 = load_image(path1)
            m_kpts0 = feats0['keypoints'][matches[:, 0]]   # (M, 2)
            m_kpts1 = feats1['keypoints'][matches[:, 1]] 

            
            axes = viz2d.plot_images([img0, img1])
            viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
            viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

            kpc0 = viz2d.cm_prune(matches01['prune0'])
            kpc1 = viz2d.cm_prune(matches01['prune1'])
            viz2d.plot_images([img0, img1])
            viz2d.plot_keypoints([feats0['keypoints'], feats1['keypoints']], colors=[kpc0, kpc1], ps=10)

            plt.show()
        match_pair_and_store(keyL, keyM, left_path, mid_path)
        match_pair_and_store(keyM, keyR, mid_path, right_path)
        match_pair_and_store(keyL, keyR, left_path, right_path)

# # === Step 3: Shortlist pairs by LightGlue match counts ===
# def shortlist_by_matches(img_fnames, feature_dir, min_matches=30, brute_thresh=20):
#     n = len(img_fnames)
#     if n <= brute_thresh:
#         return [(i, j) for i in range(n) for j in range(i+1, n)]
#     match_h5 = os.path.join(feature_dir, 'matches.h5')
#     pairs = []
#     # map basename to index
#     name_to_idx = {os.path.basename(f): idx for idx, f in enumerate(img_fnames)}
#     with h5py.File(match_h5, 'r') as f_match:
#         for k1 in f_match.keys():
#             for k2 in f_match[k1].keys():
#                 idx1, idx2 = name_to_idx[k1], name_to_idx[k2]
#                 cnt = f_match[k1][k2].shape[0]
#                 if cnt >= min_matches:
#                     pairs.append(tuple(sorted((idx1, idx2))))
#     return sorted(set(pairs))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Calculate matches for triplets of images
    left_dir  = 'left_angle'
    mid_dir   = 'middle_angle'
    right_dir = 'right_angle'
    triplets  = collect_triplets(left_dir, mid_dir, right_dir)
    feature_dir = '.featureout'

    all_img_files = sorted({p for (_idx, p, q, r) in triplets for p in (p, q, r)})
    detect_aliked(all_img_files, feature_dir, device,
                  max_num_keypoints=4096, resize=1024)

    for idx, left_p, mid_p, right_p in triplets:
        match_triplet(left_p, mid_p, right_p, feature_dir, device)