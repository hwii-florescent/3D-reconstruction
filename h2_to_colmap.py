#!/usr/bin/env python3
"""
colmap_import.py

Takes ALIKED keypoints/descriptors and LightGlue matches (HDF5 files)
and imports them into COLMAP, then runs COLMAP reconstruction (sparse
and dense) frame-by-frame.

Usage:
    # Process all frames from left_angle/, middle_angle/, right_angle/:
    python colmap_import.py

    # Process only a single frame (e.g. "001"):
    python colmap_import.py --single-frame 001
"""

import os
import shutil
import argparse
import subprocess
import glob
import h5py
import numpy as np

# ------------------------------------------------------------------------------
# 1) CONFIGURATION — edit these intrinsics/wrk paths as needed:
# ------------------------------------------------------------------------------

# Folder names where your original images currently live; they must match
# exactly what you used in `collect_triplets` when running ALIKED/LightGlue.
LEFT_DIR   = "left_angle"
MID_DIR    = "middle_angle"
RIGHT_DIR  = "right_angle"

# Path where detect_aliked created:
FEATURE_DIR = ".featureout"   # contains keypoints.h5, descriptors.h5, matches.h5

# Name of the COLMAP binary (if not on $PATH, give the full path to `colmap` here)
COLMAP_BIN = "colmap"

# Camera intrinsics: (fx, fy, cx, cy, image_width, image_height)
# Replace these with your real camera parameters.
# If all three cameras share intrinsics, use the same values here.
CAMERA_PARAMS = {
    "fx": 388.14,
    "fy": 387.46,
    "cx": 321.59,
    "cy": 240.76,
    "width": 640,
    "height": 480,
}

# If you want to run dense reconstruction, set DENSE=True. Otherwise, only sparse.
DENSE = True

# ------------------------------------------------------------------------------
# 2) Helper: build a list of triplets (idx, left_path, mid_path, right_path)
# ------------------------------------------------------------------------------
def collect_triplets(left_dir, mid_dir, right_dir):
    """
    Returns a list of (idx, left_path, mid_path, right_path) for every frame idx
    that appears in all three folders.
    """
    def build_map(folder):
        d = {}
        for fn in os.listdir(folder):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            name, _ = os.path.splitext(fn)
            parts = name.split("_")
            idx = parts[-1]   # assumes "rgb_frame_<idx>"
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

# ------------------------------------------------------------------------------
# 3) Utilities to write COLMAP‐compatible keypoint & descriptor files
# ------------------------------------------------------------------------------

import numpy as np
import h5py

def write_keypoints_and_descriptors(
    h5_kp_path,
    h5_desc_path,
    image_key,
    out_keypoint_txt,
    out_descriptor_desc,
):
    """
    Reads one image’s keypoints & descriptors from HDF5, squeezes away any
    singleton dimensions, and then truncates (or pads) every descriptor to
    exactly 128 floats. Finally, writes:

      - out_keypoint_txt   : one line per keypoint "x y 1.0 0.0"
      - out_descriptor_desc: one line per descriptor (128 space‐separated floats)

    This guarantees COLMAP will accept them as valid 128‐dim SIFT features.
    """

    # --- 1) Read raw arrays from HDF5 ---
    with h5py.File(h5_kp_path, "r") as f_kp, h5py.File(h5_desc_path, "r") as f_desc:
        kpts = f_kp[image_key][...]   # maybe shape (1, N, 2) or (N, 2)
        desc = f_desc[image_key][...]  # maybe shape (1, N, D) or (N, D,1), etc.

    # --- 2) Squeeze out any extra dimensions for keypoints ---
    kpts = np.squeeze(kpts)
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError(f"Expected keypoints→shape (N,2), but got {kpts.shape}")

    # --- 3) Squeeze & reshape descriptors into (N, D) ---
    desc = np.squeeze(desc)
    if desc.ndim == 2:
        N, D = desc.shape
    else:
        # If still >2 dims, collapse all dims after the first
        desc = desc.reshape(desc.shape[0], -1)
        N, D = desc.shape

    # --- 4) Truncate or pad so that every descriptor is length 128 ---
    if D < 128:
        # Pad with zeros on the right
        padded = np.zeros((N, 128), dtype=desc.dtype)
        padded[:, :D] = desc
        desc = padded
        D = 128
    elif D > 128:
        # Keep only the first 128 values
        desc = desc[:, :128]
        D = 128

    # At this point: kpts.shape == (N, 2) and desc.shape == (N, 128)

    # --- 5) Write keypoints (one per line: "x y 1.0 0.0") ---
    with open(out_keypoint_txt, "w") as f_kp_txt:
        for x, y in kpts:
            f_kp_txt.write(f"{x:.6f} {y:.6f} 1.0 0.0\n")

    # --- 6) Write descriptors (one 128‐float vector per line) ---
    with open(out_descriptor_desc, "w") as f_desc_txt:
        for i in range(N):
            row = desc[i].tolist()
            # e.g. "d0 d1 d2 ... d127\n"
            f_desc_txt.write(" ".join(str(v) for v in row) + "\n")


# ------------------------------------------------------------------------------
# 4) Utilities to write matches for a given image‐pair
# ------------------------------------------------------------------------------

def write_matches_for_pair(
    h5_match_path,
    image_key0,
    image_key1,
    out_match_txt,
):
    """
    Reads matches between image_key0 and image_key1 from matches.h5 and writes
    a COLMAP‐compatible match file:
      First line: M    (number of matches)
      Next M lines: idx0 idx1  (0-based indices into that image's keypoints list)
    """

    with h5py.File(h5_match_path, "r") as f_match:
        # In matches.h5, matches are stored under group=image_key0, dataset=image_key1
        matches = f_match[image_key0][image_key1][...]  # shape: (M, 2)

    M = matches.shape[0]
    with open(out_match_txt, "w") as f:
        f.write(f"{M}\n")
        for i0, i1 in matches:
            f.write(f"{i0} {i1}\n")

# ------------------------------------------------------------------------------
# 5) High‐level function: generate a COLMAP workspace for exactly one frame
# ------------------------------------------------------------------------------

def run_colmap_for_frame(
    idx,
    left_path,
    mid_path,
    right_path,
    feature_dir,
    camera_params,
    colmap_bin="colmap",
    dense=False,
):
    """
    For a single frame (idx), produce:
      workspace/frame_<idx>/
        ├─ images/             # copied images for this frame
        ├─ features/           # keypoints (*.txt) and descriptors (*.desc)
        ├─ matches/            # match files *.txt
        ├─ database.db         # COLMAP database with features+matches imported
        ├─ sparse/             # output of 'colmap mapper'
        └─ dense/              # (optional) dense reconstruction outputs

    1) Copy the three images into 'images/' under a new folder
       using their existing filenames (e.g. left_angle_rgb_frame_001.png)
    2) Read ALIKED keypoints/descriptors from feature_dir/keypoints.h5 &
       descriptors.h5, write out TXT/DESC files into 'features/'
    3) Read LightGlue matches from feature_dir/matches.h5, write out
       three match files (L‐M, M‐R, L‐R) into 'matches/'. Also create a
       'match_list.txt' that lists all pairs.
    4) Call COLMAP feature_importer, matches_importer, mapper, etc.
    """

    # (a) Compute HDF5 keys for each image: f"{folder}_{basename}"
    def make_hdf5_key(image_path):
        folder = os.path.basename(os.path.dirname(image_path))
        basename = os.path.basename(image_path)
        return f"{folder}_{basename}"

    keyL = make_hdf5_key(left_path)
    keyM = make_hdf5_key(mid_path)
    keyR = make_hdf5_key(right_path)

    # (b) Create a fresh workspace: workspace/frame_<idx>/
    ws_root = os.path.join("workspace", f"frame_{idx}")
    if os.path.exists(ws_root):
        shutil.rmtree(ws_root)
    os.makedirs(ws_root, exist_ok=True)

    # (c) Subfolders: images/, features/, matches/
    images_dir   = os.path.join(ws_root, "images")
    features_dir = os.path.join(ws_root, "features")
    matches_dir  = os.path.join(ws_root, "matches")
    os.makedirs(images_dir,   exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(matches_dir,  exist_ok=True)

    # (d) Copy the three images (use their existing filename):
    for src in (left_path, mid_path, right_path):
        dst = os.path.join(images_dir, os.path.basename(src))
        shutil.copy(src, dst)

    # (e) Write out keypoints & descriptors for each image
    h5_kp_path   = os.path.join(feature_dir, "keypoints.h5")
    h5_desc_path = os.path.join(feature_dir, "descriptors.h5")

    for img_path, img_key in [(left_path, keyL), (mid_path, keyM), (right_path, keyR)]:
        basename = os.path.basename(img_path)
        kp_txt = os.path.join(features_dir, f"{basename}.txt")
        descf  = os.path.join(features_dir, f"{basename}.desc")
        write_keypoints_and_descriptors(
            h5_kp_path,
            h5_desc_path,
            img_key,
            kp_txt,
            descf,
        )

    # (f) Write out matches for each of the three pairs:
    h5_match_path = os.path.join(feature_dir, "matches.h5")
    pair_keys = [
        ( (keyL, keyM), (left_path, mid_path) ),
        ( (keyM, keyR), (mid_path, right_path) ),
        ( (keyL, keyR), (left_path, right_path) )
    ]

    # Build a text file listing all (image1 image2 matchfile) for COLMAP
    match_list_txt = os.path.join(matches_dir, "match_list.txt")
    with open(match_list_txt, "w") as flist:
        for (k0, k1), (p0, p1) in pair_keys:
            fn0 = os.path.basename(p0)
            fn1 = os.path.basename(p1)
            out_match = os.path.join(matches_dir, f"{fn0}__{fn1}.txt")
            # Write the match‐pair file
            write_matches_for_pair(h5_match_path, k0, k1, out_match)
            # Add to match_list.txt: "<image1> <image2> <relative/path/to/match.txt>"
            # Since COLMAP is run from ws_root, we use relative paths:
            rel_match = os.path.relpath(out_match, ws_root)
            flist.write(f"{fn0} {fn1} {rel_match}\n")

    # (g) Build COLMAP database + import features & matches
    db_path = os.path.join(ws_root, "database.db")

    # 1) Feature importer
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    cam_params_csv = f"{fx},{fy},{cx},{cy}"

    feat_imp_cmd = [
        COLMAP_BIN, "feature_importer",
        "--database_path", db_path,
        "--image_path",    images_dir,
        "--import_path",   features_dir,
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", cam_params_csv,
    ]
    print("Running COLMAP feature_importer for frame", idx)
    subprocess.run(feat_imp_cmd, check=True)

    # 2) Matches importer
    match_imp_cmd = [
        COLMAP_BIN, "matches_importer",
        "--database_path",    db_path,
        "--match_list_path",  os.path.relpath(match_list_txt, ws_root),
        "--MatchReader.match_type", "raw",   # raw means idx‐idx pairs
        "--MatchReader.bit_format", "0",      # 0 = integer format
    ]
    print("Running COLMAP matches_importer for frame", idx)
    subprocess.run(match_imp_cmd, check=True)

    # (h) Run sparse mapper
    sparse_dir = os.path.join(ws_root, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    mapper_cmd = [
        COLMAP_BIN, "mapper",
        "--database_path", db_path,
        "--image_path",    images_dir,
        "--output_path",   sparse_dir,
    ]
    print("Running COLMAP mapper (sparse) for frame", idx)
    subprocess.run(mapper_cmd, check=True)

    if not dense:
        return

    # (i) Optional: Dense reconstruction pipeline
    dense_dir = os.path.join(ws_root, "dense")
    os.makedirs(dense_dir, exist_ok=True)

    # 1) Undistort images
    undistort_cmd = [
        COLMAP_BIN, "image_undistorter",
        "--image_path",    images_dir,
        "--input_path",    os.path.join(sparse_dir, "0"),  # COLMAP creates a folder named "0"
        "--output_path",   dense_dir,
        "--output_type",   "COLMAP",
        "--max_image_size", "2000"
    ]
    print("Running COLMAP image_undistorter for frame", idx)
    subprocess.run(undistort_cmd, check=True)

    # 2) Patch-match stereo
    stereo_cmd = [
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path",  dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]
    print("Running COLMAP patch_match_stereo for frame", idx)
    subprocess.run(stereo_cmd, check=True)

    # 3) Stereo fusion
    fusion_cmd = [
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path",   dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type",       "geometric",
        "--output_path",      os.path.join(dense_dir, "fused.ply"),
    ]
    print("Running COLMAP stereo_fusion for frame", idx)
    subprocess.run(fusion_cmd, check=True)

# ------------------------------------------------------------------------------
# 6) Main: parse arguments, loop over frames
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Import ALIKED+LightGlue outputs into COLMAP per frame."
    )
    parser.add_argument(
        "--single-frame", dest="single_frame", default=None,
        help="If set, only process the triplet with this frame index (e.g. 001)."
    )
    args = parser.parse_args()

    # 1) Gather all triplets
    triplets = collect_triplets(LEFT_DIR, MID_DIR, RIGHT_DIR)
    if len(triplets) == 0:
        print("No triplets found in", LEFT_DIR, MID_DIR, RIGHT_DIR)
        return

    # 2) Filter if single frame specified
    if args.single_frame:
        triplets = [t for t in triplets if t[0] == args.single_frame]
        if not triplets:
            print(f"Frame {args.single_frame} not found.")
            return

    # 3) Check that HDF5 files exist
    h5_kp   = os.path.join(FEATURE_DIR, "keypoints.h5")
    h5_desc = os.path.join(FEATURE_DIR, "descriptors.h5")
    h5_m    = os.path.join(FEATURE_DIR, "matches.h5")
    for fpath in (h5_kp, h5_desc, h5_m):
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Expected to find {fpath} (from FEATURE_DIR)")

    # 4) Loop over each frame and run COLMAP
    for idx, left_p, mid_p, right_p in triplets:
        print("\n======== Processing frame", idx, "========")
        run_colmap_for_frame(
            idx,
            left_p,
            mid_p,
            right_p,
            FEATURE_DIR,
            CAMERA_PARAMS,
            colmap_bin=COLMAP_BIN,
            dense=DENSE,
        )

    print("\nAll done! Each frame’s COLMAP outputs are in workspace/frame_<idx>/")

if __name__ == "__main__":
    main()
