"""
Requirements (Local environment):

# Install PyTorch (CPU-only example; select the appropriate CUDA build if needed)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

# Install core Python libraries
```bash
pip install open3d numpy h5py kornia tqdm
```

# Install LightGlue & ALIKED from source
```bash
# Option A: Clone & install LightGlue in editable mode
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
pip install -e .
"""
import os
import glob
import re
import numpy as np
import open3d as o3d
import torch
import h5py
import kornia as K
import kornia.feature as KF
from tqdm import tqdm

# === ALIKED & LightGlue imports ===
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd

def extract_number(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

# === Load image as torch tensor for ALIKED ===
def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

# === Step 1: Extract ALIKED keypoints & descriptors ===
def detect_aliked(img_fnames, feature_dir, num_features=4096, resize_to=1024, device=torch.device('cpu')):
    extractor = ALIKED(max_num_keypoints=num_features,
                       detection_threshold=0.05,
                       resize=resize_to).eval().to(device)
    os.makedirs(feature_dir, exist_ok=True)
    kp_h5 = os.path.join(feature_dir, 'keypoints.h5')
    desc_h5 = os.path.join(feature_dir, 'descriptors.h5')
    with h5py.File(kp_h5, 'w') as f_kp, h5py.File(desc_h5, 'w') as f_desc:
        for img_path in tqdm(img_fnames, desc='Extract ALIKED'):
            key = os.path.basename(img_path)
            with torch.inference_mode():
                im = load_torch_image(img_path, device).to(dtype=torch.float32)
                feats = extractor.extract(im)
                kpts = feats['keypoints'].reshape(-1,2).cpu().numpy()
                descs = feats['descriptors'].reshape(len(kpts), -1).cpu().numpy()
            f_kp[key] = kpts
            f_desc[key] = descs

# === Step 2: Run LightGlue matching on shortlisted or all pairs ===
def match_with_lightglue(img_fnames, feature_dir, pair_list, device=torch.device('cpu'), min_matches=25):
    """
    Run ALIKED+LightGlue matching on each image pair and save matches to HDF5
    """
    # Initialize extractor + matcher
    # extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.05, resize=1024).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    kp_h5 = os.path.join(feature_dir, 'keypoints.h5')
    desc_h5 = os.path.join(feature_dir, 'descriptors.h5')
    match_h5 = os.path.join(feature_dir, 'matches.h5')

    os.makedirs(feature_dir, exist_ok=True)

    ## I don't think we need to extract keypoints & descriptors again as it will make the process twice as slow
    # with h5py.File(kp_h5, 'w') as f_kp, h5py.File(desc_h5, 'w') as f_desc:
    #     # extract and store keypoints & descriptors
    #     for img_path in tqdm(img_fnames, desc='Extract ALIKED'):
    #         key = os.path.basename(img_path)
    #         image = load_image(img_path).to(device)  # returns 3xHxW
    #         with torch.inference_mode():
    #             feats = extractor.extract(image)
    #         # remove batch dim if present
    #         feats = rbd(feats)
    #         kpts = feats['keypoints']       # Nx2
    #         descs = feats['descriptors']    # NxD
    #         f_kp[key] = kpts.cpu().numpy()
    #         f_desc[key] = descs.cpu().numpy()

    # now match pairs
    with h5py.File(kp_h5, 'r') as f_kp, \
         h5py.File(desc_h5, 'r') as f_desc, \
         h5py.File(match_h5, 'w') as f_match:
        for i, j in tqdm(pair_list, desc='LightGlue matching'):
            key1 = os.path.basename(img_fnames[i])
            key2 = os.path.basename(img_fnames[j])
            # load features
            kpts1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kpts2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)

            # add batch dimension
            kpts1 = kpts1.unsqueeze(0)  # 1 x N x 2
            kpts2 = kpts2.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)  # 1 x N x D
            desc2 = desc2.unsqueeze(0)

            # perform matching
            with torch.inference_mode():
                feats0 = {'keypoints': kpts1, 'descriptors': desc1}
                feats1 = {'keypoints': kpts2, 'descriptors': desc2}
                matches_output = matcher({'image0': feats0, 'image1': feats1})
                matches = rbd(matches_output)['matches']  # remove batch dim, shape (K,2)

            # filter and save
            if matches.shape[0] < min_matches:
                continue
            grp = f_match.require_group(key1)
            grp.create_dataset(key2, data=matches.cpu().numpy())

# === Step 3: Shortlist pairs by LightGlue match counts ===
def shortlist_by_matches(img_fnames, feature_dir, min_matches=30, brute_thresh=20):
    n = len(img_fnames)
    if n <= brute_thresh:
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    match_h5 = os.path.join(feature_dir, 'matches.h5')
    pairs = []
    # map basename to index
    name_to_idx = {os.path.basename(f): idx for idx, f in enumerate(img_fnames)}
    with h5py.File(match_h5, 'r') as f_match:
        for k1 in f_match.keys():
            for k2 in f_match[k1].keys():
                idx1, idx2 = name_to_idx[k1], name_to_idx[k2]
                cnt = f_match[k1][k2].shape[0]
                if cnt >= min_matches:
                    pairs.append(tuple(sorted((idx1, idx2))))
    return sorted(set(pairs))

# === Step 4: RANSAC init & ICP registration in Open3D ===
def feature_init(i, j, rgb_files, depth_files, intrinsic, feature_dir, device=torch.device('cpu')):
    # requires HDF5 keypoints & match files from ALIKED+LightGlue
    kp_h5 = os.path.join(feature_dir, 'keypoints.h5')
    match_h5 = os.path.join(feature_dir, 'matches.h5')
    with h5py.File(kp_h5, 'r') as fk, h5py.File(match_h5, 'r') as fm:
        key1 = os.path.basename(rgb_files[i])
        key2 = os.path.basename(rgb_files[j])
        if key1 not in fm or key2 not in fm[key1]:
            return np.eye(4)
        matches = fm[key1][key2][...]
        kpts1 = fk[key1][...]
        kpts2 = fk[key2][...]

    # load depth maps and back-project to 3D
    di = np.asarray(o3d.io.read_image(depth_files[i])) / 1000.0
    dj = np.asarray(o3d.io.read_image(depth_files[j])) / 1000.0
    Kmat = intrinsic.intrinsic_matrix
    pts1, pts2 = [], []
    for m in matches:
        u1, v1 = int(kpts1[m[0], 0]), int(kpts1[m[0], 1])
        u2, v2 = int(kpts2[m[1], 0]), int(kpts2[m[1], 1])
        z1, z2 = di[v1, u1], dj[v2, u2]
        if z1 <= 0 or z2 <= 0:
            continue
        x1 = (u1 - Kmat[0,2]) * z1 / Kmat[0,0]
        y1 = (v1 - Kmat[1,2]) * z1 / Kmat[1,1]
        x2 = (u2 - Kmat[0,2]) * z2 / Kmat[0,0]
        y2 = (v2 - Kmat[1,2]) * z2 / Kmat[1,1]
        pts1.append([x1, y1, z1])
        pts2.append([x2, y2, z2])
    if len(pts1) < 3:
        return np.eye(4)

    # build Open3D point clouds
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pts1)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pts2)))
    # identity correspondence array
    corr = np.stack([np.arange(len(pts1)), np.arange(len(pts1))], axis=1)

    # RANSAC-based registration
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src, tgt,
        o3d.utility.Vector2iVector(corr),
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result.transformation

# === Step 5: Full reconstruction pipeline ===
def reconstruct(rgb_folder, depth_folder, feature_dir):
    # Gather and sort file paths
    rgb  = sorted(glob.glob(os.path.join(rgb_folder, '*.png')), key=extract_number)
    depth= sorted(glob.glob(os.path.join(depth_folder, '*.png')), key=extract_number)
    assert len(rgb)==len(depth), "RGB/depth count mismatch"

    # Step 1: ALIKED feature extraction & matching
    detect_aliked(rgb, feature_dir)
    all_pairs = [(i, j) for i in range(len(rgb)) for j in range(i+1, len(rgb))]
    match_with_lightglue(rgb, feature_dir, all_pairs)
    pairs = shortlist_by_matches(rgb, feature_dir)

    # Step 2: Build Open3D point clouds
    # Camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480,
                                                  388.14, 387.46,
                                                  321.59, 240.76)
    max_coarse = 0.01 * 25
    max_fine   = 0.01 * 2.5
    voxel_size = max_fine / 2.5

    pcds, pcds_down = [], []
    for r, d in zip(rgb, depth):
        color = o3d.io.read_image(r)
        dep   = o3d.io.read_image(d)
        rgbd  = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, dep, depth_scale=1000.0, depth_trunc=2.0,
            convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        pcds.append(pcd)
        down = pcd.voxel_down_sample(voxel_size)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        pcds_down.append(down)

    # Step 3: Pose graph construction
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    for i, j in pairs:
        # initial guess
        if j == i + 1:
            init = np.eye(4) if i == 0 else pose_graph.nodes[i].pose @ np.linalg.inv(pose_graph.nodes[i-1].pose)
        else:
            init = feature_init(i, j, rgb, depth, intrinsic, feature_dir)
        # coarse ICP
        reg_c = o3d.pipelines.registration.registration_icp(
            pcds_down[i], pcds_down[j], max_coarse, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # fine ICP
        reg_f = o3d.pipelines.registration.registration_icp(
            pcds_down[i], pcds_down[j], max_fine, reg_c.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcds_down[i], pcds_down[j], max_fine, reg_f.transformation)
        uncertain = (j != i + 1)
        if j == i + 1:
            odom = reg_f.transformation @ np.eye(4)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(i, j,
                                                    reg_f.transformation,
                                                    info,
                                                    uncertain))

    # Step 4: Global optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    # Step 5: Merge transformed clouds
    ## OLD CODES
    combined = o3d.geometry.PointCloud()
    for idx, p in enumerate(pcds):
        p.transform(pose_graph.nodes[idx].pose)
        combined += p
    combined = combined.voxel_down_sample(voxel_size)
    return combined

    ## NEW CODES for Step 5 using TSDF for better merging
    # volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #     voxel_length=0.005,
    #     sdf_trunc=0.04,
    #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # for i, (r, d) in enumerate(zip(rgb, depth)):
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         o3d.io.read_image(r),
    #         o3d.io.read_image(d),
    #         depth_scale=1000.0,
    #         depth_trunc=2.0,
    #         convert_rgb_to_intensity=False)
    #     # use the optimized camera pose to integrate
    #     pose = np.linalg.inv(pose_graph.nodes[i].pose)
    #     volume.integrate(rgbd, intrinsic, pose)
    # pcd_tsdf = volume.extract_point_cloud()
    # pcd_tsdf.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # return pcd_tsdf

if __name__ == '__main__':
    out = reconstruct('test_rgb', 'test_depth', '.featureout')
    o3d.io.write_point_cloud('recon_lg.ply', out)
    # o3d.visualization.draw_geometries([out])