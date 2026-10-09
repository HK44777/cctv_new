import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from src.utils.gpu_guard import get_device

# ── Add OpenGait to Python path ───────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_OPENGAIT     = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'OpenGait'))
_OPENGAIT_PKG = os.path.join(_OPENGAIT, 'opengait')
if _OPENGAIT     not in sys.path: sys.path.insert(0, _OPENGAIT)
if _OPENGAIT_PKG not in sys.path: sys.path.insert(1, _OPENGAIT_PKG)

from opengait.modeling.modules import (  # type: ignore[import]
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks, conv1x1, conv3x3,
    BasicBlock2D, BasicBlockP3D
)


# ── Constants (from skeletongait_GREW.yaml + pretreatment_heatmap.yaml) ───────
MAP_H, MAP_W = 64, 44    # SkeletonGait canonical input size
NORM_H       = 128        # Normalization canvas (CenterAndScaleNormalizer)
SIGMA        = 8.0        # Gaussian kernel sigma
MIN_CROP_H   = 80         # Skip pose estimation if person crop is too small
MIN_JOINTS   = 8          # Skip frame if fewer than this many joints are confident
JOINT_CONF   = 0.3        # Joint confidence threshold

# COCO-17 limb connections (matches OpenGait's default skeleton definition)
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (5, 7), (7, 9),
    (0, 6), (6, 8), (8, 10),
    (5, 11), (11, 13), (13, 15),
    (6, 12), (12, 14), (14, 16),
    (11, 12)
)


# ── Heatmap Generation ─────────────────────────────────────────────────────────
# These functions mirror OpenGait's GeneratePoseTarget class exactly.
# Channel 0 = keypoint heatmap (one Gaussian per joint, max-pooled)
# Channel 1 = limb heatmap    (Gaussian along each bone, max-pooled)

def _draw_joint_heatmap(kps, scores):
    """
    Draw a Gaussian blob at each keypoint position.
    All joint blobs are merged via element-wise max into a single channel.
    Returns (NORM_H, NORM_H) float32 array.
    """
    out = np.zeros((NORM_H, NORM_H), dtype=np.float32)
    for (mx, my), score in zip(kps, scores):
        if score < 1e-3:
            continue
        x0 = max(int(mx - 3 * SIGMA), 0);  x1 = min(int(mx + 3 * SIGMA) + 1, NORM_H)
        y0 = max(int(my - 3 * SIGMA), 0);  y1 = min(int(my + 3 * SIGMA) + 1, NORM_H)
        if x0 >= x1 or y0 >= y1:
            continue
        x = np.arange(x0, x1, dtype=np.float32)
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        patch = np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * SIGMA ** 2)) * score
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], patch)
    return out


def _draw_limb_heatmap(kps, scores, skeleton):
    """
    Draw a Gaussian along each limb (bone segment connecting two joints).
    All limb blobs are merged via element-wise max into a single channel.
    Returns (NORM_H, NORM_H) float32 array.
    """
    out = np.zeros((NORM_H, NORM_H), dtype=np.float32)
    for (si, ei) in skeleton:
        coeff = min(scores[si], scores[ei])
        if coeff < 1e-3:
            continue
        s, e = kps[si], kps[ei]
        x0 = max(int(min(s[0], e[0]) - 3 * SIGMA), 0)
        x1 = min(int(max(s[0], e[0]) + 3 * SIGMA) + 1, NORM_H)
        y0 = max(int(min(s[1], e[1]) - 3 * SIGMA), 0)
        y1 = min(int(max(s[1], e[1]) + 3 * SIGMA) + 1, NORM_H)
        if x0 >= x1 or y0 >= y1:
            continue

        x = np.arange(x0, x1, dtype=np.float32)
        y = np.arange(y0, y1, dtype=np.float32)[:, None]

        d2_s  = (x - s[0]) ** 2 + (y - s[1]) ** 2
        d2_e  = (x - e[0]) ** 2 + (y - e[1]) ** 2
        d2_ab = (s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2
        if d2_ab < 1:
            continue

        t = (d2_s - d2_e + d2_ab) / (2.0 * d2_ab)
        a_dom   = (t <= 0).astype(np.float32)
        b_dom   = (t >= 1).astype(np.float32)
        seg_dom = 1 - a_dom - b_dom

        proj_x = s[0] + t * (e[0] - s[0])
        proj_y = s[1] + t * (e[1] - s[1])

        px = x + np.zeros_like(y)
        py = np.zeros_like(x) + y

        d2_line = (px - proj_x) ** 2 + (py - proj_y) ** 2
        d2_seg  = a_dom * d2_s + b_dom * d2_e + seg_dom * d2_line
        patch   = np.exp(-d2_seg / (2.0 * SIGMA ** 2)) * coeff
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], patch)
    return out


def _normalize_keypoints(kps_xy, scores):
    """
    CenterAndScaleNormalizer: hip-center + scale to NORM_H space.
    Input:  kps_xy (17, 2) — raw pixel coords within crop
            scores (17,)   — confidence per joint
    Output: (17, 2) normalized coords in [~0, NORM_H] space, or None if degenerate
    """
    hip = (kps_xy[11] + kps_xy[12]) / 2.0
    centered = kps_xy - hip

    y_max = np.max(centered[:, 1])
    y_min = np.min(centered[:, 1])
    body_h = y_max - y_min
    if body_h < 1.0:
        return None  # degenerate — person not visible or too small

    scale = (NORM_H / 1.5) / body_h
    return centered * scale + NORM_H / 2.0


def _build_skeleton_map(kps_normalized, scores):
    """
    Build a 2-channel 64×44 skeleton map from normalized keypoints.
    Returns (2, MAP_H, MAP_W) float32 array.
    """
    kp_map   = _draw_joint_heatmap(kps_normalized, scores)
    limb_map = _draw_limb_heatmap(kps_normalized, scores, COCO_SKELETON)

    # Resize from (NORM_H, NORM_H) → (MAP_H, MAP_W)
    kp_map   = cv2.resize(kp_map,   (MAP_W, MAP_H))
    limb_map = cv2.resize(limb_map, (MAP_W, MAP_H))

    return np.stack([kp_map, limb_map], axis=0).astype(np.float32)  # (2, 64, 44)


# ── Inference-Only SkeletonGait Network ────────────────────────────────────────
# Mirrors DeepGaitV2 exactly but inherits nn.Module only (not BaseModel).
# This avoids torch.distributed, data loaders, and all training infrastructure.
# Config: skeletongait_GREW.yaml — in_channels=2, p3d backbone, 20000 classes.

class _SkeletonGaitNet(nn.Module):

    def __init__(self):
        super().__init__()
        in_ch      = 2
        channels   = [64, 128, 256, 512]
        layers_cfg = [1, 4, 4, 1]
        class_num  = 20000

        self._ip = channels[0]

        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_ch, channels[0], 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(
            self._make_2d(BasicBlock2D, channels[0], [1, 1], layers_cfg[0])
        )
        self.layer2 = self._make_p3d(BasicBlockP3D, channels[1], [2, 2], layers_cfg[1])
        self.layer3 = self._make_p3d(BasicBlockP3D, channels[2], [2, 2], layers_cfg[2])
        self.layer4 = self._make_p3d(BasicBlockP3D, channels[3], [1, 1], layers_cfg[3])

        self.TP      = PackSequenceWrapper(torch.max)
        self.HPP     = HorizontalPoolingPyramid(bin_num=[16])
        self.FCs     = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=class_num)

    def _make_2d(self, block, planes, stride, num_blocks):
        if max(stride) > 1 or self._ip != planes * block.expansion:
            ds = nn.Sequential(conv1x1(self._ip, planes * block.expansion, stride=stride),
                               nn.BatchNorm2d(planes * block.expansion))
        else:
            ds = lambda x: x
        blks = [block(self._ip, planes, stride=stride, downsample=ds)]
        self._ip = planes * block.expansion
        for _ in range(1, num_blocks):
            blks.append(block(self._ip, planes, stride=[1, 1]))
        return nn.Sequential(*blks)

    def _make_p3d(self, block, planes, stride, num_blocks):
        if max(stride) > 1 or self._ip != planes * block.expansion:
            ds = nn.Sequential(
                nn.Conv3d(self._ip, planes * block.expansion,
                          kernel_size=[1, 1, 1], stride=[1, *stride],
                          padding=[0, 0, 0], bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        else:
            ds = lambda x: x
        blks = [block(self._ip, planes, stride=stride, downsample=ds)]
        self._ip = planes * block.expansion
        for _ in range(1, num_blocks):
            blks.append(block(self._ip, planes, stride=[1, 1]))
        return nn.Sequential(*blks)

    def forward(self, x):
        """
        x: (B, S, C, H, W)  — B=batch, S=30 frames, C=2 channels, H=64, W=44
        returns: (B, 256)    — L2-normalized gait embedding per person
        """
        # (B, S, C, H, W) → (B, C, S, H, W) for the 3D backbone
        sils = x.transpose(1, 2).contiguous()

        out = self.layer0(sils)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Temporal max-pooling
        out = self.TP(out, seqL=None, options={"dim": 2})[0]

        feat  = self.HPP(out)    # (B, 512, 16)
        embed = self.FCs(feat)   # (B, 256, 16)

        # Average over 16 horizontal parts → (B, 256)
        return embed.mean(dim=2)


# ── GaitExtractor ──────────────────────────────────────────────────────────────

class GaitExtractor:

    def __init__(self,
                 pose_model_path='yolov8n-pose.pt',
                 gait_weights_path='src/pipeline/extractors/SkeletonGait-180000.pt',
                 strict_gpu=True):

        # strict_gpu=True  → require GPU (ingestion phase on Colab)
        # strict_gpu=False → allow CPU fallback (query server on local machine)
        self.device = get_device(strict=strict_gpu)
        # YOLO device string: 'cuda:0' for GPU, 'cpu' for CPU
        self.yolo_device = 'cuda:0' if self.device.type == 'cuda' else 'cpu'

        # Step A: Pose estimator (extracts 17 COCO keypoints per person crop)
        self.pose_model = YOLO(pose_model_path)

        # Step B: SkeletonGait embedding model (inference-only)
        self.gait_model = _SkeletonGaitNet().to(self.device)

        if os.path.exists(gait_weights_path):
            ckpt = torch.load(gait_weights_path, map_location=self.device)
            state_dict = ckpt['model']
            # Strip DDP 'module.' prefix if the checkpoint was saved from a DDP run
            clean_sd = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            self.gait_model.load_state_dict(clean_sd, strict=True)
            print(f"Initialized Gait Extractor (SkeletonGait loaded from {gait_weights_path})")
        else:
            print(f"WARNING: SkeletonGait weights not found at '{gait_weights_path}'. "
                  f"Running with random weights (embeddings will not be discriminative).")

        self.gait_model.eval()

    def extract_skeleton(self, person_crop):
        """
        Run YOLOv8-Pose on a person crop and produce a 2-channel skeleton map.

        Input:  person_crop — BGR numpy array (H, W, 3), already cropped to person bbox
        Output: (2, 64, 44) float32 numpy array — the skeleton map, or None if:
                  • crop is too small for reliable pose estimation
                  • fewer than MIN_JOINTS keypoints are confident
                  • skeleton normalization degenerates (person not walking upright)
        """
        h, w = person_crop.shape[:2]

        # Gate 1: crop too small — pose model will fail on tiny people
        if h < MIN_CROP_H:
            return None

        # Run pose estimator on the configured device (GPU or CPU)
        results = self.pose_model(person_crop, device=self.yolo_device, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints.data  # (num_persons, 17, 3) tensor
        if kpts.shape[0] == 0:
            return None

        # Use keypoints of the first (most confident) detected person
        kpts   = kpts[0].cpu().numpy()                  # (17, 3): x, y, conf
        kps_xy = kpts[:, :2].astype(np.float32)         # (17, 2)
        scores = kpts[:, 2].astype(np.float32)          # (17,)

        # Gate 2: too few reliable joints
        if np.sum(scores > JOINT_CONF) < MIN_JOINTS:
            return None

        # Zero out low-confidence joints so they don't pollute the heatmap
        low_conf = scores < JOINT_CONF
        scores[low_conf] = 0.0
        kps_xy[low_conf] = 0.0

        # Normalize: hip-center + body-scale (CenterAndScaleNormalizer)
        kps_norm = _normalize_keypoints(kps_xy, scores)
        if kps_norm is None:
            return None

        # Build 2-channel 64×44 skeleton map
        return _build_skeleton_map(kps_norm, scores)

    def extract_skeleton_batch(self, person_crops):
        """
        Run YOLOv8-Pose on a batch of person crops and produce a list of skeleton maps.
        """
        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(person_crops):
            if crop is not None and crop.shape[0] >= MIN_CROP_H:
                valid_crops.append(crop)
                valid_indices.append(i)
                
        if not valid_crops:
            return [None] * len(person_crops)
            
        # YOLO batch inference
        results = self.pose_model(valid_crops, device=self.yolo_device, verbose=False)
        
        skeletons = [None] * len(person_crops)
        for idx, res in zip(valid_indices, results):
            if res.keypoints is None: continue
            kpts = res.keypoints.data
            if kpts.shape[0] == 0: continue
            
            kpts = kpts[0].cpu().numpy()
            kps_xy = kpts[:, :2].astype(np.float32)
            scores = kpts[:, 2].astype(np.float32)
            
            if np.sum(scores > JOINT_CONF) < MIN_JOINTS:
                continue
                
            low_conf = scores < JOINT_CONF
            scores[low_conf] = 0.0
            kps_xy[low_conf] = 0.0
            
            kps_norm = _normalize_keypoints(kps_xy, scores)
            if kps_norm is None: continue
            
            skeletons[idx] = _build_skeleton_map(kps_norm, scores)
            
        return skeletons


    def extract_gait_vector(self, skeleton_buffer):
        """
        Convert a buffer of >= 15 skeleton maps into a 256-d gait embedding.

        Input:  skeleton_buffer — list of N × (2, 64, 44) float32 numpy arrays
        Output: list of 256 floats (L2-normalized), or None if buffer < 15
        """
        if len(skeleton_buffer) < 15:
            return None

        # Stack all available maps → (N, 2, 64, 44)
        seq = np.stack(skeleton_buffer, axis=0).astype(np.float32)

        # Add batch dim → (1, N, 2, 64, 44)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.gait_model(tensor)   # (1, 256)

        vec = embedding.squeeze(0).cpu().numpy()  # (256,)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()
