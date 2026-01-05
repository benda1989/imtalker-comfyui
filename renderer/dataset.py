import os
import cv2
import torch
import numpy as np
import itertools
import bisect
from glob import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

def create_eye_mouth_mask(
    landmarks_68: np.ndarray,
    image_size: int = 512,
    eye_erosion_iters: int = 1,
    eye_dilate_iters: int = 1,
    mouth_dilate_iters: int = 2
    ): 
    """
    Create binary masks for eyes and mouth based on 68 facial landmarks.
    """
    # Initialize empty masks
    eye_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mouth_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Indices for left/right eye in 68-landmarks
    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]

    # Outer lips only: [48..59], ignoring [60..67] (inner lips)
    outer_mouth_idx = list(range(48, 60))

    # Convert normalized coords -> pixel coords
    def to_px_coords(idx_list):
        return [
            (int(landmarks_68[i, 0] * image_size),
             int(landmarks_68[i, 1] * image_size))
            for i in idx_list
        ]

    left_eye_pts = to_px_coords(left_eye_idx)
    right_eye_pts = to_px_coords(right_eye_idx)
    mouth_pts = to_px_coords(outer_mouth_idx)

    def fill_polygon(mask, pts):
        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts_array, 255)

    # Fill left eye / right eye
    fill_polygon(eye_mask, left_eye_pts)
    fill_polygon(eye_mask, right_eye_pts)

    # Mouth: use convex hull on outer-lip points
    mouth_pts_array = np.array(mouth_pts, dtype=np.int32)
    mouth_hull = cv2.convexHull(mouth_pts_array)
    cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)

    # Morphological ops
    kernel= np.ones((7, 7), dtype=np.uint8)

    # Eye region
    if eye_erosion_iters > 0:
        eye_mask = cv2.erode(eye_mask, kernel, iterations=eye_erosion_iters)
    if eye_dilate_iters > 0:
        eye_mask = cv2.dilate(eye_mask, kernel, iterations=eye_dilate_iters)

    # Mouth region
    if mouth_dilate_iters > 0:
        mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=mouth_dilate_iters)

    # Convert to float32 binary in [0,1], shape (H,W,1)
    eye_mask = (eye_mask > 0).astype(np.float32)[..., None]
    mouth_mask = (mouth_mask > 0).astype(np.float32)[..., None]

    return eye_mask, mouth_mask

class TFDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir (str or Path): Path to the dataset root. 
                                    Expects structure: {root_dir}_video_frame and {root_dir}_lmd
                                    or simply folders inside root_dir if reorganized.
                                    (Adjusted below to match your previous path structure)
            split (str): 'train', 'val', or 'test'.
        """
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.split = split
        self.root_path = Path(root_dir)
        
        # Define transform for 512x512 only
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        self.meta_list = self._load_metadata()

    def _load_metadata(self):
        # Assuming folder structure: root_dir/video_frame and root_dir/lmd
        # Adjust logic if your folders are named differently (e.g., vfhq_video_frame)
        video_root = self.root_path / "video_frame"
        lmd_root = self.root_path / "lmd"
        
        # If your folders strictly follow "{name}_video_frame", pass the parent dir as root_dir
        # and modify these lines, or pass the specific subfolders. 
        # Here I assume root_dir contains the subfolders directly.
        
        if not video_root.exists():
            # Fallback for the specific naming convention in your example:
            # e.g. /mnt/buffer/chenbo/vfhq_video_frame -> passed /mnt/buffer/chenbo/vfhq
            name = self.root_path.name
            parent = self.root_path.parent
            video_root = parent / f"{name}_video_frame"
            lmd_root = parent / f"{name}_lmd"

        clip_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
        
        # Simple split strategy
        if self.split == 'train':
            clip_dirs = clip_dirs[:500] 
        else:
            clip_dirs = clip_dirs[500:] # Or whatever split logic you prefer

        meta_list = []
        for clip_path in tqdm(clip_dirs, desc=f'Processing {self.root_path.name}'):
            lmd_file = lmd_root / f"{clip_path.name}.txt"
            
            # Filter for images
            frame_files = sorted(
                [f for f in clip_path.iterdir() if f.suffix.lower() in ['.png', '.jpg']],
                key=lambda x: int(x.stem.split('_')[-1]) if '_' in x.stem else x.stem
            )
            frame_count = len(frame_files)

            if frame_count <= 25 or not lmd_file.is_file():
                continue

            meta_list.append({
                'dir': str(clip_path),
                'frames': frame_files,
                'lmd': str(lmd_file)
            })

        return meta_list

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]
        frame_paths = meta['frames']
        lmd_path = meta['lmd']
    
        # Read landmarks
        landmarks = self.read_landmark_info(lmd_path, pixel_scale=(512, 512))
    
        # Align length
        min_len = min(len(frame_paths), len(landmarks))
        if min_len < 2:
            return self.__getitem__((idx + 1) % len(self.meta_list))
    
        frame_paths = frame_paths[:min_len]
        landmarks = landmarks[:min_len]
    
        # Randomly select two distinct frames
        f_id0, f_id1 = np.random.choice(min_len, size=2, replace=False)
        image_0 = Image.open(frame_paths[f_id0]).convert("RGB")
        image_1 = Image.open(frame_paths[f_id1]).convert("RGB")
        
        mask_eye_0, mask_mouth_0 = create_eye_mouth_mask(landmarks[f_id0], 512, 0, 2, 2)
        mask_eye_1, mask_mouth_1 = create_eye_mouth_mask(landmarks[f_id1], 512, 0, 2, 2)

        # ------------------------------
        # Negative sampling (from a different video)
        # ------------------------------
        neg_idx = np.random.randint(len(self.meta_list))
        while neg_idx == idx:  # Avoid same video
            neg_idx = np.random.randint(len(self.meta_list))
        
        neg_meta = self.meta_list[neg_idx]
        neg_frame_paths = neg_meta['frames']
        neg_lmd_path = neg_meta['lmd']

        neg_landmarks = self.read_landmark_info(neg_lmd_path, pixel_scale=(512, 512))
        neg_len = min(len(neg_frame_paths), len(neg_landmarks))
        
        if neg_len > 0:
            neg_frame_id = np.random.randint(neg_len)
            neg_image = Image.open(neg_frame_paths[neg_frame_id]).convert("RGB")
            neg_mask_eye, neg_mask_mouth = create_eye_mouth_mask(neg_landmarks[neg_frame_id], 512, 0, 2, 2)
        else:
            # Fallback if negative sample is invalid (rare)
            neg_image = image_0
            neg_mask_eye, neg_mask_mouth = mask_eye_0, mask_mouth_0

        return {
            "image_0": self.transform(image_0),
            "image_1": self.transform(image_1),
            "mask_eye_0": torch.tensor(mask_eye_0).permute(2, 0, 1), # (C, H, W)
            "mask_mouth_0": torch.tensor(mask_mouth_0).permute(2, 0, 1),
            "mask_eye_1": torch.tensor(mask_eye_1).permute(2, 0, 1),
            "mask_mouth_1": torch.tensor(mask_mouth_1).permute(2, 0, 1),
            # Negative samples
            "neg_image": self.transform(neg_image),
            "neg_mask_eye": torch.tensor(neg_mask_eye).permute(2, 0, 1),
            "neg_mask_mouth": torch.tensor(neg_mask_mouth).permute(2, 0, 1)
        }

    def read_landmark_info(self, lmd_path, pixel_scale):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for line in lmd_lines:
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:] # Skip filename
            lmd_obj = []
            for coord_pair in coords:
                x, y = coord_pair.split('_')
                lmd_obj.append((float(x)/pixel_scale[0], float(y)/pixel_scale[1]))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)