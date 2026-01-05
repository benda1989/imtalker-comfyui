import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def load_pose(smirk):
    pose = smirk["pose_params"]  # (N, 3)
    cam = smirk["cam"]           # (N, 3)
    return pose, cam

class AudioMotionSmirkGazeDataset(Dataset):
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames
        
        # Define subdirectories
        root_path = Path(opt.dataset_path)
        motion_dir = root_path / "motion"
        audio_dir = root_path / "audio"
        smirk_dir = root_path / "smirk"
        gaze_dir = root_path / "gaze"

        # Get all motion files as anchor, sorted to ensure deterministic split
        motion_files = sorted(list(motion_dir.glob("*.pt")))
        
        # Apply split (start:end)
        motion_files = motion_files[start:end]

        self.samples = []
        for motion_path in tqdm(motion_files, desc="Filtering valid samples"):
            file_stem = motion_path.stem
            
            # Construct paths for other modalities
            # Assuming audio/gaze are .npy and smirk is .pt based on previous loading logic
            audio_path = audio_dir / f"{file_stem}.npy"
            gaze_path = gaze_dir / f"{file_stem}.npy"
            smirk_path = smirk_dir / f"{file_stem}.pt"

            if not (audio_path.exists() and gaze_path.exists() and smirk_path.exists()):
                continue

            # Load to check lengths
            # Note: This might be slow for large datasets. 
            # Ideally, use a pre-calculated index if speed is critical.
            try:
                motion = torch.load(motion_path)
                smirk = torch.load(smirk_path)
                
                # Use mmap_mode='r' for faster length checking of numpy files
                audio = np.load(audio_path, mmap_mode='r')
                gaze = np.load(gaze_path, mmap_mode='r')
                
                motion_len = len(motion)
                audio_len = len(audio)
                gaze_len = len(gaze)
                smirk_len = len(smirk["pose_params"]) 
                
                min_len = min(motion_len, audio_len, gaze_len, smirk_len)

                if min_len >= self.required_len:
                    self.samples.append({
                        "motion_path": str(motion_path),
                        "audio_path": str(audio_path),
                        "smirk_path": str(smirk_path),
                        "gaze_path": str(gaze_path)
                    })
            except Exception as e:
                print(f"[Warning] Error checking file {file_stem}: {e}")
                continue

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root_path}")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def _get_full_clip(self, index):
        item = self.samples[index]
        
        motion = torch.load(item['motion_path'])
        audio = np.load(item['audio_path'], mmap_mode='r')
        gaze = np.load(item['gaze_path'], mmap_mode='r')
        smirk = torch.load(item['smirk_path'])
        pose, cam = load_pose(smirk)

        min_len = min(len(motion), len(audio), len(gaze), len(pose))
        start_idx = random.randint(0, min_len - self.required_len)
        end_idx = start_idx + self.required_len

        audio_seg = torch.from_numpy(audio[start_idx:end_idx].copy()).float()
        motion_seg = motion[start_idx:end_idx]
        gaze_seg = torch.from_numpy(gaze[start_idx:end_idx].copy()).float()
        pose_seg = pose[start_idx:end_idx]
        cam_seg = cam[start_idx:end_idx]

        motion_prev = motion_seg[:self.num_prev_frames]
        audio_prev = audio_seg[:self.num_prev_frames]
        gaze_prev = gaze_seg[:self.num_prev_frames]
        pose_prev = pose_seg[:self.num_prev_frames]
        cam_prev = cam_seg[:self.num_prev_frames]

        motion_clip = motion_seg[self.num_prev_frames:]
        audio_clip = audio_seg[self.num_prev_frames:]
        gaze_clip = gaze_seg[self.num_prev_frames:]
        pose_clip = pose_seg[self.num_prev_frames:]
        cam_clip = cam_seg[self.num_prev_frames:]

        return (motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, 
                gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev)

    def __getitem__(self, index):
        try:
            (motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, 
             gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev) = self._get_full_clip(index)
        except Exception as e:
            print(f"[Error] Failed to get clip for index {index}: {e}. Trying a random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        ref_idx = torch.randint(low=0, high=motion_seg.shape[0], size=(1,)).item()
        m_ref = motion_seg[ref_idx]

        return {
            "m_now": motion_clip,
            "a_now": audio_clip,
            "gaze_now": gaze_clip,
            "pose_now": pose_clip,
            "cam_now": cam_clip,
            "m_prev": motion_prev,
            "a_prev": audio_prev,
            "gaze_prev": gaze_prev,
            "pose_prev": pose_prev,
            "cam_prev": cam_prev,
            "m_ref": m_ref,
        }