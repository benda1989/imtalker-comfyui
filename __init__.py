import os
import sys
import tempfile
import subprocess
import numpy as np
import cv2
import torch
import librosa
import face_alignment
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator.FM import FMGenerator
from renderer.models import IMTRenderer

class Config:
    """Configuration class matching the original app settings"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.fix_noise_seed = False
        self.renderer_path = "./checkpoints/renderer.ckpt"
        self.generator_path = "./checkpoints/generator.ckpt"
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"
        self.input_size = 512
        self.input_nc = 3
        self.fps = 25.0
        self.rank = "cuda"
        self.sampling_rate = 16000
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.nfe = 10
        self.torchdiffeq_ode_method = 'euler'
        self.a_cfg_scale = 3.0
        self.swin_res_threshold = 128
        self.window_size = 8


class Processor:
    """Data processing class from the original app"""
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        print(f"GKK·IMTalker: Loading Face Alignment...")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)
        print(f"GKK·IMTalker: Loading local wav2vec from {opt.wav2vec_model_path}")
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path,local_files_only=True)
        self.transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    def process_img(self, img: Image.Image) -> Image.Image:
        """Process and crop image to focus on face"""
        img_arr = np.array(img)
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
        h, w = img_arr.shape[:2]
        
        try:
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
            if bboxes is None or len(bboxes) == 0:
                bboxes = self.fa.face_detector.detect_from_image(img_arr)
        except Exception as e:
            print(f"GKK·IMTalker: Face detection failed: {e}")
            bboxes = None
            
        valid_bboxes = []
        if bboxes is not None:
            valid_bboxes = [(int(x1), int(y1), int(x2), int(y2), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.5]
            
        if not valid_bboxes:
            print("GKK·IMTalker: No face detected. Using center crop.")
            cx, cy = w // 2, h // 2
            half = min(w, h) // 2
            x1_new, x2_new = cx - half, cx + half
            y1_new, y2_new = cy - half, cy + half
        else:
            x1, y1, x2, y2, _ = valid_bboxes[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w_face = x2 - x1
            h_face = y2 - y1
            half_side = int(max(w_face, h_face) * 0.8)
            x1_new = cx - half_side
            y1_new = cy - half_side
            x2_new = cx + half_side
            y2_new = cy + half_side
            
            # Boundary checks
            if x1_new < 0: x2_new += (0 - x1_new); x1_new = 0
            if y1_new < 0: y2_new += (0 - y1_new); y1_new = 0
            if x2_new > w: x1_new -= (x2_new - w); x2_new = w
            if y2_new > h: y1_new -= (y2_new - h); y2_new = h
            x1_new = max(0, x1_new); y1_new = max(0, y1_new)
            x2_new = min(w, x2_new); y2_new = min(h, y2_new)
            
            curr_w = x2_new - x1_new; curr_h = y2_new - y1_new
            min_side = min(curr_w, curr_h)
            x2_new = x1_new + min_side; y2_new = y1_new + min_side
            
        crop_img = img_arr[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
        crop_pil = Image.fromarray(crop_img)
        return crop_pil.resize((self.opt.input_size, self.opt.input_size))

    def process_audio(self, path: str) -> torch.Tensor:
        """Process audio file for wav2vec features"""
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors='pt').input_values[0]

    def crop_video_stable(self, from_mp4_file_path, to_mp4_file_path, expanded_ratio=0.6, skip_per_frame=15):
        """Stable video cropping based on face detection analysis"""
        if os.path.exists(to_mp4_file_path):
            os.remove(to_mp4_file_path)
        
        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"GKK·IMTalker: Analyzing video for stable cropping: {from_mp4_file_path}")
        
        while video.isOpened():
            success = video.grab()
            if not success:
                break
            if index % skip_per_frame == 0:
                success, frame = video.retrieve()
                if not success:
                    break
                h, w = frame.shape[:2]
                mult = 360.0 / h
                resized_frame = cv2.resize(frame, dsize=(0, 0), fx=mult, fy=mult, interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
                try:
                    detected_bboxes = self.fa.face_detector.detect_from_image(resized_frame)
                except:
                    detected_bboxes = None
                    
                current_frame_bboxes = []
                if detected_bboxes is not None:
                    for d_box in detected_bboxes:
                        bx1, by1, bx2, by2, score = d_box
                        if score > 0.5:
                            current_frame_bboxes.append([int(bx1 / mult), int(by1 / mult), int(bx2 / mult), int(by2 / mult), score])
                            
                if len(current_frame_bboxes) > 0:
                    max_bboxes = max(current_frame_bboxes, key=lambda bbox: bbox[2] - bbox[0])
                    bboxes_lists.append(max_bboxes)
            index += 1
        video.release()
        
        x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
        for bbox in bboxes_lists:
            x1, y1, x2, y2 = bbox[:4]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_lists.append(x_center)
            y_center_lists.append(y_center)
            width_lists.append(x2 - x1)
            height_lists.append(y2 - y1)
            
        if not (x_center_lists and y_center_lists and width_lists and height_lists):
            import shutil
            shutil.copy(from_mp4_file_path, to_mp4_file_path)
            return
            
        x_center = sorted(x_center_lists)[len(x_center_lists) // 2]
        y_center = sorted(y_center_lists)[len(y_center_lists) // 2]
        median_width = sorted(width_lists)[len(width_lists) // 2]
        median_height = sorted(height_lists)[len(height_lists) // 2]
        expanded_width = int(median_width * (1 + expanded_ratio))
        expanded_height = int(median_height * (1 + expanded_ratio))
        fixed_cropped_width = min(max(expanded_width, expanded_height), width, height)
        x1, y1 = int(x_center - fixed_cropped_width / 2), int(y_center - fixed_cropped_width / 2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        if x1 + fixed_cropped_width > width:
            x1 = width - fixed_cropped_width
        if y1 + fixed_cropped_width > height:
            y1 = height - fixed_cropped_width
            
        target_size = self.opt.input_size
        
        cmd = (f'ffmpeg -i "{from_mp4_file_path}" -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},scale={target_size}:{target_size}:flags=lanczos" -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 128k "{to_mp4_file_path}" -y -loglevel error')
        if os.system(cmd) != 0:
            print("GKK·IMTalker: FFmpeg command failed. Copying original.")
            import shutil
            shutil.copy(from_mp4_file_path, to_mp4_file_path)
    
    def save_video(self, vid_tensor, fps, audio_path=None):
        """Save video tensor to file with optional audio"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            raw_path = tmp.name
        
        if vid_tensor.dim() == 4:
            vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            vid = vid_tensor.detach().cpu().numpy()
            
        if vid.min() < 0:
            vid = (vid + 1) / 2
        vid = np.clip(vid, 0, 1)
        vid = (vid * 255).astype(np.uint8)
        
        height, width = vid.shape[1], vid.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
        
        for frame in vid:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        
        if audio_path:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
                final_path = tmp_out.name
            cmd = f'ffmpeg -y -i "{raw_path}" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}" -loglevel error'
            subprocess.call(cmd, shell=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return final_path
        else:
            return raw_path

class base:
    FUNCTION = "run"
    CATEGORY = "GKK·IMTalker"
class ModelNode(base):
    """ComfyUI node for loading IMTalker models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "renderer_path": ("STRING", {"default": "./checkpoints/renderer.ckpt"}),
                "generator_path": ("STRING", {"default": "./checkpoints/generator.ckpt"}),
                "wav2vec_path": ("STRING", {"default": "./checkpoints/wav2vec2-base-960h"}),
            }
        }
    
    RETURN_TYPES = ("IMTALKER_MODELS",)
    RETURN_NAMES = ("models",)

    def __init__(self):
        self.models = None
        self.config = None
    
    def run(self, renderer_path, generator_path, wav2vec_path):
        """Load IMTalker renderer and generator models"""
        if self.models is not None:
            return (self.models,)
            
        from folder_paths import models_dir
        config = Config()
        config.renderer_path = os.path.join(models_dir, renderer_path)
        config.generator_path = os.path.join(models_dir, generator_path)
        config.wav2vec_model_path = os.path.join(models_dir, wav2vec_path)
 
        print("GKK·IMTalker: Loading IMTalker models...")
        
        data_processor = Processor(config)
        renderer = IMTRenderer(config).to(config.device)
        generator = FMGenerator(config).to(config.device)
        
        if not os.path.exists(config.renderer_path) or not os.path.exists(config.generator_path):
            raise FileNotFoundError("Model checkpoints not found.")
            
        self._load_ckpt(renderer, config.renderer_path, "gen.")
        self._load_fm_ckpt(generator, config.generator_path, config.device)
        
        renderer.eval()
        generator.eval()
        
        self.models = {
            'renderer': renderer,
            'generator': generator,
            'data_processor': data_processor,
            'config': config
        }
        
        print("GKK·IMTalker: models loaded successfully.")
        return (self.models,)
    
    def _load_ckpt(self, model, path, prefix="gen."):
        """Load checkpoint for renderer"""
        if not os.path.exists(path):
            print(f"GKK·IMTalker: Checkpoint {path} not found.")
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        model.load_state_dict(clean_state_dict, strict=False)

    def _load_fm_ckpt(self, model, path, device):
        """Load checkpoint for generator"""
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        prefix = 'model.'
        clean_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(device))


class ProcessNode(base):
    """Main processing node that handles both audio-driven and video-driven modes"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("IMTALKER_MODELS",),
                "source_image": ("IMAGE",),
                "crop_face": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "audio": ("AUDIO",),  # ComfyUI LoadAudio format
                "video": ("IMAGE",),  # VideoHelperSuite format
                "video_info": ("VHS_VIDEOINFO", {}),  # VideoHelperSuite info
                "fps": ("FLOAT", {"default": 25.0, "min": 10.0, "max": 60.0}),
                "nfe": ("INT", {"default": 10, "min": 5, "max": 50}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT")  # VideoHelperSuite expects IMAGE format + fps
    RETURN_NAMES = ("video_frames", "fps")
    
    def run(self, models, source_image, crop_face, seed, audio=None, video=None, video_info=None, fps=25.0, nfe=10, cfg_scale=2.0):
        """Main processing function that auto-detects mode based on inputs"""
        
        # Validate and fix parameters
        if isinstance(nfe, str):
            nfe = 10 if nfe == "" else int(nfe)
        if isinstance(cfg_scale, str):
            cfg_scale = 2.0 if cfg_scale == "" else float(cfg_scale)
        if isinstance(fps, str):
            fps = 25.0 if fps == "" else float(fps)
        
        # Clamp values to valid ranges
        nfe = max(5, min(50, nfe))
        cfg_scale = max(1.0, min(10.0, cfg_scale))
        fps = max(10.0, min(60.0, fps))
        
        renderer = models['renderer']
        generator = models['generator'] 
        data_processor = models['data_processor']
        config = models['config']
        
        # Convert ComfyUI image format to PIL
        if isinstance(source_image, torch.Tensor):
            # Handle different tensor dimensions
            if source_image.dim() == 4:
                source_image = source_image.squeeze(0)  # Remove batch dimension
            elif source_image.dim() == 2:
                source_image = source_image.unsqueeze(-1).repeat(1, 1, 3)  # Convert grayscale to RGB
            
            # Ensure correct format: [H, W, C]
            if source_image.dim() == 3:
                if source_image.shape[0] == 3 or source_image.shape[0] == 1:  # [C, H, W]
                    source_image = source_image.permute(1, 2, 0)
                # Now should be [H, W, C]
                
            source_image = source_image.cpu().numpy()
        
        # Validate and fix image shape
        if source_image.ndim == 2:
            source_image = np.stack([source_image] * 3, axis=-1)  # Convert grayscale to RGB
        elif source_image.ndim == 3 and source_image.shape[2] == 1:
            source_image = np.repeat(source_image, 3, axis=2)  # Convert single channel to RGB
        elif source_image.ndim == 3 and source_image.shape[2] == 4:
            source_image = source_image[:, :, :3]  # Remove alpha channel
        
        # Ensure proper value range
        if source_image.max() <= 1.0:
            source_image = (source_image * 255).astype(np.uint8)
        else:
            source_image = source_image.astype(np.uint8)
            
        # Validate final shape
        if source_image.ndim != 3 or source_image.shape[2] != 3:
            raise ValueError(f"Invalid image shape after processing: {source_image.shape}, expected [H, W, 3]")
            
        source_pil = Image.fromarray(source_image)
        
        # Determine mode based on inputs
        if audio is not None:
            result = self._run_audio_inference(
                renderer, generator, data_processor, config,
                source_pil, audio, fps, crop_face, seed, nfe, cfg_scale
            )
            return (result, fps)  # Audio mode: use user-specified fps
        elif video is not None:
            result = self._run_video_inference_vhs(
                renderer, data_processor, config,
                source_pil, video, crop_face
            )
            # Video mode: get fps from driving video info
            video_fps = fps  # fallback
            if video_info and 'fps' in video_info:
                video_fps = video_info['fps']
            return (result, video_fps)
        else:
            raise ValueError("Provide either audio (for audio-driven) or video (for video-driven).")
    
    @torch.no_grad()
    def _run_audio_inference(self, renderer, generator, data_processor, config, img_pil, audio_data, fps, crop, seed, nfe, cfg_scale):
        """Audio-driven inference mode"""
        s_pil = data_processor.process_img(img_pil) if crop else img_pil.resize((config.input_size, config.input_size))
        s_tensor = data_processor.transform(s_pil).unsqueeze(0).to(config.device)
        
        # Handle ComfyUI AUDIO format
        if isinstance(audio_data, dict) and 'waveform' in audio_data:
            # ComfyUI LoadAudio format: {"waveform": tensor, "sample_rate": int}
            waveform = audio_data['waveform']
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            if waveform.ndim > 1:
                waveform = waveform[0]  # Take first channel if stereo
            
            # Resample to 16kHz if needed
            sample_rate = audio_data.get('sample_rate', 16000)
            if sample_rate != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            
            # Process with wav2vec
            a_tensor = data_processor.wav2vec_preprocessor(waveform, sampling_rate=16000, return_tensors='pt').input_values[0].unsqueeze(0).to(config.device)
        elif isinstance(audio_data, str):
            # Fallback: file path
            a_tensor = data_processor.process_audio(audio_data).unsqueeze(0).to(config.device)
        else:
            raise ValueError(f"Unsupported audio format: {type(audio_data)}")
        
        data = {'s': s_tensor, 'a': a_tensor, 'pose': None, 'cam': None, 'gaze': None, 'ref_x': None}
        f_r, g_r = renderer.dense_feature_encoder(s_tensor)
        t_lat = renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data['ref_x'] = t_lat
        
        torch.manual_seed(seed)
        sample = generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        
        d_hat = []
        T = sample.shape[1]
        ta_r = renderer.adapt(t_lat, g_r)
        m_r = renderer.latent_token_decoder(ta_r)
        
        for t in range(T):
            ta_c = renderer.adapt(sample[:, t, ...], g_r)
            m_c = renderer.latent_token_decoder(ta_c)
            out_frame = renderer.decode(m_c, m_r, f_r)
            d_hat.append(out_frame)
            
        # Convert to VideoHelperSuite format: [Frames, Height, Width, Channels]
        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)
        if vid_tensor.dim() == 4:
            vid_tensor = vid_tensor.permute(0, 2, 3, 1)  # [F,C,H,W] -> [F,H,W,C]
        print(f"GKK·IMTalker: Generated {vid_tensor.shape[0]} frames for audio-driven inference at {fps} fps")
        
        return vid_tensor
    
    @torch.no_grad()
    def _run_video_inference_vhs(self, renderer, data_processor, config, source_img_pil, video_tensor, crop):
        """Video-driven inference mode for VideoHelperSuite format"""
        s_pil = data_processor.process_img(source_img_pil) if crop else source_img_pil.resize((config.input_size, config.input_size))
        s_tensor = data_processor.transform(s_pil).unsqueeze(0).to(config.device)
        f_r, g_r = renderer.dense_feature_encoder(s_tensor)
        t_lat = renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        ta_r = renderer.adapt(t_lat, g_r)
        m_r = renderer.latent_token_decoder(ta_r)
        
        vid_results = []
        
        # Process video tensor from VideoHelperSuite
        if video_tensor.dim() == 4:
            for i in range(video_tensor.shape[0]):
                frame = video_tensor[i]
                # Convert to numpy and then PIL
                if frame.max() <= 1.0:
                    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                else:
                    frame_np = frame.cpu().numpy().astype(np.uint8)
                
                frame_pil = Image.fromarray(frame_np).resize((config.input_size, config.input_size))
                d_tensor = data_processor.transform(frame_pil).unsqueeze(0).to(config.device)
                
                t_c = renderer.latent_token_encoder(d_tensor)
                if isinstance(t_c, tuple):
                    t_c = t_c[0]
                ta_c = renderer.adapt(t_c, g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                out = renderer.decode(m_c, m_r, f_r)
                vid_results.append(out.cpu())
            
        if not vid_results:
            raise Exception("Driving video processing failed.")
            
        # Stack and convert to VideoHelperSuite format
        vid_tensor = torch.cat(vid_results, dim=0)
        if vid_tensor.dim() == 4:
            vid_tensor = vid_tensor.permute(0, 2, 3, 1)  # [F,C,H,W] -> [F,H,W,C]
        
        return vid_tensor

    @torch.no_grad()
    def _run_video_inference(self, renderer, data_processor, config, source_img_pil, video_frames, fps, video_path, crop):
        """Video-driven inference mode"""
        s_pil = data_processor.process_img(source_img_pil) if crop else source_img_pil.resize((config.input_size, config.input_size))
        s_tensor = data_processor.transform(s_pil).unsqueeze(0).to(config.device)
        
        # Fix method names to match the actual renderer interface  
        f_r, g_r = renderer.dense_feature_encoder(s_tensor)
        t_lat = renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        ta_r = renderer.adapt(t_lat, g_r)
        m_r = renderer.latent_token_decoder(ta_r)
        
        # Handle video cropping if requested and path is provided
        final_driving_path = None
        temp_crop_video = None
        
        if video_path and crop:
            # Create temporary file for cropped video
            temp_crop_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            data_processor.crop_video_stable(video_path, temp_crop_video)
            final_driving_path = temp_crop_video
        elif video_path:
            final_driving_path = video_path
        
        vid_results = []
        
        # If we have a video path, read from file (more memory efficient)
        if final_driving_path:
            cap = cv2.VideoCapture(final_driving_path)
            actual_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else fps
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame).resize((config.input_size, config.input_size))
                d_tensor = data_processor.transform(frame_pil).unsqueeze(0).to(config.device)
                t_c = renderer.latent_token_encoder(d_tensor)
                if isinstance(t_c, tuple):
                    t_c = t_c[0]
                ta_c = renderer.adapt(t_c, g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                out = renderer.decode(m_c, m_r, f_r)
                vid_results.append(out.cpu())
            cap.release()
            
            # Clean up temporary file
            if temp_crop_video and os.path.exists(temp_crop_video):
                os.remove(temp_crop_video)
        else:
            # Process from frame list (fallback)
            actual_fps = fps
            for frame in video_frames:
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame).resize((config.input_size, config.input_size))
                else:
                    frame_pil = frame.resize((config.input_size, config.input_size))
                    
                d_tensor = data_processor.transform(frame_pil).unsqueeze(0).to(config.device)
                t_c = renderer.latent_token_encoder(d_tensor)
                if isinstance(t_c, tuple):
                    t_c = t_c[0]
                ta_c = renderer.adapt(t_c, g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                out = renderer.decode(m_c, m_r, f_r)
                vid_results.append(out.cpu())
            
        if not vid_results:
            raise Exception("Driving video processing failed.")
            
        vid_tensor = torch.cat(vid_results, dim=0)
        
        # Save video with audio if path is provided
        if video_path:
            return data_processor.save_video(vid_tensor, actual_fps, video_path)
        else:
            return vid_tensor

NODE_CLASS_MAPPINGS = {
    "IMTalkerModel": ModelNode,
    "IMTalkerProcess": ProcessNode,
}

