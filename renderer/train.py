import argparse
import os
import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Assuming these are your custom modules
from renderer.models import IMTRenderer
from renderer.discriminator import PatchDiscriminator
from vgg19_mask import VGGLoss_mask
from dataset import TFDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class IMFSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        
        self.gen = IMTRenderer(args)
        self.disc = PatchDiscriminator()

        self.criterion_vgg = VGGLoss_mask()
        
        # Manual optimization is required for GANs in Lightning
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # Retrieve optimizers and schedulers
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
    
        real = batch["image_1"]
        ref = batch["image_0"]
        neg = batch["neg_image"]
        
        # ===================================================================
        #  Forward Generator
        # ===================================================================
        f_0, id_0 = self.gen.app_encode(ref)
        f_1, id_1 = self.gen.app_encode(real)
        _, id_2 = self.gen.app_encode(neg)
        
        t_0 = self.gen.mot_encode(ref)
        t_1 = self.gen.mot_encode(real)
        
        ta_10 = self.gen.id_adapt(t_1, id_0)
        ta_11 = self.gen.id_adapt(t_1, id_1)
        ta_12 = self.gen.id_adapt(t_1, id_2)
        ta_00 = self.gen.id_adapt(t_0, id_0)
        
        ma_10 = self.gen.mot_decode(ta_10)
        ma_00 = self.gen.mot_decode(ta_00)
        
        pred = self.gen.decode(ma_10, ma_00, f_0)

        # ===================================================================
        #  Step 1: Train Discriminator
        # ===================================================================
        opt_d.zero_grad()
        
        fake_detached = pred.detach()

        pred_real = self.disc(real)
        pred_fake = self.disc(fake_detached)
        
        loss_d = self.calculate_gan_loss(pred_real, pred_fake, is_generator=False)

        self.manual_backward(loss_d)
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=1.0)
        opt_d.step()
        sch_d.step()

        # ===================================================================
        #  Step 2: Train Generator
        # ===================================================================
        opt_g.zero_grad()
        
        # Reconstruction Loss (L1, VGG)
        l1_loss = F.l1_loss(pred, real)
        vgg_loss_all, vgg_loss_face = self.criterion_vgg(
            pred, real, batch["mask_eye_1"] + batch["mask_mouth_1"]
        )

        # Canonical/Distance Loss
        dist1 = torch.norm(t_1 - ta_11, dim=1)
        dist2 = torch.norm(t_1 - ta_12, dim=1)
        dist_loss = torch.abs(dist1 - dist2).mean()
        
        # Base Generator Loss
        total_g_loss = (self.args.loss_l1 * l1_loss + 
                        self.args.loss_vgg_all * vgg_loss_all + 
                        self.args.loss_vgg_face * vgg_loss_face +
                        self.args.loss_dist * dist_loss)
        
        # GAN Loss for Generator
        for p in self.disc.parameters():
            p.requires_grad = False
        
        pred_fake_for_g = self.disc(pred)
        loss_g_gan = self.calculate_gan_loss(None, pred_fake_for_g, is_generator=True)
        total_g_loss += self.args.gan_weight * loss_g_gan
        
        # Unfreeze D
        for p in self.disc.parameters():
            p.requires_grad = True
        
        self.manual_backward(total_g_loss)
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0)
        opt_g.step()
        sch_g.step()
    
        # ===================================================================
        #  Logging
        # ===================================================================
        log_dict = {
            'train/g_total_loss': total_g_loss,
            'train/l1_loss': l1_loss,
            'train/vgg_loss_all': vgg_loss_all,
            'train/vgg_loss_face': vgg_loss_face,
            'train/dist_loss': dist_loss,
            'train/g_gan_loss': loss_g_gan,
            'train/d_loss': loss_d,
        }

        self.log_dict(log_dict, prog_bar=True)
        
        return total_g_loss
    
    def validation_step(self, batch, batch_idx):
        pred = self.gen(batch["image_1"], batch["image_0"])
        recon = self.gen(batch["image_0"], batch["image_0"])            

        pred_l1 = F.l1_loss(pred, batch["image_1"])
        pred_vgg_all, pred_vgg_face = self.criterion_vgg(
            pred, batch["image_1"], batch["mask_eye_1"] + batch["mask_mouth_1"]
        )
        recon_l1 = F.l1_loss(recon, batch["image_0"])
        recon_vgg_all, recon_vgg_face = self.criterion_vgg(
            recon, batch["image_0"], batch["mask_eye_0"] + batch["mask_mouth_0"]
        )

        loss_pred = (self.hparams.loss_l1 * pred_l1 + 
                     self.hparams.loss_vgg_all * pred_vgg_all + 
                     self.hparams.loss_vgg_face * pred_vgg_face)
        loss_recon = (self.hparams.loss_l1 * recon_l1 + 
                      self.hparams.loss_vgg_all * recon_vgg_all + 
                      self.hparams.loss_vgg_face * recon_vgg_face)
        val_loss = (loss_pred + loss_recon) / 2

        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/pred_loss", loss_pred, sync_dist=True)
        self.log("val/recon_loss", loss_recon, sync_dist=True)

        if self.trainer.global_rank == 0 and batch_idx == 0:
            name_list = ['input_0', 'input_1', 'pred', 'recon', 'mask_0', 'mask_1']
            img_list = [batch["image_0"], batch["image_1"], pred, recon, \
                        (batch["mask_eye_0"] + batch["mask_mouth_0"]), 
                        (batch["mask_eye_1"] + batch["mask_mouth_1"])]
            for name, img in zip(name_list, img_list):
                self.logger.experiment.add_images(
                    tag=name,
                    img_tensor=img.clamp(0, 1), 
                    global_step=self.global_step
                )

        return val_loss

    def calculate_gan_loss(self, pred_real, pred_fake, is_generator):
        if isinstance(pred_fake, list):
            if is_generator:
                return sum([F.softplus(-p).mean() for p in pred_fake])
            real_loss = sum([F.softplus(-r).mean() for r in pred_real])
            fake_loss = sum([F.softplus(f).mean() for f in pred_fake])
            return real_loss + fake_loss
        else:
            if is_generator:
                return F.softplus(-pred_fake).mean()
            real_loss = F.softplus(-pred_real).mean()
            fake_loss = F.softplus(pred_fake).mean()
            return real_loss + fake_loss
            
    def configure_optimizers(self):
        g_params = filter(lambda p: p.requires_grad, self.gen.parameters())
        
        # Generator Optimizer
        opt_g = optim.Adam(g_params, lr=self.args.lr, betas=(0.5, 0.999))
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            opt_g, T_max=self.args.iter, eta_min=1e-4
        )
        
        # Discriminator Optimizer
        opt_d = optim.Adam(self.disc.parameters(),
                           lr=self.args.lr * self.args.gan_weight, 
                           betas=(0.5, 0.999))
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=self.args.iter, eta_min=1e-4
        )
        
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        self._safe_load(self.gen, state_dict, prefix="gen.")
        
        # Always load discriminator
        if hasattr(self, "disc"):
            self._safe_load(self.disc, state_dict, prefix="disc.")

    def _safe_load(self, model, state_dict, prefix):
        my_state_dict = model.state_dict()
        safe_dict = {}
        
        ckpt_dict_noprefix = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        
        if not ckpt_dict_noprefix:
            ckpt_dict_noprefix = state_dict

        for k, v in ckpt_dict_noprefix.items():
            if k in my_state_dict:
                if v.shape == my_state_dict[k].shape:
                    safe_dict[k] = v
                else:
                    print(f"[WARN] Shape mismatch for {k}: ckpt {v.shape} vs model {my_state_dict[k].shape}")
        
        msg = model.load_state_dict(safe_dict, strict=False)
        print(f"[INFO] Loaded {prefix.strip('.')} weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def setup(self, stage=None):
        self.train_dataset = TFDataset(root_dir=self.args.dataset_path, split='train')
        self.val_dataset = TFDataset(root_dir=self.args.dataset_path, split='val')
        
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=4,
            num_workers=4,
            shuffle=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root")
    
    # Basic Training Params
    parser.add_argument("--iter", type=int, default=7000000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--exp_path", type=str, default='./exps')
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # GAN Params
    parser.add_argument("--gan_weight", type=float, default=1.0)
    parser.add_argument("--loss_l1", type=float, default=1.0)
    parser.add_argument("--loss_vgg_all", type=float, default=10.0)
    parser.add_argument("--loss_vgg_face", type=float, default=100.0)
    parser.add_argument("--loss_dist", type=float, default=1.0)

    # Model Architecture Params
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--swin_res_threshold', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--low_res_depth', type=int, default=2)

    args = parser.parse_args()

    # Init System and DataModule
    system = IMFSystem(args)
    dm = DataModule(args)
    
    # Load Checkpoint
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        system.load_ckpt(args.resume_ckpt)

    # Logger
    logger = TensorBoardLogger(save_dir=args.exp_path, name=args.exp_name)
    
    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.exp_path, args.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=args.save_freq,
        save_top_k=-1,
        save_last=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=args.iter,
        check_val_every_n_epoch=None,
        val_check_interval=args.display_freq,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )
    
    trainer.fit(system, dm)



