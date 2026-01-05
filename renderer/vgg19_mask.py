import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramid for computing pyramid perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)

        return out_dict

class Vgg19(nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_model = models.vgg19(pretrained=True)
        vgg_pretrained_features = vgg_model.features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                 requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = X.clamp(-1, 1)
        X = X / 2 + 0.5
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out

class VGGLoss_mask(nn.Module):
    def __init__(self, device="cuda"):
        super(VGGLoss_mask, self).__init__()
        self.device = device
        self.scales = [1.0, 0.5, 0.25, 0.125] 
        
        self.pyramid = ImagePyramide(self.scales, 3).to(self.device)
        self.vgg = Vgg19().to(self.device)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0) 

    def forward(self, img_recon, img_real, mask):
        pyramid_real = self.pyramid(img_real)
        pyramid_recon = self.pyramid(img_recon)

        loss_all = 0.0
        loss_face = 0.0
        
        for scale in self.scales:
            scale_str = str(scale)
            scale_key = f'prediction_{scale_str}'

            if scale == 1.0:
                mask_at_scale = mask
            else:
                mask_at_scale = F.interpolate(mask, scale_factor=scale, mode='nearest', recompute_scale_factor=True)

            recon_feats = self.vgg(pyramid_recon[scale_key])
            real_feats = self.vgg(pyramid_real[scale_key])

            for i, weight in enumerate(self.weights):
                feat_real = real_feats[i].detach()
                feat_recon = recon_feats[i]

                # Global loss
                all_loss_i = torch.abs(feat_recon - feat_real).mean()
                loss_all += all_loss_i * weight
                
                # Face masked loss
                mask_i = F.interpolate(mask_at_scale, size=feat_real.shape[2:], mode='nearest')
                
                diff_mask = torch.abs((feat_recon - feat_real) * mask_i)
                
                mask_sum = mask_i.sum()
                if mask_sum > 1e-6:
                    mask_loss_i = diff_mask.sum() / mask_sum
                    loss_face += mask_loss_i * weight
        
        return loss_all, loss_face
