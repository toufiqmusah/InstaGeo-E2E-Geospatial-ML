import os
import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

from transformers import AutoModel, AutoImageProcessor

class DINOv3VITBlock(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-sat493m", freeze_backbone=True):
        super(DINOv3VITBlock, self).__init__()

        self.dinov3_model = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.dinov3_model.parameters():
                param.requires_grad = False

        self.embed_dim = self.dinov3_model.config.hidden_size
        self.patch_size = self.dinov3_model.config.patch_size
        self.num_layers = self.dinov3_model.config.num_hidden_layers

        self.input_proj = nn.Conv2d(18, 3, kernel_size=1, bias=False)

        self.feature_projections = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 64, kernel_size=1),
            nn.Conv2d(self.embed_dim, 128, kernel_size=1),
            nn.Conv2d(self.embed_dim, 256, kernel_size=1),
            nn.Conv2d(self.embed_dim, 512, kernel_size=1),
            nn.Conv2d(self.embed_dim, 512, kernel_size=1),
        ])

    def extract_hierarchical_features(self, x):
        """
        Extracting hierarchical features from different DINOv3 transformer layers
        """
        # Handle different input dimensions
        original_shape = x.shape
        
        if x.dim() == 5:  # (B, C, T, H, W) - temporal dimension
            B, C, T, H, W = x.shape
            # Reshape to (B*T, C, H, W) for processing each timestep
            x = x.view(B * T, C, H, W)
            process_batch_size = B * T
        elif x.dim() == 4:  # (B, C, H, W) - standard format
            B, C, H, W = x.shape
            process_batch_size = B
        elif x.dim() == 3:  # (C, H, W) - missing batch dimension
            C, H, W = x.shape
            x = x.unsqueeze(0)  # Add batch dimension
            B = 1
            process_batch_size = B
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 3D, 4D, or 5D tensor.")

        # Ensure we have 18 channels total
        if x.shape[1] != 18:
            raise ValueError(f"Expected 18 channels, got {x.shape[1]} channels. Input shape: {original_shape}")

        x_proj = self.input_proj(x)

        target_h = (H // self.patch_size) * self.patch_size
        target_w = (W // self.patch_size) * self.patch_size

        if H != target_h or W != target_w:
            x_proj = F.interpolate(x_proj, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

        with torch.set_grad_enabled(self.dinov3_model.training):
            outputs = self.dinov3_model(x_proj, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        layer_indices = [
            max(1, self.num_layers // 6),
            self.num_layers // 3,
            2 * self.num_layers // 3,
            self.num_layers - 2,
            self.num_layers - 1
        ]

        hierarchical_features = []

        for i, layer_idx in enumerate(layer_indices):
            patch_features = hidden_states[layer_idx][:, 5:]

            num_patches_h = target_h // self.patch_size
            num_patches_w = target_w // self.patch_size

            spatial_features = patch_features.reshape(
                process_batch_size, num_patches_h, num_patches_w, self.embed_dim
            ).permute(0, 3, 1, 2)

            if spatial_features.shape[2:] != (H, W):
                spatial_features = F.interpolate(
                    spatial_features, size=(H, W),
                    mode='bilinear', align_corners=False
                )

            # If we had 5D input, reshape back to include temporal dimension
            if original_shape.__len__() == 5:
                # Reshape from (B*T, C, H, W) back to (B, C, H, W) by averaging over time
                B_orig = original_shape[0]
                T_orig = original_shape[2]
                spatial_features = spatial_features.view(B_orig, T_orig, -1, H, W).mean(dim=1)

            projected_features = self.feature_projections[i](spatial_features)
            hierarchical_features.append(projected_features)

        return hierarchical_features

    def forward(self, x):
        return self.extract_hierarchical_features(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=True, norm_type='group'):
        super(EncoderBlock, self).__init__()

        self.mednext_block = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res and (in_channels == out_channels),
            norm_type=norm_type,
            dim='2d',
            grn=False
        )
        self.down_block = MedNeXtDownBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            dim='2d',
            grn=False
        )

    def forward(self, x, dinov3_features=None):
        x = self.mednext_block(x)

        if dinov3_features is not None:
            if x.shape[2:] != dinov3_features.shape[2:]:
                dinov3_features = F.interpolate(
                    dinov3_features,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            x = x + dinov3_features

        skip_features = x.clone()
        x = self.down_block(x)
        return x, skip_features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None,
                 exp_r=4, kernel_size=7, norm_type='group'):
        super(DecoderBlock, self).__init__()
        self.skip_channels = skip_channels

        # 2D upsampling block
        self.up_block = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim='2d',
            grn=False
        )

        if skip_channels is not None:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.ReLU(inplace=True)
            )

            self.combine_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_conv = None
            self.combine_conv = None

    def forward(self, x, skip_features=None, use_skip=True):

        x = self.up_block(x)

        if use_skip and skip_features is not None and self.skip_channels is not None:
            skip = self.skip_conv(skip_features)

            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:],
                                   mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.combine_conv(x)

        return x


class BiomassNet(nn.Module):

    """
    DINOv3-based network for Above-Ground Biomass estimation
    Input: 18 × 256 × 256 (3 timesteps × 6 bands × spatial)
    Output: 1 × 256 × 256 (AGB values)
    """

    def __init__(self, in_channels=18, out_channels=1, use_skip_connections=True,
                 dinov3_model="facebook/dinov3-vitl16-pretrain-sat493m", freeze_dinov3=True):

        super(BiomassNet, self).__init__()

        self.use_skip_connections = use_skip_connections
        self.dinov3_backbone = DINOv3VITBlock(dinov3_model, freeze_dinov3)
        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.encoder1 = EncoderBlock(64, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = nn.Sequential(
            MedNeXtBlock(512, 512, exp_r=4, kernel_size=7, do_res=True,
                        norm_type='group', dim='2d', grn=False),
            MedNeXtBlock(512, 512, exp_r=4, kernel_size=7, do_res=True,
                        norm_type='group', dim='2d', grn=False)
        )

        skip_channels = [64, 128, 256, 512] if use_skip_connections else [None]*4
        self.decoder4 = DecoderBlock(512, 256, skip_channels[3])
        self.decoder3 = DecoderBlock(256, 128, skip_channels[2])
        self.decoder2 = DecoderBlock(128, 64, skip_channels[1])
        self.decoder1 = DecoderBlock(64, 64, skip_channels[0])

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def set_skip_connections(self, use_skip):
        self.use_skip_connections = use_skip

    def forward(self, x):
        # Handle temporal dimension if present
        original_shape = x.shape
        
        if x.dim() == 5:  # (B, C, T, H, W) - temporal dimension
            B, C, T, H, W = x.shape
            # Flatten temporal dimension into channels: (B, C*T, H, W)
            x = x.view(B, C * T, H, W)
        
        dinov3_features = self.dinov3_backbone(x)

        x = self.input_conv(x)
        skip_connections = []

        x, skip1 = self.encoder1(x, dinov3_features[0])
        skip_connections.append(skip1)

        x, skip2 = self.encoder2(x, dinov3_features[1])
        skip_connections.append(skip2)

        x, skip3 = self.encoder3(x, dinov3_features[2])
        skip_connections.append(skip3)

        x, skip4 = self.encoder4(x, dinov3_features[3])
        skip_connections.append(skip4)

        if dinov3_features[4] is not None:
            bottleneck_features = dinov3_features[4]
            if x.shape[2:] != bottleneck_features.shape[2:]:
                bottleneck_features = F.interpolate(
                    bottleneck_features, size=x.shape[2:],
                    mode='bilinear', align_corners=False
                )
            x = x + bottleneck_features

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        x = self.decoder4(x, skip_connections[0], self.use_skip_connections)
        x = self.decoder3(x, skip_connections[1], self.use_skip_connections)
        x = self.decoder2(x, skip_connections[2], self.use_skip_connections)
        x = self.decoder1(x, skip_connections[3], self.use_skip_connections)

        biomass_map = self.output_conv(x)

        return biomass_map

'''# Test the model
if __name__ == "__main__":
    model = BiomassNet(
        in_channels=18,
        out_channels=1,
        dinov3_model="facebook/dinov3-vitl16-pretrain-sat493m",
        freeze_dinov3=True
    )

    # Test with different input shapes
    print("Testing different input shapes:")
    
    # 4D input (standard) - what your model expects
    print("\n1. Testing 4D input (B, C, H, W):")
    x = torch.randn(4, 18, 128, 128)
    try:
        biomass_pred = model(x)
        print(f"Success! Input: {x.shape}, Output: {biomass_pred.shape}")
    except Exception as e:
        print(f"Error with 4D input: {e}")
    
    # 5D input (with temporal dimension) - if your dataloader provides this
    print("\n2. Testing 5D input (B, C, T, H, W):")
    x = torch.randn(4, 6, 3, 128, 128)  # 6 bands, 3 timesteps
    try:
        biomass_pred = model(x)
        print(f"Success! Input: {x.shape}, Output: {biomass_pred.shape}")
    except Exception as e:
        print(f"Error with 5D input: {e}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''