import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import blocks

class ModernTransUNetV2(nn.Module):
    def __init__(self, config, device, padding="same"):
        super().__init__()
        self.transformer_params = config["transformer_params"]
        self.parallel_settings = config["parallel_settings"]
        self.interpolation_settings = config["interpolation_settings"]
        self.leakyReLU = config["leakyReLU"]
        self.instanceNorm = config["instanceNorm"]
        self.device = device

        self.encoder = blocks.Encoder(config, self.leakyReLU, False, True)

        # Adjusts the transformer's output channels to be upsampled
        self.inter = nn.Sequential(
            nn.Conv2d(self.transformer_params["hidden_dim"], 512, 3, 1, 'same'),
            nn.InstanceNorm2d(512, affine=True, track_running_stats=True) if self.instanceNorm else nn.BatchNorm2d(512),
            nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
        )

        # Decoder
        self.up1 = blocks.UpBlock(512, 256, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm, up_mode=config["up_mode"])
        self.up2 = blocks.UpBlock(256, 128, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm, up_mode=config["up_mode"])
        self.up3 = blocks.UpBlock(128, 64, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm, up_mode=config["up_mode"])
        self.up4 = blocks.UpBlock(64, 16, skip=False, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm, up_mode=config["up_mode"])

        # 512x512 feature map treatment
        self.end = nn.Sequential(
            #nn.Conv2d(64, 16, kernel_size=3, padding="same"),
            #nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Network used to enhance final output
        if self.interpolation_settings["flag"] and self.interpolation_settings["topper"]:
            self.complement = blocks.Complement(self.interpolation_settings)

    def forward(self, x, medium=None, large=None, original_images=None):

        x, features = self.encoder(x, medium, large)

        x = self.inter(x) # (B, 512, 32, 32)
        # Decoder
        x = self.up1(x, features[2])
        x = self.up2(x, features[1])
        x = self.up3(x, features[0])
        x = self.up4(x)

        x = self.end(x)

        # Resizes the masks back to their original shapes
        if self.interpolation_settings["flag"]:
            masks = []

            # Go through each predicted mask "x" and their original shape at index "i", resizing them to their original dimensions and appending them to a list to return.
            for i in range(len(original_images)):
                input = torch.unsqueeze((x[i]), 0)
                mask = F.interpolate(input, size=original_images[i].squeeze().shape, mode='bilinear', align_corners=False)
                masks.append(mask)
            
            if self.interpolation_settings["topper"]:
                enhanced_masks = []
                for i, mask in enumerate(masks):
                    mask = self.complement(mask.to(self.device), original_images[i].to(self.device))
                    enhanced_masks.append(mask)
                return enhanced_masks
            return masks
        return x
        

class ModernTransUNet(nn.Module):
    def __init__(self, config, padding="same"):
        super().__init__()

        self.transformer_params = config["transformer_params"]
        self.parallel_settings = config["parallel_settings"]
        self.interpolation_settings = config["interpolation_settings"]
        self.leakyReLU = config["leakyReLU"]
        self.instanceNorm = config["instanceNorm"]


        in_channels = 1

        # Takes a 1024x1024 resolution version of the image to hopefully get more image details into the model
        if self.parallel_settings["flag"]:
            # 512x512
            self.branch1 = nn.Sequential(
                nn.Conv2d(1, self.parallel_settings["out_channels"], 1, 1),
                nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            )

            # 1024x1024
            self.branch2 = nn.Sequential(
                nn.Conv2d(1, self.parallel_settings["out_channels"], 2, 2),
                nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            )

            # 2048x2048
            self.branch3 = nn.Sequential(
                nn.Conv2d(1, self.parallel_settings["out_channels"], 4, 4),
                nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            )

            if self.parallel_settings["concatenate"]:
                combined_maps = self.parallel_settings["out_channels"] * 3
                in_channels += self.parallel_settings["out_channels"]
            else:
                combined_maps = self.parallel_settings["out_channels"]
                in_channels = self.parallel_settings["out_channels"]

            # -> (B, in_channels, 512, 512)
            self.trunk = nn.Sequential(
                nn.Conv2d(combined_maps, in_channels, 3, 1, "same"),
                nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True) if self.instanceNorm else nn.BatchNorm2d(in_channels),
                nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            )


        # Encoder
        self.cnn_encoder = blocks.ResNet(in_channels)
        self.embeddings = blocks.Embeddings(1024, self.transformer_params["hidden_dim"])
        self.transformer = blocks.Transformer(self.transformer_params["num_layers"], self.transformer_params["hidden_dim"], self.transformer_params["num_heads"])

        # Adjusts the transformer's output channels to be upsampled
        self.inter = nn.Sequential(
            nn.Conv2d(self.transformer_params["hidden_dim"], 512, 3, 1, 'same'),
            nn.InstanceNorm2d(512, affine=True, track_running_stats=True) if self.instanceNorm else nn.BatchNorm2d(512),
            nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
        )

        #self.quick = blocks.ConvBlock(512, 512, padding=padding)

        # Decoder
        self.up1 = blocks.UpBlock(512, 256, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm)
        self.up2 = blocks.UpBlock(256, 128, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm)
        self.up3 = blocks.UpBlock(128, 64, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm)
        self.up4 = blocks.UpBlock(64, 16, skip=False, padding=padding, use_leakyReLU=self.leakyReLU, use_instanceNorm=self.instanceNorm) # nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # 512x512 feature map treatment
        self.end = nn.Sequential(
            #nn.Conv2d(64, 16, kernel_size=3, padding="same"),
            #nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Network used to enhance final output
        if self.interpolation_settings["flag"] and self.interpolation_settings["topper"]:
            self.complement = nn.Sequential(

            )

    def forward(self, x, medium=None, large=None, shapes=None, original_images=None):
        B, _, H, W = x.size()

    
        if self.parallel_settings["flag"]:
            x = self.branch1(x)
            medium = self.branch2(medium)
            large = self.branch3(large)

            if self.parallel_settings["concatenate"]:
                x = torch.cat((medium, x), dim=1)
                x = torch.cat((large, x), dim=1)
            else:
                x = x + medium + large

            x = self.trunk(x)

        # Encoder
        x, feature_maps = self.cnn_encoder(x) # (B, 1024, H/16, W/16) (B, 1024, n_patch^0.5, n_patch^0.5)
        x = self.embeddings(x) # (B, n_patches, hidden) (B, n_patches, 768)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.permute(0, 2, 1) #(B, 768, 1024)
        x = x.contiguous().view(B, 768, int(H/16), int(W/16))

        x = self.inter(x) # (B, 512, 32, 32)
        # x = self.quick(x)
        # Decoder
        x = self.up1(x, feature_maps['6.0.conv1'])
        x = self.up2(x, feature_maps['5.0.conv1'])
        x = self.up3(x, feature_maps['0'])
        x = self.up4(x)

        x = self.end(x)

        # Resizes the masks back to their original shapes
        if self.interpolation_settings["flag"]:
            masks = []

            # Go through each predicted mask "x" and their original shape at index "i", resizing them to their original dimensions and appending them to a list to return.
            for i in range(len(shapes)):
                input = torch.unsqueeze((x[i]), 0)
                mask = F.interpolate(input, size=shapes[i], mode='bilinear', align_corners=False)
                masks.append(mask)
            
            if self.interpolation_settings["topper"]:
                enhanced_masks = []
                for mask in masks:
                    mask = self.complement(mask)
                    enhanced_masks.append(mask)
                return enhanced_masks
            return masks
        
        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, embed_in=1024, embed_out=768, num_tlayers=12, hidden_size=768, n_head=12, padding="same"):
        super().__init__()
        #(B, 1, 512, 512)
        self.cnn = blocks.ResNet(in_channels)

        self.embeddings = blocks.Embeddings(embed_in, embed_out)
        self.transformer = blocks.Transformer(num_tlayers, hidden_size, n_head)
        self.inter = nn.Sequential(
            nn.Conv2d(embed_out, 512, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.quick = blocks.ConvBlock(512, 512, padding=padding)

        self.up1 = blocks.UpBlock(512, 256, padding=padding)
        self.up2 = blocks.UpBlock(256, 128, padding=padding)
        self.up3 = blocks.UpBlock(128, 64, padding=padding)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)


        self.end = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        B, _, H, W = x.size()
        x, feature_maps = self.cnn(x) # (B, 1024, H/16, W/16) (B, 1024, n_patch^0.5, n_patch^0.5)
        x = self.embeddings(x) # (B, n_patches, hidden) (B, n_patches, 768)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1) #(B, 768, 1024)
        x = x.contiguous().view(B, 768, int(H/16), int(W/16))

        x = self.inter(x) # (B, 512, 32, 32)
        x = self.quick(x)
        
        x = self.up1(x, feature_maps['6.0.conv1'])
        x = self.up2(x, feature_maps['5.0.conv1'])
        x = self.up3(x, feature_maps['0'])
        x = self.up4(x)

        x = self.end(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.down1 = blocks.DownBlock(1, 64, padding=padding)
        self.down2 = blocks.DownBlock(64, 128, padding=padding)
        self.down3 = blocks.DownBlock(128, 256, padding=padding)
        self.down4 = blocks.DownBlock(256, 512, padding=padding)

        self.inter = blocks.ConvBlock(512, 1024, padding=padding)

        self.up1 = blocks.UpBlock(1024, 512, padding=padding)
        self.up2 = blocks.UpBlock(512, 256, padding=padding)
        self.up3 = blocks.UpBlock(256, 128, padding=padding)
        self.up4 = blocks.UpBlock(128, 64, padding=padding)

        self.end = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, map1 = self.down1(x)
        x, map2 = self.down2(x)
        x, map3 = self.down3(x)
        x, map4 = self.down4(x)
        
        x = self.inter(x)
        
        x = self.up1(x, map4)
        x = self.up2(x, map3)
        x = self.up3(x, map2)
        x = self.up4(x, map1)
        x = self.end(x)


        return x