from collections import OrderedDict
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_dim=3, stride_dim=1, padding=0, use_instanceNorm=False, use_leakyReLU=False):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_dim, stride_dim, padding),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=True) if use_instanceNorm else nn.BatchNorm2d(out_ch),
            nn.LeakyReLU() if use_leakyReLU else nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_dim, stride_dim, padding),
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=True) if use_instanceNorm else nn.BatchNorm2d(out_ch),
            nn.LeakyReLU() if use_leakyReLU else nn.ReLU(),
        )

    def forward(self, x):
        x = self.down(x)
        return x
    

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_dim=3, stride_dim=1, padding=0):
        super().__init__()

        self.conv = ConvBlock(in_ch, out_ch, kernel_dim, stride_dim, padding)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        map = self.conv(x)
        x = self.downsample(map)
        return x, map

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_dim=3, stride_dim=1, skip=True, padding=0, use_instanceNorm=False, use_leakyReLU=False, up_mode=0):
        super().__init__()

        if up_mode == 0:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2) if skip else nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = ConvBlock(in_ch, out_ch, kernel_dim, stride_dim, padding, use_instanceNorm, use_leakyReLU)

    def forward(self, x, map=None):
        x = self.upsample(x)
        if map is not None:
            x = torch.cat((map, x), dim=1)
        x = self.conv(x)

        return x
    
class Embeddings(nn.Module):
    def __init__(self, in_channels=1024, out_channels=768):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1024, out_channels)) # (1, num_patches, hidden)

    def forward(self, x):
        x = self.patch_embeddings(x) # (B, 768, 32, 32)
        x = x.flatten(2) #(B, 768, 1024)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)  (B, 1024,768)
        x = x + self.position_embeddings
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead, mlp_dim):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead, mlp_dim),
            num_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the network to accept 512x512 images and output maps of 32x32 (H/16, W/16)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-3]))
        self.feature_maps = {}

        
    def forward(self, x):
        # Create a list to store hooks
        hooks = []

        # Function to register hooks to store feature maps
        def register_hooks(module_name):
            def hook(module, input, output):
                self.feature_maps[module_name] = output
            return hook
        
        # Register hooks to store feature maps at each module
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(register_hooks(module_name))
                hooks.append(hook)

        # Run the forward prop
        x = self.backbone(x)

        # Remove hooks to avoid interfering with future computations
        for hook in hooks:
            hook.remove()
            
        return x, self.feature_maps

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GatedConv2d, self).__init__()
        self.content_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        content = self.content_conv(x)
        gating = F.sigmoid(self.gating_conv(x))
        return F.tanh(content) * gating

class WNConv2d(nn.Conv2d):
    '''
    Convolutional layer with optional weight normalization.
    '''
    def __init__(self, *args, enable_weight_norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_weight_norm = enable_weight_norm

    def forward(self, x):
        if self.enable_weight_norm:
            w = self.weight
            v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
            w = (w - m) / torch.sqrt(v + 1e-5)
            return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False, enable_weight_norm=True):
    return WNConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups, enable_weight_norm=enable_weight_norm)


def conv1x1(cin, cout, stride=1, bias=False, enable_weight_norm=True):
    return WNConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias, enable_weight_norm=enable_weight_norm)

def get_normalization(channels, normalization, groups=32):
    if normalization == 'group':
        return nn.GroupNorm(groups, channels)
    elif normalization == 'instance':
        return nn.InstanceNorm2d(channels, eps=1e-6, affine=True, track_running_stats=True)
    elif normalization == 'batch':
        return nn.BatchNorm2d(channels)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")

class ResBottleNeck(nn.Module):
    def __init__(self, cin, cmid=None, cout=None, stride=1, bias=False, weight_normalization=True, leakyReLU = False, normalization="batch", group=32):
        super().__init__()

        cout = cout or cin
        cmid = cmid or cout//4

        self.conv1 = conv1x1(cin, cmid, bias=bias, enable_weight_norm=weight_normalization)
        self.norm1 = get_normalization(cmid, normalization, group)

        self.conv2 = conv3x3(cmid, cmid, stride, bias=bias, enable_weight_norm=weight_normalization)
        self.norm2 = get_normalization(cmid, normalization, group)

        self.conv3 = conv1x1(cmid, cout, bias=bias, enable_weight_norm=weight_normalization)
        self.norm3 = get_normalization(cout, normalization, group)

        self.activation = nn.LeakyReLU() if leakyReLU else nn.ReLU()

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=bias, enable_weight_norm=weight_normalization)
            self.n_proj = get_normalization(cout, normalization, cout)
        
    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.n_proj(residual)

        y = self.activation(self.norm1(self.conv1(x)))
        y = self.activation(self.norm2(self.conv2(y)))
        y = self.norm3(self.conv3(y))

        y = self.activation(residual + y)
        return y
    
class CustomResNet(nn.Module):
    def __init__(self, input_ch, resnet_settings, leakyReLU=False, bias=False, weight_normalization=True):
        super().__init__()
        #patch_count = image_dimension**2 / patch_size**2
        #downsample_count = math.log2(image_dimension) - math.log2(math.sqrt(patch_count))

        width = resnet_settings["width"]
        block_units = resnet_settings["blocks"]
        normalization = resnet_settings["normalization"]

        self.root = nn.Sequential(OrderedDict([
            ('conv', WNConv2d(input_ch, width, kernel_size=7, stride=2, bias=bias, padding=3)),
            ('norm', get_normalization(width, normalization)),
            ('activation', nn.LeakyReLU() if leakyReLU else nn.ReLU()),
        ]))

        self.block1 = nn.Sequential(OrderedDict([('unit1', ResBottleNeck(cin=width, cout=width*4, cmid=width, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization))] + 
                                                [(f'unit{i:d}', ResBottleNeck(cin=width*4, cout=width*4, cmid=width, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization)) for i in range(2, block_units[0] + 1)]))

        self.block2 = nn.Sequential(OrderedDict([('unit1', ResBottleNeck(cin=width*4, cout=width*8, cmid=width*2, stride=2, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization))] + 
                                                [(f'unit{i:d}', ResBottleNeck(cin=width*8, cout=width*8, cmid=width*2, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization)) for i in range(2, block_units[1] + 1)]))

        self.block3 = nn.Sequential(OrderedDict([('unit1', ResBottleNeck(cin=width*8, cout=width*16, cmid=width*4, stride=2, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization))] + 
                                                [(f'unit{i:d}', ResBottleNeck(cin=width*16, cout=width*16, cmid=width*4, bias=bias, weight_normalization=weight_normalization, leakyReLU=leakyReLU, normalization=normalization)) for i in range(2, block_units[2] + 1)]))
        
    def forward(self, x):
        features = []
        x = self.root(x)
        features.append(x) # B, width, dim/2, dim/2
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        x = self.block1(x) 
        features.append(x) # B, width, dim/2, dim/2

        x = self.block2(x)
        features.append(x) # B, width, dim/2, dim/2

        x = self.block3(x) # B, width, dim/2, dim/2

        return x, features

class MultiScale(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.parallel_settings = model_config["parallel_settings"]
        self.leakyReLU = model_config["leakyReLU"]

        self.trunk_out = model_config["trunk/res_channels"]
        if self.parallel_settings["concatenate"]:
            self.trunk_in = self.parallel_settings["branch_out_channels"] * 3
        else:
            self.trunk_in = self.parallel_settings["branch_out_channels"]

        branch_blocks = []
        for i in range(self.parallel_settings["branch_blocks"]):
            branch_blocks.append(ResBottleNeck(cin=self.parallel_settings["branch_out_channels"], cout=self.parallel_settings["branch_out_channels"], leakyReLU=self.leakyReLU, normalization=model_config["resnet_settings"]["normalization"], group=8))

        self.branch1 = nn.Sequential(
            nn.Conv2d(1, self.parallel_settings["branch_out_channels"], 1, 1),
            get_normalization(self.parallel_settings["branch_out_channels"], model_config["resnet_settings"]["normalization"], 8),
            nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            *branch_blocks
        )
        # 1024x1024
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, self.parallel_settings["branch_out_channels"], 2, 2),
            get_normalization(self.parallel_settings["branch_out_channels"], model_config["resnet_settings"]["normalization"], 8),
            nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            *branch_blocks
        )
        # 2048x2048
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, self.parallel_settings["branch_out_channels"], 4, 4),
            get_normalization(self.parallel_settings["branch_out_channels"], model_config["resnet_settings"]["normalization"], 8),
            nn.LeakyReLU() if self.leakyReLU else nn.ReLU(),
            *branch_blocks
        )

        trunk_blocks = []
        if self.parallel_settings["trunk_blocks"] == 0:
            trunk_blocks.append(nn.Conv2d(self.trunk_in, self.trunk_out, 3, 1, "same"))
            trunk_blocks.append(get_normalization(self.trunk_out, model_config["resnet_settings"]["normalization"]))
            trunk_blocks.append(nn.LeakyReLU() if self.leakyReLU else nn.ReLU())
        else:
            trunk_blocks.append(ResBottleNeck(cin=self.trunk_in, cout=self.trunk_out, leakyReLU=self.leakyReLU, normalization=model_config["resnet_settings"]["normalization"], group=8))
            for i in range(self.parallel_settings["trunk_blocks"] - 1):
                trunk_blocks.append(ResBottleNeck(cin=self.trunk_out, cout=self.trunk_out, leakyReLU=self.leakyReLU, normalization=model_config["resnet_settings"]["normalization"], group=8))
        
        # -> (B, in_channels, 512, 512)
        self.trunk = nn.Sequential(
            *trunk_blocks
        )
    
    def forward(self, x, medium, large):
        small = self.branch1(x)
        medium = self.branch2(medium)
        large = self.branch3(large)

        if self.parallel_settings["concatenate"]:
            x = torch.cat((medium, small), dim=1)
            x = torch.cat((large, x), dim=1)
        else:
            x = small + medium + large

        x = self.trunk(x)

        x = x + small + medium + large

        return x



class Encoder(nn.Module):
    def __init__(self, model_config, leakyReLU=False, bias=False, weight_normalization=True):
        super().__init__()

        self.parallel_settings = model_config["parallel_settings"]
        self.resnet_settings = model_config["resnet_settings"]
        self.transformer_settings = model_config["transformer_params"]

        res_in_ch = 1
        if self.parallel_settings["flag"]:
            self.parallel = MultiScale(model_config)
            res_in_ch = model_config["trunk/res_channels"]


        self.resnet = CustomResNet(res_in_ch , self.resnet_settings, leakyReLU, bias, weight_normalization)
        self.embeddings = Embeddings(self.resnet_settings["width"] * 16, self.transformer_settings["hidden_dim"])
        self.transformer = Transformer(self.transformer_settings["num_layers"], self.transformer_settings["hidden_dim"], self.transformer_settings["num_heads"], self.transformer_settings['mlp_dim'])

        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            get_normalization(128, self.resnet_settings["normalization"], 32),
            nn.LeakyReLU() if leakyReLU else nn.ReLU(),
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            get_normalization(256, self.resnet_settings["normalization"], 32),
            nn.LeakyReLU() if leakyReLU else nn.ReLU(),
        )

    def forward(self, x, medium=None, large=None):
        B, _, H, W = x.size()
        if hasattr(self, 'parallel'):
            x = self.parallel(x, medium, large)
        x, features = self.resnet(x)
        x = self.embeddings(x) # (B, n_patches, hidden) (B, n_patches, 768)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.permute(0, 2, 1) #(B, 768, 1024)
        x = x.contiguous().view(B, self.transformer_settings["hidden_dim"], int(H/16), int(W/16))

        features[1] = self.feature1(features[1])
        features[2] = self.feature2(features[2])

        return x, features
    
class ComplementBlock(nn.Module):
    def __init__(self, interpolation_settings):
        super().__init__()
        self.settings = interpolation_settings
        root_out = self.settings["root_out"]

        self.conv = nn.Conv2d(root_out, root_out, 3, 1, padding=1)
        self.normalization = get_normalization(root_out, self.settings["normalization"], self.settings["groups"])
        self.activation = nn.LeakyReLU() if self.settings["leakyReLU"] else nn.ReLU()

    def forward(self, x):
        y = self.activation(self.normalization(self.conv(x)))
        y += x
        return y
    
class Complement(nn.Module):
    def __init__(self, interpolation_settings):
        super().__init__()
        self.settings = interpolation_settings
        input_ch = 2 if self.settings['mode'] == 'concatenation' else 1
        root_out = self.settings["root_out"]

        blocks = []
        for i in range(self.settings["blocks"]):
            blocks.append(ComplementBlock(self.settings))

        self.root = nn.Sequential(
            nn.Conv2d(input_ch, root_out, 1, 1),
            #nn.GroupNorm(8, root_out),
            get_normalization(root_out, self.settings["normalization"], self.settings["groups"]),
            nn.LeakyReLU() if self.settings["leakyReLU"] else nn.ReLU(),
        )

        self.middle = nn.Sequential(
            *blocks
        )

        self.end = nn.Sequential(
            nn.Conv2d(root_out, 1, 3, 1, padding=1),
            #nn.Sigmoid()
        )
        
    def forward(self, x, original_image):
        if self.settings['mode'] == 'concatenation':
            x = torch.cat((x , original_image), dim=1)
        else:
            x += original_image
        x = self.root(x)
        x = self.middle(x)
        x = self.end(x)

        return x