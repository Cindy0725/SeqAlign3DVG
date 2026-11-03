
import torch
import torch.nn as nn



dependencies = ["torch"]


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = True,
    **kwargs,
):
    from dinov2.models import vision_transformer as vits

    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused", pretrained=pretrained, **kwargs)


def _make_dinov2_linear_head(
    *,
    model_name: str = "dinov2_vitl14",
    embed_dim: int = 768,
    layers: int = 4,
    pretrained: bool = False,
    **kwargs,
):
    assert layers in (1, 4), f"Unsupported number of layers: {layers}"
    # linear_head = nn.Conv2d((layers) * embed_dim * 2, 255, kernel_size = 1) # output shape: (batch_size, 256, image_height, image_width)

    linear_head = nn.Sequential(
            nn.Conv2d((layers) * embed_dim * 2, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1))

    if pretrained:
        layers_str = str(layers) if layers == 4 else ""
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_linear{layers_str}_head.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        linear_head.load_state_dict(state_dict, strict=False)

    return linear_head



class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)



class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class _LinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module, layers: int = 1, train: bool = True):
        super().__init__()
        self.backbone = backbone
        # todo: try to freeze the ViT backbone. How about the small version? If we want to spare more images / larger 3D grid resolution for rendering loss. 
        # self.linear_head = linear_head
        self.layers = layers
        # self.image_height = 392
        # self.image_width = 1260

        # self.image_height = 280
        # self.image_width = 980

        # self.image_height = 448
        # self.image_width = 784

        self.image_height = 364
        self.image_width = 560
        
        # self.image_height = 238
        # self.image_width = 322
        # self.image_height = 224
        # self.image_width = 280
        # self.image_height = 532
        # self.image_width = 728
        self.patch_size = 14
        self.features = [96, 192, 384, 768]
        # self.project = nn.Sequential(nn.Linear(768*2, 768), nn.GELU()) # base-v
        # self.project = nn.Sequential(nn.Linear(1024*2, 768), nn.GELU()) # large-v
        self.project = nn.Sequential(nn.Linear(768, 768), nn.GELU()) # small-v

        self.reassemble0 = nn.Sequential(
        nn.Unflatten(2, torch.Size([self.image_height // 14, self.image_width // 14])),
        nn.Conv2d(
            in_channels= 768,
            out_channels=self.features[0],
            kernel_size=1,
            stride=1,
            padding=0,
            ),
        nn.ConvTranspose2d(
            in_channels=self.features[0],
            out_channels=self.features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
            ),
        nn.Conv2d(self.features[0], 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.linear_project = nn.Sequential(
            nn.Conv2d(
            in_channels= 768,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            ),
        )
        
        self.reassemble1 = nn.Sequential(
        nn.Unflatten(2, torch.Size([self.image_height // 14, self.image_width // 14])),
        nn.Conv2d(
            in_channels= 768,
            out_channels=self.features[1],
            kernel_size=1,
            stride=1,
            padding=0,
            ),
        nn.ConvTranspose2d(
            in_channels=self.features[1],
            out_channels=self.features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
            ),
        nn.Conv2d(self.features[1], 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.reassemble2 = nn.Sequential(
        nn.Unflatten(2, torch.Size([self.image_height // 14, self.image_width // 14])),
        nn.Conv2d(
            in_channels= 768,
            out_channels=self.features[2],
            kernel_size=1,
            stride=1,
            padding=0,
            ),
        nn.Conv2d(self.features[2], 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )
        
        self.reassemble3 = nn.Sequential(
        nn.Unflatten(2, torch.Size([self.image_height // 14, self.image_width // 14])),
        nn.Conv2d(
            in_channels= 768,
            out_channels=self.features[3],
            kernel_size=1,
            stride=1,
            padding=0,
            ),
        nn.Conv2d(
            in_channels=self.features[3],
            out_channels=self.features[3],
            kernel_size=3,
            stride=2,
            padding=1,
            ),
        nn.Conv2d(self.features[3], 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.fusion0 = _make_fusion_block(256, False)
        self.fusion1 = _make_fusion_block(256, False)
        self.fusion2 = _make_fusion_block(256, False)
        self.fusion3 = _make_fusion_block(256, False)



        self.head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode = 'bilinear', align_corners=True),
            # Interpolate(scale_factor=2, mode = 'nearest', align_corners=None),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.Identity()
            nn.Softplus()
        )
        self.linear_proj = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)





        
    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            cls_tokens = cls_token.unsqueeze(1).repeat(1, patch_tokens.shape[1], 1)
            linear_input = torch.cat([cls_tokens, patch_tokens], dim=2) # shape: (batch_size, num_tokens, C)

            batch_size, num_tokens, channel = linear_input.shape
            linear_input_ = torch.transpose(linear_input,1,2) # shape: (batch_size, C, num_tokens)
            linear_input_ = linear_input_.view(batch_size, channel, self.image_height // self.patch_size, self.image_width // self.patch_size)
            linear_input_ = nn.functional.interpolate(linear_input_, size = [4 * self.image_height // self.patch_size, 4 * self.image_width // self.patch_size], mode = 'bilinear') 
            reassemble0 = self.linear_proj(linear_input_)
            
            # batch_size, num_tokens, channel = linear_input.shape
            linear_input = self.project(linear_input)
            linear_input = torch.transpose(linear_input,1,2) # shape: (batch_size, C, num_tokens)
            linear_input = self.reassemble0(linear_input)
            # linear_input = linear_input.view(batch_size, channel, self.image_height // self.patch_size, self.image_width // self.patch_size)
            path_1 = nn.functional.interpolate(linear_input, size = [4 * self.image_height // self.patch_size, 4 * self.image_width // self.patch_size], mode = 'bilinear') 
            out = self.head(path_1)
            # reassemble0 = path_1


            # scale_4 = self.scale_4(linear_input)
            # scale_2 = self.scale_2(linear_input)
            # scale_1 = self.scale_1(linear_input)
            # scale_05 = self.scale_05(linear_input)

        # elif self.layers == 4:

        #     layers_n = 4
        #     x = self.backbone.get_intermediate_layers(x, n=layers_n, return_class_token=True)
        #     # fmt: off
        #     patch_tokens0 = x[0][0] # 4, 1976, 768 x[0][1] 4, 768
        #     cls_tokens0 = x[0][1].unsqueeze(1).repeat(1, patch_tokens0.shape[1], 1) # 4, 1976, 768
        #     read_project0 = self.project(torch.cat([cls_tokens0, patch_tokens0], dim=2)) # 4, 1976, 768 s: 20 1040 768
        #     read_project0 = torch.transpose(read_project0, 1, 2) # 4, 768, 1976 s: 20 1040 768
        #     reassemble0 = self.reassemble0(read_project0)

        #     patch_tokens1 = x[1][0]
        #     cls_tokens1 = x[1][1].unsqueeze(1).repeat(1, patch_tokens1.shape[1], 1) 
        #     read_project1 = self.project(torch.cat([cls_tokens1, patch_tokens1], dim=2))
        #     read_project1 = torch.transpose(read_project1, 1, 2)
        #     reassemble1 = self.reassemble1(read_project1)

            
        #     patch_tokens2 = x[2][0]
        #     cls_tokens2 = x[2][1].unsqueeze(1).repeat(1, patch_tokens2.shape[1], 1) 
        #     read_project2 = self.project(torch.cat([cls_tokens2, patch_tokens2], dim=2))
        #     read_project2 = torch.transpose(read_project2, 1, 2)
        #     reassemble2 = self.reassemble2(read_project2)

        #     patch_tokens3 = x[3][0]
        #     cls_tokens3 = x[3][1].unsqueeze(1).repeat(1, patch_tokens3.shape[1], 1) 
        #     read_project3 = self.project(torch.cat([cls_tokens3, patch_tokens3], dim=2))
        #     read_project3 = torch.transpose(read_project3, 1, 2)
        #     reassemble3 = self.reassemble3(read_project3)

        #     paths = []
        #     path_4 = self.fusion3(reassemble3)
 
        #     path_3 = self.fusion2(path_4, reassemble2)

        #     path_2 = self.fusion1(path_3, reassemble1)
         
        #     path_1 = self.fusion0(path_2, reassemble0)

        #     paths.append(path_1)
        #     paths.append(path_2)
        #     paths.append(path_3)
        #     paths.append(path_4)

        #     out = self.head(path_1)
        elif self.layers == 4:
            # baseline_x = self.backbone.forward_features(x)
            # cls_token = baseline_x["x_norm_clstoken"]
            # patch_tokens = baseline_x["x_norm_patchtokens"]
            # cls_tokens = cls_token.unsqueeze(1).repeat(1, patch_tokens.shape[1], 1)
            # linear_input = torch.cat([cls_tokens, patch_tokens], dim=2) # shape: (batch_size, num_tokens, C)

            # batch_size, num_tokens, channel = linear_input.shape
            # linear_input_ = torch.transpose(linear_input,1,2) # shape: (batch_size, C, num_tokens)
            # linear_input_ = linear_input_.view(batch_size, channel, self.image_height // self.patch_size, self.image_width // self.patch_size)
            # linear_input_ = nn.functional.interpolate(linear_input_, size = [4 * self.image_height // self.patch_size, 4 * self.image_width // self.patch_size], mode = 'bilinear') 
            # reassemble0_baseline = self.linear_proj(linear_input_)



            layers_n = 4
            x = self.backbone.get_intermediate_layers(x, n=layers_n, return_class_token=True)
            # fmt: off
            patch_tokens0 = x[0][0] # 4, 1976, 768 x[0][1] 4, 768
            cls_tokens0 = x[0][1].unsqueeze(1).repeat(1, patch_tokens0.shape[1], 1) # 4, 1976, 768
            read_project0 = self.project(torch.cat([cls_tokens0, patch_tokens0], dim=2)) # 4, 1976, 768 s: 20 1040 768
            read_project0 = torch.transpose(read_project0, 1, 2) # 4, 768, 1976 s: 20 1040 768
            reassemble0 = self.reassemble0(read_project0)

            patch_tokens1 = x[1][0]
            cls_tokens1 = x[1][1].unsqueeze(1).repeat(1, patch_tokens1.shape[1], 1) 
            read_project1 = self.project(torch.cat([cls_tokens1, patch_tokens1], dim=2))
            read_project1 = torch.transpose(read_project1, 1, 2)
            reassemble1 = self.reassemble1(read_project1)

            
            patch_tokens2 = x[2][0]
            cls_tokens2 = x[2][1].unsqueeze(1).repeat(1, patch_tokens2.shape[1], 1) 
            read_project2 = self.project(torch.cat([cls_tokens2, patch_tokens2], dim=2))
            read_project2 = torch.transpose(read_project2, 1, 2)
            reassemble2 = self.reassemble2(read_project2)

            patch_tokens3 = x[3][0]
            cls_tokens3 = x[3][1].unsqueeze(1).repeat(1, patch_tokens3.shape[1], 1) 
            read_project3 = self.project(torch.cat([cls_tokens3, patch_tokens3], dim=2))
            read_project3 = torch.transpose(read_project3, 1, 2)
            reassemble3 = self.reassemble3(read_project3)

            paths = []
            path_4 = self.fusion3(reassemble3)
 
            path_3 = self.fusion2(path_4, reassemble2)

            path_2 = self.fusion1(path_3, reassemble1)
         
            path_1 = self.fusion0(path_2, reassemble0)

            paths.append(path_1)
            paths.append(path_2)
            paths.append(path_3)
            paths.append(path_4)

            out = self.head(path_1)
            

            # I want to fuse image features reassemble0_baseline with path_1, which is the output of the 4-layer fusion block.
            
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        # return out, self.head_feature(path_1)
        # return out, path_1
        return reassemble0, out, path_1
        # return scale_4, scale_2, scale_1, scale_05

def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_large",
    layers: int = 1,
    pretrained: bool = True,
    train: bool = True,
    **kwargs,
):
    backbone = _make_dinov2_model(arch_name=arch_name, pretrained=pretrained, **kwargs)

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    model_name = _make_dinov2_model_name(arch_name, patch_size)
    linear_head = _make_dinov2_linear_head(
        model_name=model_name, embed_dim=embed_dim, layers=layers, pretrained=False
    )

    return _LinearClassifierWrapper(backbone=backbone, linear_head=linear_head, layers=layers, train=train)


def dinov2_vits14_lc_dpt(*, layers: int = 4, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_small", layers=layers, pretrained=pretrained, **kwargs)


def dinov2_vitb14_lc_dpt(*, pretrained: bool = True, train: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_base", pretrained=pretrained, train=train, **kwargs)


def dinov2_vitl14_lc_dpt(*, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14_lc_dpt(*, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2", ffn_layer="swiglufused", pretrained=pretrained, **kwargs
    )






