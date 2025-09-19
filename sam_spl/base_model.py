import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_spl.base_layer import DenseBlock, SELayer, VGGBlock, Res_CBAM_block
from sam_spl.UpBlock_layer import UpBlock_attention
from sam_spl.hieradet import Hiera
from sam_spl.pmt_generator import MultiScaleBlock, MultiScalePositionalEncoder

from sam_spl.transformer import TwoWayTransformer
from sam_spl.utils import LayerNorm2d, MLP
from sam_spl.image_encoder import ImageEncoder


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    try:
        if classname.find("Conv") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("Linear") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except AttributeError:
        pass

class DynamicConvBlock(nn.Module):
    def __init__(self, skip_channel, n):
        super().__init__()
        self.skip_channel = skip_channel
        if n > 4:
            raise ValueError(f"n must be <= 4, but got {n}")
        self.n = n
        self.conv_blocks = self._build_blocks()
    
    def _build_blocks(self):
        layers = [
            nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.skip_channel),
            nn.GELU()
        ]

        num_extra_blocks = 4 - self.n
        
        for _ in range(num_extra_blocks):
            layers.extend([
                nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=2, stride=2),
                nn.GELU()
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv_blocks(x)
    
def build_dynamic_conv(skip_channel, n):
    if n > 4:
        raise ValueError(f"n must be <= 4, but got {n}")

    layers = [
        nn.Conv2d(skip_channel, skip_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(skip_channel),
        nn.GELU()
    ]

    for _ in range(4 - n):
        layers.extend([
            nn.Conv2d(skip_channel, skip_channel, kernel_size=2, stride=2),
            nn.GELU()
        ])
    
    return nn.Sequential(*layers)

class SamAdaptor(nn.Module):
    def __init__(
        self,
        sam_encoder: nn.Module,
        decoder_transformer: nn.Module,
        backbone_channel_list: list[int] = [384, 192, 96],
        stages=[1, 2, 7],
        block="res",
        dense_low_channels: list[int] = [96, 48, 24],
        num_mask_tokens=1,
        use_sam_decoder=True,
        mode=4,
        pe_inch=[24, 48, 96],
    ):
        super().__init__()
        self.use_sam_decoder = use_sam_decoder
        self.num_mask_tokens = num_mask_tokens
        self.dense_low_channels = backbone_channel_list + dense_low_channels[1:]
        self.pe_inch = pe_inch
        if block == "res":
            _block = Res_CBAM_block
        elif block == "vgg":
            _block = VGGBlock
        elif block == "dense":
            _block = DenseBlock

        _block = self._select_block(block)
        self.image_encoder = ImageEncoder(
            sam_encoder,
            _block=_block,
            backbone_channel_list=backbone_channel_list,
            stages=stages,
        )
        
        self.skip_channel_gen = dense_low_channels
        self.mask_channel_gen = [ch // 2 for ch in dense_low_channels]
        if self.use_sam_decoder:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs()
        else:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs2()

        self.reduction_convs = self._initialize_reduction_convs()

        # Projection block to match the dimensions of the dense low channels
        if self.use_sam_decoder:
            self.decoder_dim = decoder_transformer.embedding_dim
            self.decoder_transformer = decoder_transformer
            self.mask_token = nn.Embedding(1, self.decoder_dim)
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(self.decoder_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(self.decoder_dim // 4, dense_low_channels[0], kernel_size=2, stride=2),
                nn.GELU(),
            )
            self.output_hypernetworks_mlp = MLP(self.decoder_dim, self.decoder_dim, dense_low_channels[0], 3)
            # self.image_pe_encoder = MultiScalePositionalEncoder(
            #     in_chans=dense_low_channels[::-1],
            #     down_times=[len(self.dense_low_channels) - i - 1 for i in range(len(dense_low_channels))],
            # )
            self.image_pe_encoder = MultiScalePositionalEncoder(
                in_chans=pe_inch,
                down_times=[len(self.dense_low_channels) - i - 1 for i in range(len(pe_inch))],
            )
            # self.proj_block = nn.Sequential(
            #     nn.Conv2d(self.skip_channel_gen[0], self.skip_channel_gen[0], kernel_size=1, stride=1),
            #     nn.BatchNorm2d(self.skip_channel_gen[0]),
            #     nn.GELU(),
            #     nn.Conv2d(self.skip_channel_gen[0], self.skip_channel_gen[0], kernel_size=2, stride=2),
            #     nn.GELU(),
            # )
            self.proj_block = build_dynamic_conv(self.skip_channel_gen[0], len(stages))

            self.deep_conv_block = nn.Sequential(
                nn.Conv2d(backbone_channel_list[0], self.decoder_dim, kernel_size=1, stride=1),
                LayerNorm2d(self.decoder_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1, stride=1),
                nn.GELU(),
            )
        else:
            self.proj_block = nn.Sequential(
                nn.Conv2d(self.dense_low_channels[0], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.dense_low_channels[1]),
                nn.GELU(),
                nn.Conv2d(self.dense_low_channels[1], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.GELU(),
            )

        
        self.apply(weights_init_kaiming)

    def _select_block(self, block: str) -> nn.Module:
        """Select the appropriate block type based on the provided string."""
        blocks = {"res": Res_CBAM_block, "vgg": VGGBlock, "dense": DenseBlock}
        return blocks.get(block, Res_CBAM_block)  # Default to Res_CBAM_block if not found

    def _initialize_up_decoders_and_skip_convs(self) -> tuple:
        """Initialize up decoders and skip connections."""

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.skip_channel_gen:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )

        return up_decoders, skip_convs

    def _initialize_up_decoders_and_skip_convs2(self) -> tuple:
        """Initialize up decoders and skip connections."""

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.dense_low_channels[1:]:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )
        return up_decoders, skip_convs

    def _initialize_reduction_convs(self) -> nn.ModuleList:
        """Initialize mask convolution layers."""
        reducttion_conv = nn.ModuleList()
        for ch in self.mask_channel_gen:
            reducttion_conv.append(nn.Conv2d(ch, 1, kernel_size=1, stride=1))
        return reducttion_conv

    def _sam_load_state_dict(model, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "sam_mask_decoder.transformer" in key:
                new_key = key.replace("sam_mask_decoder.transformer", "decoder_f")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=False)

    def _load_sam_checkpoint(self, ckpt_path):
        """Load the parameters of sam2"""
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            # sd = {k.replace("sam_mask_decoder.transformer", "decoder_transformer"): v for k, v in sd.items()}
            unexpected_keys, missing_keys = self.load_state_dict(sd, strict=False)
        print("Finish loading sam2 checkpoint")

    def _freeze_encoder(self):
        """Freeze the image_encoder of sam2"""
        for name, para in self.named_parameters():
            if "image_encoder.trunk" in name and "promote_genertor" not in name:
                para.requires_grad_(False)
            if "image_encoder.neck" in name and "promote_genertor" not in name:
                para.requires_grad_(True)
            elif "decoder_transformer" in name:
                para.requires_grad_(True)

    def print_param_quantity(self):
        """Calculate model parameters for training."""
        trunk_param = sum(p.numel() for p in self.image_encoder.trunk.parameters()) / 1_000_000
        pmtg_param = sum(p.numel() for p in self.image_encoder.trunk.promote_genertor.parameters()) / 1_000_000
        all_param = sum(p.numel() for p in self.parameters()) / 1_000_000
        print(f"The parameter number of the model is {all_param - trunk_param + pmtg_param:.2f}M")

    def _process_deep_features(self, features: dict) -> list:
        """Process deep features through convolution and upsampling."""
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]
        try:
            image_embeddings = sam_feature[-1]
        except IndexError:
            image_embeddings = dense_features[-1]

        pe_input = dense_features + sam_feature
        pe_input = pe_input[:len(self.pe_inch)]
        image_pe = self.image_pe_encoder(pe_input)

        B, C, W, H = image_embeddings.shape
        src = self.deep_conv_block(image_embeddings)
        token = self.mask_token.weight.unsqueeze(0).expand(B, -1, -1)
        hs, src = self.decoder_transformer(src, image_pe, token)
        src = src.transpose(1, 2).contiguous().view(B, self.decoder_dim, W, H)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in = self.output_hypernetworks_mlp(hs.squeeze(1))
        deep_feat = hyper_in.unsqueeze(-1).unsqueeze(-1) * upscaled_embedding

        deep_feat = self.proj_block(deep_feat)
        for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[::-1], self.skip_convs, self.up_decoders)):
            deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
            masks.append(deep_feat)

        return masks

    def _process_deep_features2(self, features: dict) -> list:
        """Process deep features through convolution and upsampling."""
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]

        dense_features = (dense_features + sam_feature)[::-1]

        deep_feat = self.proj_block(dense_features[0])

        for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[1:], self.skip_convs, self.up_decoders)):
            deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
            masks.append(deep_feat)

        return masks[-len(self.mask_channel_gen):]

    def _generate_masks(self, deep_feats: list, image_size: list[int, int]) -> list:
        """Generate masks from the deep features."""
        masks = []
        for mask_conv, feature_map in zip(self.reduction_convs[::-1], deep_feats[::-1]):
            mask_0 = F.interpolate(feature_map, image_size, mode="bilinear", align_corners=False)
            mask_0 = mask_conv(mask_0)
            masks.append(mask_0)

        return masks

    def forward(self, x: torch.tensor):
        out_image_size = x.shape[-2:]
        features = self.image_encoder(x)
        if self.use_sam_decoder:
            masks = self._process_deep_features(features)
        else:
            masks = self._process_deep_features2(features)

        masks = self._generate_masks(masks, out_image_size)
        return masks


def make_adaptor(
    backbone_channel_list: list[int] = [384, 192, 96],
    dense_low_channels: list[int] = [96, 48, 24],
    stages: list[int] = [1, 2, 7],
    global_att_blocks: list[int] = [5, 7, 9],
    window_pos_embed_bkg_spatial_size: list[int] = [7, 7],
    window_spec: list[int] = [8, 4, 16],
    block: str = "res",
    embed_dim=96,
    use_sam_decoder=True,
    pe_inch=[24, 48, 96],
    sam_ckpt_path=None,
):
    """_summary_

    Args:
        backbone_channel_list (list[int], optional): The list of encoder channels. Defaults to [384, 192, 96].
        out_dim (int, optional): Number of masks in the final output. Defaults to 4.
        down_times (int, optional): Times of feature map dimensionality drop for shallow feature extraction. Defaults to 3.
        stages (list[int], optional): The stages of hieradet. Defaults to [1, 2, 7].
        global_att_blocks (list[int], optional): global attention blocks. Defaults to [5, 7, 9].
        window_pos_embed_bkg_spatial_size (list[int], optional): window size. Defaults to [7, 7].
        window_spec (list[int], optional): window spec. Defaults to [8, 4, 16].
        block (str, optional): The type of block used in encoder. Defaults to "res".

    Returns:
        nn.Module: sam adaptor
    """
    promote_generator = MultiScaleBlock(stages=stages, embed_dim=embed_dim)

    sam_encoder = Hiera(
        promote_genertor=promote_generator,
        embed_dim=embed_dim,
        num_heads=1,
        stages=stages,
        global_att_blocks=global_att_blocks,
        window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
        window_spec=window_spec,
    )

    decoder_transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )

    predictor = SamAdaptor(
        sam_encoder=sam_encoder,
        decoder_transformer=decoder_transformer,
        backbone_channel_list=backbone_channel_list,
        stages=stages,
        block=block,
        dense_low_channels=dense_low_channels,
        use_sam_decoder=use_sam_decoder,
        pe_inch=pe_inch,
    )
    if sam_ckpt_path is not None:
        predictor._load_sam_checkpoint(sam_ckpt_path)
    return predictor
