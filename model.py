import torch
import torch.nn as nn
import math
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock
from einops import rearrange
from typing import Sequence, Tuple, Union
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, dim), nn.Tanh())

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class D_SE(nn.Module):
    def __init__(self, h, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.h = h
        self.dim = dim * h
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, self.h),
                                nn.Softmax(dim=-1))

    def forward(self, x):
        x = x.view(x.shape[0], self.h, self.h, -1)  # (b, h, w, d*c)
        # a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(x)
        a = a.unsqueeze(-1)
        x = x.view(x.shape[0], self.h, self.h, self.h, -1)
        x = a * x
        x = x.view(x.shape[0], -1, x.shape[-1])
        return x


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ TNT Block
    """
    def __init__(self,
                 outer_dim,
                 inner_dim,
                 outer_num_heads,
                 inner_num_heads,
                 num_words,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = Attention(inner_dim,
                                        inner_dim,
                                        num_heads=inner_num_heads,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        proj_drop=drop)
            self.inner_norm2 = norm_layer(inner_dim)
            self.inner_mlp = Mlp(in_features=inner_dim,
                                 hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim,
                                 act_layer=act_layer,
                                 drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
            self.inner_mask_token = nn.Parameter(torch.zeros(1, 1, inner_dim, requires_grad=False))
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(outer_dim,
                                    outer_dim,
                                    num_heads=outer_num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim,
                             hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim,
                             act_layer=act_layer,
                             drop=drop)
        # SE
        self.se = se
        self.outer_se = 0
        # self.se_layer = None
        if self.se > 0:
            self.inner_se_layer = D_SE(4, inner_dim, 0.25)

    def restore_feat(self, inner, ids_restore):
        L = inner.shape[1]
        mean_inner = torch.mean(inner.detach(), dim=1, keepdim=True)
        inner_mask_token = mean_inner.repeat(1, ids_restore.shape[1] - L, 1)
        # inner_mask_token = self.inner_mask_token.repeat(inner.shape[0], ids_restore.shape[1] - L, 1)
        inner = torch.cat([inner, inner_mask_token], dim=1)  # no cls token
        inner = torch.gather(inner, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, inner.shape[2]))  # unshuffle

        return inner

    def forward(self, inner_tokens, outer_tokens, inner_ids_restore, inner_ids_keep):
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))  # B*N, k*k, c
            # inner_tokens = self.inner_mlp(self.inner_norm2(inner_tokens))
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))  # B*N, k*k, c
            B, N, _ = outer_tokens.size()
            if inner_ids_restore is not None:
                inner_tokens = self.restore_feat(inner_tokens, inner_ids_restore)
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, N, -1))))  # B, N, C
            if inner_ids_restore is not None:
                inner_tokens = torch.gather(inner_tokens, dim=1, index=inner_ids_keep.unsqueeze(-1).repeat(1, 1, inner_tokens.shape[-1]))
        if self.outer_se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.outer_se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """
    def __init__(self, img_size=96, patch_size=16, in_chans=1, outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        num_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride) * math.ceil(patch_size[2] / inner_stride)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))
        from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
        to_chars = f"(b {' '.join([c[0] for c in chars])}) c {' '.join([c[1] for c in chars])}"
        axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
        self.patch_embeddings = nn.Sequential(Rearrange(f"{from_chars} -> {to_chars}", **axes_len),
                                              nn.Conv3d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride))
        # self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)
        # self.proj = nn.Conv3d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, _, H, W, D = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.unfold(x) # B, Ck2, N (b, 3*16*16, 14*14)
        # x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size) # B*N, C, 16, 16 (b*6*6*6, 1, 16, 16)
        # x = self.proj(x) # B*N, C, 8, 8
        x = self.patch_embeddings(x)
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)  # B*N, 4*4*4, C
        return x


class UnetrPrUpBlock_(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.conv11 = get_conv_layer(
            spatial_dims,
            48,
            768,
            kernel_size=1,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        get_conv_layer(
                            spatial_dims,
                            out_channels,
                            out_channels,
                            kernel_size=upsample_kernel_size,
                            stride=upsample_stride,
                            conv_only=True,
                            is_transposed=True,
                        ),
                        UnetResBlock(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            norm_name=norm_name,
                        ),
                    ) for i in range(num_layer)
                ])
            else:
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        get_conv_layer(
                            spatial_dims,
                            out_channels,
                            out_channels,
                            kernel_size=upsample_kernel_size,
                            stride=upsample_stride,
                            conv_only=True,
                            is_transposed=True,
                        ),
                        UnetBasicBlock(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            norm_name=norm_name,
                        ),
                    ) for i in range(num_layer)
                ])
        else:
            self.blocks = nn.ModuleList([
                get_conv_layer(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=upsample_kernel_size,
                    stride=upsample_stride,
                    conv_only=True,
                    is_transposed=True,
                ) for i in range(num_layer)
            ])

    def forward(self, inner, outer):
        ratio = inner.shape[-1] // outer.shape[-1]
        inner = self.conv11(inner)
        outer = outer[:, :, :, None, :, None, :, None].repeat(1, 1, 1, ratio, 1, ratio, 1, ratio).view(*inner.shape)
        x = inner + outer

        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrBasicBlock_(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        self.conv11 = get_conv_layer(
            spatial_dims,
            48,
            768,
            kernel_size=1,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )
        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inner, outer):
        ratio = inner.shape[-1] // outer.shape[-1]
        inner = self.conv11(inner)
        outer = outer[:, :, :, None, :, None, :, None].repeat(1, 1, 1, ratio, 1, ratio, 1, ratio).view(*inner.shape)
        x = inner + outer
        return self.layer(x)


class TNT_MIM(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """
    def __init__(self,
                 img_size=96,
                 patch_size=16,
                 in_chans=1,
                 out_channels=14,
                 outer_dim=768,
                 inner_dim=48,
                 depth=12,
                 outer_num_heads=12,
                 inner_num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 inner_stride=4,
                 se=0,
                 feature_size=16,
                 norm_name='instance',
                 conv_block=True,
                 res_block=True,
                 spatial_dims=3):
        super().__init__()
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_stride = inner_stride
        # self.use_contour = use_contour
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      outer_dim=outer_dim,
                                      inner_dim=inner_dim,
                                      inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        # masked_num_words = int(num_words*(1-inner_mask_ratio))
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        # self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(
                    Block(outer_dim=outer_dim,
                          inner_dim=-1,
                          outer_num_heads=outer_num_heads,
                          inner_num_heads=inner_num_heads,
                          num_words=num_words,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          se=se))
            else:
                blocks.append(
                    Block(outer_dim=outer_dim,
                          inner_dim=inner_dim,
                          outer_num_heads=outer_num_heads,
                          inner_num_heads=inner_num_heads,
                          num_words=num_words,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(outer_dim, representation_size)
        #self.repr_act = nn.Tanh()

        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)

        self.outer_mask_token1 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token2 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token3 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token4 = nn.Parameter(torch.zeros(1, 1, outer_dim))

        self.inner_mask_token2 = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.inner_mask_token3 = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.inner_mask_token4 = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self.outer_mask_token_for_inner = nn.Parameter(torch.zeros(1, 1, num_words, inner_dim, requires_grad=False))

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock_(
            spatial_dims=spatial_dims,
            in_channels=outer_dim,
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock_(
            spatial_dims=3,
            in_channels=outer_dim,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock_(
            spatial_dims=3,
            in_channels=outer_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=outer_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.contour_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()
    def random_masking(self, outer_tokens, mask_ratio, inner_tokens=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = outer_tokens.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=outer_tokens.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        outer_tokens_masked = torch.gather(outer_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=outer_tokens.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        if inner_tokens is not None:
            _, P, C = inner_tokens.shape  #
            inner_tokens = rearrange(inner_tokens, '(b l) p c -> b l p c', l=L)
            inner_tokens_masked = torch.gather(inner_tokens, dim=1, index=ids_keep[:, :, None, None].repeat(1, 1, P, C))
            inner_tokens_masked = inner_tokens_masked.view(-1, P, C)
            return inner_tokens_masked, outer_tokens_masked, ids_keep, ids_restore

        return outer_tokens_masked, ids_keep, ids_restore

    def forward_features(self, x, outer_mask_ratio, inner_mask_ratio):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 4*4*4, C, N表示patch个数，4表示patch大小

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        # outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        inner_ids_restore = None
        outer_ids_restore = None
        inner_ids_keep = None
        if outer_mask_ratio > 0:
            inner_tokens, outer_tokens, _, outer_ids_restore = self.random_masking(outer_tokens, outer_mask_ratio, inner_tokens)
        if inner_mask_ratio > 0:
            inner_tokens, inner_ids_keep, inner_ids_restore = self.random_masking(inner_tokens, inner_mask_ratio)

        inner_tokens_list = []
        outer_tokens_list = []
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, inner_ids_restore, inner_ids_keep)

            inner_tokens_list.append(inner_tokens)
            outer_tokens_list.append(outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens, inner_tokens_list, outer_tokens_list, inner_ids_restore, outer_ids_restore

    def proj_inner_feat(self, inner_tokens):
        n = self.img_size // self.patch_size
        p = self.patch_size // self.inner_stride
        n_len = {f"n{i+1}": n for i, n in enumerate(ensure_tuple_rep(n, 3))}
        p_len = {f"p{i+1}": p for i, p in enumerate(ensure_tuple_rep(p, 3))}
        inner_feat = rearrange(inner_tokens, '(b n1 n2 n3) (p1 p2 p3) c -> b c (n1 p1) (n2 p2) (n3 p3)', **n_len, **p_len)  # (b, c, 24, 24, 24)
        return inner_feat.contiguous()

    def proj_outer_feat(self, outer_tokens):
        n = self.img_size // self.patch_size
        n_len = {f"n{i+1}": n for i, n in enumerate(ensure_tuple_rep(n, 3))}
        outer_feat = rearrange(outer_tokens, 'b (n1 n2 n3) c -> b c n1 n2 n3', **n_len)
        return outer_feat.contiguous()

    def restore_feat(self, outer, ids_restore, outer_mask_token, inner=None):
        L = outer.shape[1]
        outer_mask_token = outer_mask_token.repeat(outer.shape[0], ids_restore.shape[1] - L, 1)
        outer = torch.cat([outer, outer_mask_token], dim=1)  # no cls token
        outer = torch.gather(outer, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, outer.shape[2]))  # unshuffle

        if inner is not None:
            _, P, C = inner.shape
            inner = rearrange(inner, '(b l) p c -> b l p c', l=L)
            outer_mask_token_for_inner = self.outer_mask_token_for_inner.repeat(outer.shape[0], ids_restore.shape[1] - L, 1, 1)
            inner = torch.cat([inner, outer_mask_token_for_inner], dim=1)  # no cls token
            inner = torch.gather(inner, dim=1, index=ids_restore[:, :, None, None].repeat(1, 1, P, C))  # unshuffle
            inner = inner.view(-1, P, C)

            return inner, outer

        return outer

    def single_forward(self, x_in, outer_mask_ratio=0., inner_mask_ratio=0., use_contour=False):
        x, inner_feat_list, outer_feat_list, inner_ids_restore, outer_ids_restore = self.forward_features(x_in, outer_mask_ratio, inner_mask_ratio)

        if outer_mask_ratio > 0:
            x = self.restore_feat(x, outer_ids_restore, self.outer_mask_token1)
        enc1 = self.encoder1(x_in)

        inner2, outer2 = inner_feat_list[3], outer_feat_list[3]
        if inner_mask_ratio > 0:
            inner2 = self.restore_feat(inner2, inner_ids_restore, self.inner_mask_token2)
        if outer_mask_ratio > 0:
            inner2, outer2 = self.restore_feat(outer2, outer_ids_restore, self.outer_mask_token2, inner2)
        enc2 = self.encoder2(self.proj_inner_feat(inner2), self.proj_outer_feat(outer2))  # 48

        inner3, outer3 = inner_feat_list[6], outer_feat_list[6]
        if inner_mask_ratio > 0:
            inner3 = self.restore_feat(inner3, inner_ids_restore, self.inner_mask_token3)
        if outer_mask_ratio > 0:
            inner3, outer3 = self.restore_feat(outer3, outer_ids_restore, self.outer_mask_token3, inner3)
        enc3 = self.encoder3(self.proj_inner_feat(inner3), self.proj_outer_feat(outer3))

        inner4, outer4 = inner_feat_list[9], outer_feat_list[9]
        if inner_mask_ratio > 0:
            inner4 = self.restore_feat(inner4, inner_ids_restore, self.inner_mask_token4)
        if outer_mask_ratio > 0:
            inner4, outer4 = self.restore_feat(outer4, outer_ids_restore, self.outer_mask_token4, inner4)
        enc4 = self.encoder4(self.proj_inner_feat(inner4), self.proj_outer_feat(outer4))

        dec4 = self.proj_outer_feat(x)

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        if use_contour:
            return self.contour_out(out)
        return self.out(out)

    def forward(self, x1, x2=None, outer_mask_ratio=0., inner_mask_ratio=0., use_contour=False):
        if x2 is None:
            return self.single_forward(x1)
        else:
            output1 = self.single_forward(x1, outer_mask_ratio, inner_mask_ratio, use_contour=use_contour)
            output2 = self.single_forward(x2)
            return output1, output2


class TNT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """
    def __init__(self,
                 img_size=96,
                 patch_size=16,
                 in_chans=1,
                 out_channels=14,
                 outer_dim=768,
                 inner_dim=48,
                 depth=12,
                 outer_num_heads=12,
                 inner_num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 inner_stride=4,
                 se=0,
                 feature_size=16,
                 norm_name='instance',
                 conv_block=True,
                 res_block=True,
                 spatial_dims=3):
        super().__init__()
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_stride = inner_stride
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      outer_dim=outer_dim,
                                      inner_dim=inner_dim,
                                      inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        # masked_num_words = int(num_words*(1-inner_mask_ratio))
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        # self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(
                    Block(outer_dim=outer_dim,
                          inner_dim=-1,
                          outer_num_heads=outer_num_heads,
                          inner_num_heads=inner_num_heads,
                          num_words=num_words,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          se=se))
            else:
                blocks.append(
                    Block(outer_dim=outer_dim,
                          inner_dim=inner_dim,
                          outer_num_heads=outer_num_heads,
                          inner_num_heads=inner_num_heads,
                          num_words=num_words,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(outer_dim, representation_size)
        #self.repr_act = nn.Tanh()

        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)

        self.outer_mask_token1 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token2 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token3 = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_mask_token4 = nn.Parameter(torch.zeros(1, 1, outer_dim))

        self.inner_mask_token2 = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.inner_mask_token3 = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.inner_mask_token4 = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self.outer_mask_token_for_inner = nn.Parameter(torch.zeros(1, 1, num_words, inner_dim, requires_grad=False))

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock_(
            spatial_dims=spatial_dims,
            in_channels=outer_dim,
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock_(
            spatial_dims=3,
            in_channels=outer_dim,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock_(
            spatial_dims=3,
            in_channels=outer_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=outer_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()
    def random_masking(self, outer_tokens, mask_ratio, inner_tokens=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = outer_tokens.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=outer_tokens.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        outer_tokens_masked = torch.gather(outer_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=outer_tokens.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        if inner_tokens is not None:
            _, P, C = inner_tokens.shape  #
            inner_tokens = rearrange(inner_tokens, '(b l) p c -> b l p c', l=L)
            inner_tokens_masked = torch.gather(inner_tokens, dim=1, index=ids_keep[:, :, None, None].repeat(1, 1, P, C))
            inner_tokens_masked = inner_tokens_masked.view(-1, P, C)
            return inner_tokens_masked, outer_tokens_masked, ids_keep, ids_restore

        return outer_tokens_masked, ids_keep, ids_restore

    def forward_features(self, x, outer_mask_ratio, inner_mask_ratio):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 4*4*4, C, N表示patch个数，4表示patch大小

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        # outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        inner_ids_restore = None
        outer_ids_restore = None
        inner_ids_keep = None
        if outer_mask_ratio > 0:
            inner_tokens, outer_tokens, _, outer_ids_restore = self.random_masking(outer_tokens, outer_mask_ratio, inner_tokens)
        if inner_mask_ratio > 0:
            inner_tokens, inner_ids_keep, inner_ids_restore = self.random_masking(inner_tokens, inner_mask_ratio)

        inner_tokens_list = []
        outer_tokens_list = []
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, inner_ids_restore, inner_ids_keep)

            inner_tokens_list.append(inner_tokens)
            outer_tokens_list.append(outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens, inner_tokens_list, outer_tokens_list, inner_ids_restore, outer_ids_restore

    def proj_inner_feat(self, inner_tokens):
        n = self.img_size // self.patch_size
        p = self.patch_size // self.inner_stride
        n_len = {f"n{i+1}": n for i, n in enumerate(ensure_tuple_rep(n, 3))}
        p_len = {f"p{i+1}": p for i, p in enumerate(ensure_tuple_rep(p, 3))}
        inner_feat = rearrange(inner_tokens, '(b n1 n2 n3) (p1 p2 p3) c -> b c (n1 p1) (n2 p2) (n3 p3)', **n_len, **p_len)  # (b, c, 24, 24, 24)
        return inner_feat.contiguous()

    def proj_outer_feat(self, outer_tokens):
        n = self.img_size // self.patch_size
        n_len = {f"n{i+1}": n for i, n in enumerate(ensure_tuple_rep(n, 3))}
        outer_feat = rearrange(outer_tokens, 'b (n1 n2 n3) c -> b c n1 n2 n3', **n_len)
        return outer_feat.contiguous()

    def restore_feat(self, outer, ids_restore, outer_mask_token, inner=None):
        L = outer.shape[1]
        outer_mask_token = outer_mask_token.repeat(outer.shape[0], ids_restore.shape[1] - L, 1)
        outer = torch.cat([outer, outer_mask_token], dim=1)  # no cls token
        outer = torch.gather(outer, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, outer.shape[2]))  # unshuffle

        if inner is not None:
            _, P, C = inner.shape
            inner = rearrange(inner, '(b l) p c -> b l p c', l=L)
            outer_mask_token_for_inner = self.outer_mask_token_for_inner.repeat(outer.shape[0], ids_restore.shape[1] - L, 1, 1)
            inner = torch.cat([inner, outer_mask_token_for_inner], dim=1)  # no cls token
            inner = torch.gather(inner, dim=1, index=ids_restore[:, :, None, None].repeat(1, 1, P, C))  # unshuffle
            inner = inner.view(-1, P, C)

            return inner, outer

        return outer

    def forward(self, x_in, outer_mask_ratio=0., inner_mask_ratio=0., ifcontractive=None):
        x, inner_feat_list, outer_feat_list, inner_ids_restore, outer_ids_restore = self.forward_features(x_in, outer_mask_ratio, inner_mask_ratio)

        if outer_mask_ratio > 0:
            x = self.restore_feat(x, outer_ids_restore, self.outer_mask_token1)
        enc1 = self.encoder1(x_in)

        inner2, outer2 = inner_feat_list[3], outer_feat_list[3]
        if inner_mask_ratio > 0:
            inner2 = self.restore_feat(inner2, inner_ids_restore, self.inner_mask_token2)
        if outer_mask_ratio > 0:
            inner2, outer2 = self.restore_feat(outer2, outer_ids_restore, self.outer_mask_token2, inner2)
        enc2 = self.encoder2(self.proj_inner_feat(inner2), self.proj_outer_feat(outer2))  # 48

        inner3, outer3 = inner_feat_list[6], outer_feat_list[6]
        if inner_mask_ratio > 0:
            inner3 = self.restore_feat(inner3, inner_ids_restore, self.inner_mask_token3)
        if outer_mask_ratio > 0:
            inner3, outer3 = self.restore_feat(outer3, outer_ids_restore, self.outer_mask_token3, inner3)
        enc3 = self.encoder3(self.proj_inner_feat(inner3), self.proj_outer_feat(outer3))

        inner4, outer4 = inner_feat_list[9], outer_feat_list[9]
        if inner_mask_ratio > 0:
            inner4 = self.restore_feat(inner4, inner_ids_restore, self.inner_mask_token4)
        if outer_mask_ratio > 0:
            inner4, outer4 = self.restore_feat(outer4, outer_ids_restore, self.outer_mask_token4, inner4)
        enc4 = self.encoder4(self.proj_inner_feat(inner4), self.proj_outer_feat(outer4))

        dec4 = self.proj_outer_feat(x)

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        if ifcontractive:
            return dec1, self.out(out)

        return self.out(out)


def get_model(args,
              img_size=96,
              patch_size=16,
              in_chans=1,
              out_channels=14,
              outer_dim=768,
              inner_dim=48,
              depth=12,
              outer_num_heads=12,
              inner_num_heads=4,
              mlp_ratio=4.,
              qkv_bias=False,
              qk_scale=None,
              drop_rate=0.,
              attn_drop_rate=0.,
              drop_path_rate=0.,
              norm_layer=nn.LayerNorm,
              inner_stride=4,
              se=0,
              feature_size=16,
              norm_name='instance',
              conv_block=True,
              res_block=True,
              spatial_dims=3):
    if args.MIM is True:
        model = TNT_MIM
    else:
        model = TNT
    net = model(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_channels=out_channels,
        outer_dim=outer_dim,
        inner_dim=inner_dim,
        depth=depth,
        outer_num_heads=outer_num_heads,
        inner_num_heads=inner_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        inner_stride=inner_stride,
        se=se,
        feature_size=feature_size,
        norm_name=norm_name,
        conv_block=conv_block,
        res_block=res_block,
        spatial_dims=spatial_dims,
    )
    return net


if __name__ == "__main__":
    input = torch.randn((2, 1, 96, 96, 96))
    model = TNT(se=1)
    output = model(input, 0.5, 0.5)
    print(output.shape)
