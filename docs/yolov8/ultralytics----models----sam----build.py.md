# `.\yolov8\ultralytics\models\sam\build.py`

```py
# 导入 functools 模块中的 partial 函数
from functools import partial

# 导入 torch 库
import torch

# 导入下载函数 attempt_download_asset
from ultralytics.utils.downloads import attempt_download_asset

# 导入模块中的类和函数
from .modules.decoders import MaskDecoder
from .modules.encoders import ImageEncoderViT, PromptEncoder
from .modules.sam import Sam
from .modules.tiny_encoder import TinyViT
from .modules.transformer import TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    """构建并返回一个 SAM h-size 模型。"""
    return _build_sam(
        encoder_embed_dim=1280,  # 编码器嵌入维度
        encoder_depth=32,  # 编码器深度
        encoder_num_heads=16,  # 编码器头数
        encoder_global_attn_indexes=[7, 15, 23, 31],  # 全局注意力索引
        checkpoint=checkpoint,  # 检查点
    )


def build_sam_vit_l(checkpoint=None):
    """构建并返回一个 SAM l-size 模型。"""
    return _build_sam(
        encoder_embed_dim=1024,  # 编码器嵌入维度
        encoder_depth=24,  # 编码器深度
        encoder_num_heads=16,  # 编码器头数
        encoder_global_attn_indexes=[5, 11, 17, 23],  # 全局注意力索引
        checkpoint=checkpoint,  # 检查点
    )


def build_sam_vit_b(checkpoint=None):
    """构建并返回一个 SAM b-size 模型。"""
    return _build_sam(
        encoder_embed_dim=768,  # 编码器嵌入维度
        encoder_depth=12,  # 编码器深度
        encoder_num_heads=12,  # 编码器头数
        encoder_global_attn_indexes=[2, 5, 8, 11],  # 全局注意力索引
        checkpoint=checkpoint,  # 检查点
    )


def build_mobile_sam(checkpoint=None):
    """构建并返回 Mobile-SAM 模型。"""
    return _build_sam(
        encoder_embed_dim=[64, 128, 160, 320],  # 编码器嵌入维度列表
        encoder_depth=[2, 2, 6, 2],  # 编码器深度列表
        encoder_num_heads=[2, 4, 5, 10],  # 编码器头数列表
        encoder_global_attn_indexes=None,  # 全局注意力索引
        mobile_sam=True,  # 是否是移动 SAM
        checkpoint=checkpoint,  # 检查点
    )


def _build_sam(
    encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, checkpoint=None, mobile_sam=False
):
    """构建选定的 SAM 模型架构。"""
    prompt_embed_dim = 256  # 提示嵌入维度
    image_size = 1024  # 图像尺寸
    vit_patch_size = 16  # ViT 补丁大小
    image_embedding_size = image_size // vit_patch_size  # 图像嵌入大小
    # 创建图像编码器对象，根据条件选择不同的实现方式：TinyViT 或 ImageEncoderViT
    image_encoder = (
        TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=encoder_embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
        if mobile_sam  # 如果 mobile_sam 变量为真，则使用 TinyViT
        else ImageEncoderViT(  # 否则使用 ImageEncoderViT
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    )
    
    # 创建 SAM 模型对象，包括图像编码器、提示编码器和蒙版解码器
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],  # 像素均值用于数据标准化
        pixel_std=[58.395, 57.12, 57.375],  # 像素标准差用于数据标准化
    )
    
    # 如果提供了检查点文件路径，则加载模型状态字典
    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)  # 尝试下载检查点文件
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)  # 加载检查点的状态字典
        sam.load_state_dict(state_dict)  # 将加载的状态字典应用到 SAM 模型
    
    sam.eval()  # 设置 SAM 模型为评估模式
    
    # 返回配置好的 SAM 模型对象
    return sam
# SAM 模型映射，将模型文件名映射到对应的构建函数
sam_model_map = {
    "sam_h.pt": build_sam_vit_h,    # 如果文件名以 "sam_h.pt" 结尾，则使用 build_sam_vit_h 构建函数
    "sam_l.pt": build_sam_vit_l,    # 如果文件名以 "sam_l.pt" 结尾，则使用 build_sam_vit_l 构建函数
    "sam_b.pt": build_sam_vit_b,    # 如果文件名以 "sam_b.pt" 结尾，则使用 build_sam_vit_b 构建函数
    "mobile_sam.pt": build_mobile_sam,  # 如果文件名为 "mobile_sam.pt"，则使用 build_mobile_sam 构建函数
}

# 构建 SAM 模型的函数，根据给定的检查点（ckpt）选择合适的构建函数
def build_sam(ckpt="sam_b.pt"):
    """Build a SAM model specified by ckpt."""
    model_builder = None
    ckpt = str(ckpt)  # 将检查点转换为字符串类型，以支持路径检查点类型

    # 遍历 SAM 模型映射中的每个键（模型文件名）
    for k in sam_model_map.keys():
        # 如果给定的检查点（ckpt）以当前模型文件名（k）结尾，则选择对应的构建函数
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)

    # 如果未找到匹配的模型构建函数，则抛出文件未找到异常
    if not model_builder:
        raise FileNotFoundError(f"{ckpt} is not a supported SAM model. Available models are: \n {sam_model_map.keys()}")

    # 使用选定的模型构建函数构建模型，并返回结果
    return model_builder(ckpt)
```