# `.\segment-anything\segment_anything\build_sam.py`

```
# 导入 torch 库
import torch

# 导入 partial 函数
from functools import partial

# 从当前目录下的 modeling 模块中导入 ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer 类
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

# 构建 SAM-ViT-H 模型
def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

# 将 build_sam_vit_h 赋值给 build_sam
build_sam = build_sam_vit_h

# 构建 SAM-ViT-L 模型
def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

# 构建 SAM-ViT-B 模型
def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

# SAM 模型注册表，包含默认模型和不同配置的 ViT 模型
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

# 构建 SAM 模型的内部函数，接受一系列参数
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    # 设置提示嵌入维度
    prompt_embed_dim = 256
    # 设置图像大小
    image_size = 1024
    # 设置 ViT 补丁大小
    vit_patch_size = 16
    # 计算图像嵌入大小
    image_embedding_size = image_size // vit_patch_size
    # 创建一个 Sam 模型实例，包括图像编码器、提示编码器、掩码解码器等组件
    sam = Sam(
        # 图像编码器使用 Vision Transformer 模型，设置相关参数
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,  # ViT 模型的深度
            embed_dim=encoder_embed_dim,  # ViT 模型的嵌入维度
            img_size=image_size,  # 图像尺寸
            mlp_ratio=4,  # MLP 层的比例
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # 规范化层
            num_heads=encoder_num_heads,  # 多头注意力机制的头数
            patch_size=vit_patch_size,  # ViT 模型的补丁大小
            qkv_bias=True,  # 是否使用偏置项
            use_rel_pos=True,  # 是否使用相对位置编码
            global_attn_indexes=encoder_global_attn_indexes,  # 全局注意力索引
            window_size=14,  # 窗口大小
            out_chans=prompt_embed_dim,  # 输出通道数
        ),
        # 提示编码器，设置相关参数
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,  # 提示编码器的嵌入维度
            image_embedding_size=(image_embedding_size, image_embedding_size),  # 图像嵌入大小
            input_image_size=(image_size, image_size),  # 输入图像大小
            mask_in_chans=16,  # 掩码输入通道数
        ),
        # 掩码解码器，设置相关参数
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,  # 多掩码输出数量
            transformer=TwoWayTransformer(
                depth=2,  # 双向 Transformer 模型的深度
                embedding_dim=prompt_embed_dim,  # 嵌入维度
                mlp_dim=2048,  # MLP 层的维度
                num_heads=8,  # 多头注意力机制的头数
            ),
            transformer_dim=prompt_embed_dim,  # Transformer 模型的维度
            iou_head_depth=3,  # IoU 头的深度
            iou_head_hidden_dim=256,  # IoU 头的隐藏层维度
        ),
        pixel_mean=[123.675, 116.28, 103.53],  # 像素均值
        pixel_std=[58.395, 57.12, 57.375],  # 像素标准差
    )
    # 将模型设置为评估模式
    sam.eval()
    # 如果提供了检查点文件，则加载模型状态字典
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)  # 加载模型状态字典
    # 返回 Sam 模型实例
    return sam
```