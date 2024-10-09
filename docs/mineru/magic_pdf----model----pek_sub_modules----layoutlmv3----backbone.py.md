# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\backbone.py`

```
# --------------------------------------------------------------------------------
# VIT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


import torch  # 导入 PyTorch 库以支持张量和深度学习功能

from detectron2.layers import (  # 从 detectron2 库中导入层的相关功能
    ShapeSpec,  # 导入 ShapeSpec 类，用于表示张量的形状规格
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN  # 导入 Backbone、BACKBONE_REGISTRY 和 FPN
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool  # 导入特定的 FPN 层

from .beit import beit_base_patch16, dit_base_patch16, dit_large_patch16, beit_large_patch16  # 导入 BEIT 模型
from .deit import deit_base_patch16, mae_base_patch16  # 导入 DEIT 模型
from .layoutlmft.models.layoutlmv3 import LayoutLMv3Model  # 导入 LayoutLMv3 模型
from transformers import AutoConfig  # 从 transformers 库导入自动配置功能

__all__ = [  # 定义模块导出接口
    "build_vit_fpn_backbone",  # 指定可导出的函数
]


class VIT_Backbone(Backbone):  # 定义 VIT_Backbone 类，继承自 Backbone
    """
    Implement VIT backbone.  # 该类实现 VIT 的主干网络
    """

    def forward(self, x):  # 定义前向传播方法
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.  # 输入的张量形状要求

        Returns:
            dict[str->Tensor]: names and the corresponding features  # 返回特征名称与对应张量的字典
        """
        if "layoutlmv3" in self.name:  # 检查模型名称是否包含 'layoutlmv3'
            return self.backbone.forward(  # 调用 backbone 的前向传播方法
                input_ids=x["input_ids"] if "input_ids" in x else None,  # 获取输入 ID
                bbox=x["bbox"] if "bbox" in x else None,  # 获取边界框
                images=x["images"] if "images" in x else None,  # 获取图像数据
                attention_mask=x["attention_mask"] if "attention_mask" in x else None,  # 获取注意力掩码
                # output_hidden_states=True,  # 可选参数，是否输出隐藏状态
            )
        assert x.dim() == 4, f"VIT takes an input of shape (N, C, H, W). Got {x.shape} instead!"  # 确保输入是四维张量
        return self.backbone.forward_features(x)  # 调用 backbone 的特征提取方法

    def output_shape(self):  # 定义输出形状的方法
        return {  # 返回一个字典，包含每个输出特征的形状规格
            name: ShapeSpec(  # 创建 ShapeSpec 实例
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]  # 包含通道和步幅信息
            )
            for name in self._out_features  # 遍历输出特征名称
        }


def build_VIT_backbone(cfg):  # 定义创建 VIT 主干网络的函数
    """
    Create a VIT instance from config.  # 根据配置创建 VIT 实例

    Args:
        cfg: a detectron2 CfgNode  # 输入参数是 detectron2 的配置节点

    Returns:
        A VIT backbone instance.  # 返回一个 VIT 主干网络实例
    """
    # fmt: off  # 禁用格式化以保持代码格式
    name = cfg.MODEL.VIT.NAME  # 从配置中获取模型名称
    out_features = cfg.MODEL.VIT.OUT_FEATURES  # 从配置中获取输出特征
    drop_path = cfg.MODEL.VIT.DROP_PATH  # 从配置中获取丢弃路径参数
    img_size = cfg.MODEL.VIT.IMG_SIZE  # 从配置中获取图像大小
    pos_type = cfg.MODEL.VIT.POS_TYPE  # 从配置中获取位置类型

    model_kwargs = eval(str(cfg.MODEL.VIT.MODEL_KWARGS).replace("`", ""))  # 解析模型参数
    # 检查模型名称是否包含 'layoutlmv3'
        if 'layoutlmv3' in name:
            # 如果配置路径不为空，则使用配置路径
            if cfg.MODEL.CONFIG_PATH != '':
                config_path = cfg.MODEL.CONFIG_PATH
            else:
                # 否则，从权重路径中去掉 'pytorch_model.bin' 以获取配置路径
                config_path = cfg.MODEL.WEIGHTS.replace('pytorch_model.bin', '')  # layoutlmv3 预训练模型
                # 从权重路径中去掉 'model_final.pth' 以获取配置路径
                config_path = config_path.replace('model_final.pth', '')  # 检测微调模型
        else:
            # 如果模型名称不包含 'layoutlmv3'，则配置路径为 None
            config_path = None
    
        # 返回 VIT_Backbone 对象，传入相应的参数
        return VIT_Backbone(name, out_features, drop_path, img_size, pos_type, model_kwargs,
                            config_path=config_path, image_only=cfg.MODEL.IMAGE_ONLY, cfg=cfg)
# 注册一个新的骨干网络构建函数到 BACKBONE_REGISTRY
@BACKBONE_REGISTRY.register()
# 定义构建 VIT FPN 骨干网络的函数
def build_vit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    创建一个 VIT 与 FPN 结合的骨干网络。

    参数:
        cfg: 一个 detectron2 的 CfgNode 配置节点

    返回:
        backbone (Backbone): 骨干模块，必须是 :class:`Backbone` 的子类。
    """
    # 根据配置创建 VIT 骨干网络
    bottom_up = build_VIT_backbone(cfg)
    # 从配置中获取输入特征的名称
    in_features = cfg.MODEL.FPN.IN_FEATURES
    # 从配置中获取输出通道的数量
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    # 创建 FPN 网络，使用先前创建的 VIT 骨干网络
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        # 从配置中获取归一化方式
        norm=cfg.MODEL.FPN.NORM,
        # 使用最大池化作为顶层块
        top_block=LastLevelMaxPool(),
        # 从配置中获取融合类型
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    # 返回构建的骨干网络
    return backbone
```