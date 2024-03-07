# `.\YOLO-World\yolo_world\models\necks\yolo_world_pafpn.py`

```py
# 导入必要的库
import copy
from typing import List, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

# 导入自定义的模型注册器和工具函数
from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN

# 注册YOLOWorldPAFPN类为模型
@MODELS.register_module()
class YOLOWorldPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    # 初始化函数，定义模型结构和参数
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        # 设置引导通道数、嵌入通道数和头数
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        # 调用父类的初始化函数，传入参数
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
    # 构建自顶向下的层
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        # 深拷贝块配置
        block_cfg = copy.deepcopy(self.block_cfg)
        # 更新块配置参数
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        # 构建模型
        return MODELS.build(block_cfg)
    # 构建底部向上的层
    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        # 深拷贝块配置
        block_cfg = copy.deepcopy(self.block_cfg)
        # 更新块配置
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        # 构建模型
        return MODELS.build(block_cfg)
    # 定义前向传播函数，接受多层级的图像特征和文本特征作为输入，返回元组
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        # 断言图像特征的数量与输入通道数相同
        assert len(img_feats) == len(self.in_channels)
        
        # 减少层级
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # 自顶向下路径
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # 自底向上路径
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # 输出层
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
# 使用 @MODELS 注册 YOLOWorldDualPAFPN 类
@MODELS.register_module()
# 定义 YOLOWorldDualPAFPN 类，继承自 YOLOWorldPAFPN 类
class YOLOWorldDualPAFPN(YOLOWorldPAFPN):
    """Path Aggregation Network used in YOLO World v8."""
    # 初始化函数，接受多个参数
    def __init__(self,
                 in_channels: List[int],  # 输入通道列表
                 out_channels: Union[List[int], int],  # 输出通道列表或整数
                 guide_channels: int,  # 引导通道数
                 embed_channels: List[int],  # 嵌入通道列表
                 num_heads: List[int],  # 多头注意力机制的头数列表
                 deepen_factor: float = 1.0,  # 加深因子，默认为1.0
                 widen_factor: float = 1.0,  # 扩宽因子，默认为1.0
                 num_csp_blocks: int = 3,  # CSP块的数量，默认为3
                 freeze_all: bool = False,  # 是否冻结所有层，默认为False
                 text_enhancder: ConfigType = dict(  # 文本增强器配置
                     type='ImagePoolingAttentionModule',  # 类型为图像池化注意力模块
                     embed_channels=256,  # 嵌入通道数为256
                     num_heads=8,  # 多头注意力机制的头数为8
                     pool_size=3),  # 池化大小为3
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),  # 块配置，默认为CSPLayerWithTwoConv
                 norm_cfg: ConfigType = dict(type='BN',  # 归一化配置，默认为BN
                                             momentum=0.03,  # 动量为0.03
                                             eps=0.001),  # epsilon为0.001
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),  # 激活函数配置，默认为SiLU
                 init_cfg: OptMultiConfig = None) -> None:  # 初始化配置，默认为None，返回None
        # 调用父类的初始化函数
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        # 更新文本增强器配置
        text_enhancder.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],  # 图像通道数根据输出通道和扩宽因子计算
                text_channels=guide_channels,  # 文本通道数为引导通道数
                num_feats=len(out_channels),  # 特征数量为输出通道数的长度
            ))
        # 打印文本增强器配置
        print(text_enhancder)
        # 构建文本增强器模型
        self.text_enhancer = MODELS.build(text_enhancder)
    # 定义前向传播函数，接受图像特征列表和文本特征作为输入，返回元组
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""
        # 断言图像特征列表的长度与输入通道数相同
        assert len(img_feats) == len(self.in_channels)
        
        # 减少层
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # 自顶向下路径
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # 对文本特征进行增强
        txt_feats = self.text_enhancer(txt_feats, inner_outs)
        
        # 自底向上路径
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # 输出层
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
```