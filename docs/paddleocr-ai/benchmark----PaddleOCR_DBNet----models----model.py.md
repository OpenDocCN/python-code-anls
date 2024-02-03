# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\model.py`

```
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
# 导入需要的模块
from addict import Dict
from paddle import nn
import paddle.nn.functional as F

# 导入自定义模块
from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head

# 定义模型类
class Model(nn.Layer):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        # 将传入的模型配置转换为字典类型
        model_config = Dict(model_config)
        # 获取并移除backbone、neck、head的类型
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        # 构建backbone、neck、head模块
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(
            neck_type,
            in_channels=self.backbone.out_channels,
            **model_config.neck)
        self.head = build_head(
            head_type, in_channels=self.neck.out_channels, **model_config.head)
        # 设置模型名称
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        # 获取输入张量的高度和宽度
        _, _, H, W = x.shape
        # 通过backbone模块处理输入张量
        backbone_out = self.backbone(x)
        # 通过neck模块处理backbone的输出
        neck_out = self.neck(backbone_out)
        # 通过head模块处理neck的输出
        y = self.head(neck_out)
        # 对输出进行插值，调整大小为输入张量的大小
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        # 返回处理后的输出
        return y
```