# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\backbone\__init__.py`

```py
# -*- coding: utf-8 -*-
# 定义文件编码格式为 utf-8
# @Time    : 2019/8/23 21:54
# 定义作者和时间信息

from .resnet import *
# 从 resnet 模块中导入所有内容

__all__ = ['build_backbone']
# 定义模块中对外暴露的接口

support_backbone = [
    'resnet18', 'deformable_resnet18', 'deformable_resnet50', 'resnet50',
    'resnet34', 'resnet101', 'resnet152'
]
# 支持的骨干网络名称列表

def build_backbone(backbone_name, **kwargs):
    # 断言所选的骨干网络名称在支持的列表中，否则抛出异常
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    # 根据所选的骨干网络名称动态创建对应的骨干网络对象
    backbone = eval(backbone_name)(**kwargs)
    # 返回创建的骨干网络对象
    return backbone
```