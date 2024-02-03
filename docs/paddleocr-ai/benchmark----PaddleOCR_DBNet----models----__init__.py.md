# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\__init__.py`

```
# -*- coding: utf-8 -*-
# 定义文件编码格式为 utf-8
# @Time    : 2019/8/23 21:55
# 指定时间信息
# @Author  : zhoujun
# 指定作者信息
import copy
# 导入 copy 模块
from .model import Model
# 从当前目录下的 model 模块中导入 Model 类
from .losses import build_loss
# 从当前目录下的 losses 模块中导入 build_loss 函数

__all__ = ['build_loss', 'build_model']
# 定义模块中可以被导入的对象列表
support_model = ['Model']
# 定义支持的模型列表，包含 Model 类名

def build_model(config):
    """
    get architecture model class
    """
    # 深拷贝配置信息
    copy_config = copy.deepcopy(config)
    # 弹出配置信息中的 'type' 键对应的值
    arch_type = copy_config.pop('type')
    # 断言 'type' 值在支持的模型列表中
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    # 根据 'type' 值创建对应的模型对象
    arch_model = eval(arch_type)(copy_config)
    # 返回创建的模型对象
    return arch_model
```