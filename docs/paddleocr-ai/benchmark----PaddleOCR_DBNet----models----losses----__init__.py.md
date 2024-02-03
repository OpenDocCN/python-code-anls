# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\losses\__init__.py`

```py
# -*- coding: utf-8 -*-
# 定义文件编码格式为 UTF-8
# @Time    : 2020/6/5 11:36
# 定义作者和时间信息
# @Author  : zhoujun
# 导入 copy 模块
import copy
# 从当前目录下的 DB_loss 模块中导入 DBLoss 类
from .DB_loss import DBLoss

# 定义可以被外部导入的模块列表
__all__ = ['build_loss']
# 支持的损失函数列表
support_loss = ['DBLoss']

# 定义构建损失函数的函数
def build_loss(config):
    # 深拷贝配置信息
    copy_config = copy.deepcopy(config)
    # 弹出配置信息中的损失函数类型
    loss_type = copy_config.pop('type')
    # 断言损失函数类型在支持的损失函数列表中
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    # 根据损失函数类型和配置信息创建损失函数对象
    criterion = eval(loss_type)(**copy_config)
    # 返回创建的损失函数对象
    return criterion
```