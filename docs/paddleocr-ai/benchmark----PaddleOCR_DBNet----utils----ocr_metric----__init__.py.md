# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\ocr_metric\__init__.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 15:36
# @Author  : zhoujun
# 导入 QuadMetric 类
from .icdar2015 import QuadMetric

# 定义一个函数，根据配置获取指定的度量标准对象
def get_metric(config):
    # 尝试执行以下代码，如果出现异常则返回 None
    try:
        # 如果配置中没有 'args' 键，则创建一个空字典
        if 'args' not in config:
            args = {}
        else:
            # 否则将 'args' 对应的值赋给 args
            args = config['args']
        
        # 如果 args 是字典类型，则使用 eval 函数根据 'type' 键的值创建对象
        if isinstance(args, dict):
            cls = eval(config['type'])(**args)
        else:
            # 否则直接使用 'type' 键的值创建对象
            cls = eval(config['type'])(args)
        
        # 返回创建的对象
        return cls
    except:
        # 如果出现异常则返回 None
        return None
```