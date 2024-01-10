# `MetaGPT\metagpt\provider\base_llm.py`

```

#!/usr/bin/env python
# 指定解释器为 python，使得脚本可以直接执行

# 设置文件编码为 utf-8
# -*- coding: utf-8 -*-

"""
@Time    : 2023/5/5 23:04
@Author  : alexanderwu
@File    : base_llm.py
@Desc    : mashenquan, 2023/8/22. + try catch
"""
# 文件注释，包括时间、作者、文件名、描述信息

# 导入 json 模块
import json
# 从 abc 模块中导入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod
# 从 typing 模块中导入 Optional 类型提示
from typing import Optional

```