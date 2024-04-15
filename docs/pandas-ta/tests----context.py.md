# `.\pandas-ta\tests\context.py`

```
# 导入 os 模块，提供对操作系统功能的访问
import os
# 导入 sys 模块，提供对 Python 解释器的访问和控制
import sys

# 将当前文件所在目录的父目录添加到 Python 模块搜索路径中，以便导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入 pandas_ta 模块，该模块提供了一系列用于技术分析的函数和指标
import pandas_ta
```