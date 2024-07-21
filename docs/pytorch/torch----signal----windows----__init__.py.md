# `.\pytorch\torch\signal\windows\__init__.py`

```py
# 从当前目录下的 windows 模块中导入多个窗口函数
from .windows import (
    bartlett,          # 巴特利特窗口函数
    blackman,          # 布莱克曼窗口函数
    cosine,            # 余弦窗口函数
    exponential,       # 指数窗口函数
    gaussian,          # 高斯窗口函数
    general_cosine,    # 一般余弦窗口函数
    general_hamming,   # 一般 Hamming 窗口函数
    hamming,           # Hamming 窗口函数
    hann,              # Hann 窗口函数
    kaiser,            # Kaiser 窗口函数
    nuttall,           # Nuttall 窗口函数
)

# 模块中导出的所有窗口函数的名称列表
__all__ = [
    'bartlett',        # 巴特利特窗口函数
    'blackman',        # 布莱克曼窗口函数
    'cosine',          # 余弦窗口函数
    'exponential',     # 指数窗口函数
    'gaussian',        # 高斯窗口函数
    'general_cosine',  # 一般余弦窗口函数
    'general_hamming', # 一般 Hamming 窗口函数
    'hamming',         # Hamming 窗口函数
    'hann',            # Hann 窗口函数
    'kaiser',          # Kaiser 窗口函数
    'nuttall',         # Nuttall 窗口函数
]
```