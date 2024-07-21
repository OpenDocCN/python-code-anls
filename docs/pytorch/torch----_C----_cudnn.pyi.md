# `.\pytorch\torch\_C\_cudnn.pyi`

```py
# 导入 Enum 类型，用于定义枚举类型变量
from enum import Enum

# 导入类型声明 _bool 和 Tuple，来自 torch.types 模块
from torch.types import _bool, Tuple

# 声明全局变量 is_cuda，类型为 _bool，在 torch/csrc/cuda/shared/cudnn.cpp 文件中定义
is_cuda: _bool

# 声明函数 getRuntimeVersion，返回一个包含三个整数的元组
def getRuntimeVersion() -> Tuple[int, int, int]: ...

# 声明函数 getCompileVersion，返回一个包含三个整数的元组
def getCompileVersion() -> Tuple[int, int, int]: ...

# 声明函数 getVersionInt，返回一个整数
def getVersionInt() -> int: ...

# 定义枚举类 RNNMode，继承自 int 类型和 Enum 类型
class RNNMode(int, Enum):
    # 定义枚举成员变量 value
    value: int
    # 定义枚举成员 rnn_relu，未指定具体值
    rnn_relu = ...
    # 定义枚举成员 rnn_tanh，未指定具体值
    rnn_tanh = ...
    # 定义枚举成员 lstm，未指定具体值
    lstm = ...
    # 定义枚举成员 gru，未指定具体值
    gru = ...
```