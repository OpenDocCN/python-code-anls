# `.\pytorch\torch\nn\intrinsic\modules\fused.py`

```
# 导入 torch 中 AO（自动优化）模块下的神经网络组件，包括批归一化 + ReLU 的二维版本
from torch.ao.nn.intrinsic import BNReLU2d
# 导入 torch 中 AO 模块下的神经网络组件，包括批归一化 + ReLU 的三维版本
from torch.ao.nn.intrinsic import BNReLU3d
# 导入 torch 中 AO 模块下的神经网络组件，包括一维卷积 + 批归一化
from torch.ao.nn.intrinsic import ConvBn1d
# 导入 torch 中 AO 模块下的神经网络组件，包括二维卷积 + 批归一化
from torch.ao.nn.intrinsic import ConvBn2d
# 导入 torch 中 AO 模块下的神经网络组件，包括三维卷积 + 批归一化
from torch.ao.nn.intrinsic import ConvBn3d
# 导入 torch 中 AO 模块下的神经网络组件，包括一维卷积 + 批归一化 + ReLU
from torch.ao.nn.intrinsic import ConvBnReLU1d
# 导入 torch 中 AO 模块下的神经网络组件，包括二维卷积 + 批归一化 + ReLU
from torch.ao.nn.intrinsic import ConvBnReLU2d
# 导入 torch 中 AO 模块下的神经网络组件，包括三维卷积 + 批归一化 + ReLU
from torch.ao.nn.intrinsic import ConvBnReLU3d
# 导入 torch 中 AO 模块下的神经网络组件，包括一维卷积 + ReLU
from torch.ao.nn.intrinsic import ConvReLU1d
# 导入 torch 中 AO 模块下的神经网络组件，包括二维卷积 + ReLU
from torch.ao.nn.intrinsic import ConvReLU2d
# 导入 torch 中 AO 模块下的神经网络组件，包括三维卷积 + ReLU
from torch.ao.nn.intrinsic import ConvReLU3d
# 导入 torch 中 AO 模块下的神经网络组件，包括一维线性 + 批归一化
from torch.ao.nn.intrinsic import LinearBn1d
# 导入 torch 中 AO 模块下的神经网络组件，包括线性 + ReLU
from torch.ao.nn.intrinsic import LinearReLU
# 导入 torch 中 AO 模块下的融合模块，用于神经网络优化，不使用 F401 错误检查
from torch.ao.nn.intrinsic.modules.fused import _FusedModule  # noqa: F401

# 将所有导入的模块名称放入 __all__ 列表中，用于模块的自动导入
__all__ = [
    'BNReLU2d',
    'BNReLU3d',
    'ConvBn1d',
    'ConvBn2d',
    'ConvBn3d',
    'ConvBnReLU1d',
    'ConvBnReLU2d',
    'ConvBnReLU3d',
    'ConvReLU1d',
    'ConvReLU2d',
    'ConvReLU3d',
    'LinearBn1d',
    'LinearReLU',
]
```