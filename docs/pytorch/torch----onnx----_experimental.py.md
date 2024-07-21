# `.\pytorch\torch\onnx\_experimental.py`

```py
"""
Experimental classes and functions used by ONNX export.
"""

# 导入必要的库和模块
import dataclasses  # 导入用于数据类的模块
from typing import Mapping, Optional, Sequence, Set, Type, Union  # 导入类型提示需要的类型

import torch  # 导入PyTorch库
import torch._C._onnx as _C_onnx  # 导入PyTorch内部的ONNX相关模块


@dataclasses.dataclass
class ExportOptions:
    """
    Arguments used by :func:`torch.onnx.export`.

    TODO: Adopt this in `torch.onnx.export` api to replace keyword arguments.
    """

    # 导出参数，默认为True
    export_params: bool = True
    # 是否详细输出信息，默认为False
    verbose: bool = False
    # ONNX导出时的训练模式，默认为EVAL
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
    # 输入张量的名称列表，可选参数
    input_names: Optional[Sequence[str]] = None
    # 输出张量的名称列表，可选参数
    output_names: Optional[Sequence[str]] = None
    # 操作符导出类型，默认为ONNX
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX
    # 使用的操作集版本号，可选参数
    opset_version: Optional[int] = None
    # 是否进行常量折叠，默认为True
    do_constant_folding: bool = True
    # 动态轴的定义，用于动态维度的张量，可选参数
    dynamic_axes: Optional[Mapping[str, Union[Mapping[int, str], Sequence[int]]]] = None
    # 是否保留初始化器作为输入，默认为None
    keep_initializers_as_inputs: Optional[bool] = None
    # 自定义操作集的版本映射表，可选参数
    custom_opsets: Optional[Mapping[str, int]] = None
    # 是否将模块导出为函数，默认为False
    export_modules_as_functions: Union[bool, Set[Type[torch.nn.Module]]] = False
```