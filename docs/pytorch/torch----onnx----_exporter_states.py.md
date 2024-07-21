# `.\pytorch\torch\onnx\_exporter_states.py`

```
from __future__ import annotations
# 导入了用于未来版本兼容的 annotations 特性

from typing import Dict
# 导入了 Dict 类型提示，用于声明字典类型的变量

from torch import _C
# 导入了 Torch 库的 _C 模块

class ExportTypes:
    r"""Specifies how the ONNX model is stored."""
    # 定义了一个枚举类 ExportTypes，用于指定 ONNX 模型存储方式

    PROTOBUF_FILE = "Saves model in the specified protobuf file."
    # 枚举类型：将模型保存在指定的 protobuf 文件中

    ZIP_ARCHIVE = "Saves model in the specified ZIP file (uncompressed)."
    # 枚举类型：将模型保存在指定的 ZIP 文件中（未压缩）

    COMPRESSED_ZIP_ARCHIVE = "Saves model in the specified ZIP file (compressed)."
    # 枚举类型：将模型保存在指定的 ZIP 文件中（已压缩）

    DIRECTORY = "Saves model in the specified folder."
    # 枚举类型：将模型保存在指定的文件夹中

class SymbolicContext:
    """Extra context for symbolic functions.

    Args:
        params_dict (Dict[str, _C.IValue]): Mapping from graph initializer name to IValue.
            参数 params_dict: 字典类型，将图的初始化名称映射到 IValue 对象
        env (Dict[_C.Value, _C.Value]): Mapping from Torch domain graph Value to ONNX domain graph Value.
            参数 env: 字典类型，将 Torch 领域图中的 Value 映射到 ONNX 领域图中的 Value
        cur_node (_C.Node): Current node being converted to ONNX domain.
            参数 cur_node: _C.Node 类型，正在转换为 ONNX 领域的当前节点
        onnx_block (_C.Block): Current ONNX block that converted nodes are being appended to.
            参数 onnx_block: _C.Block 类型，正在追加转换节点的当前 ONNX 块
    """

    def __init__(
        self,
        params_dict: Dict[str, _C.IValue],
        env: dict,
        cur_node: _C.Node,
        onnx_block: _C.Block,
    ):
        self.params_dict: Dict[str, _C.IValue] = params_dict
        # 初始化 params_dict 属性为给定的 params_dict 参数

        self.env: Dict[_C.Value, _C.Value] = env
        # 初始化 env 属性为给定的 env 参数

        # Current node that is being converted.
        self.cur_node: _C.Node = cur_node
        # 初始化 cur_node 属性为给定的 cur_node 参数，并注释其为当前正在转换的节点

        # Current onnx block that converted nodes are being appended to.
        self.onnx_block: _C.Block = onnx_block
        # 初始化 onnx_block 属性为给定的 onnx_block 参数，并注释其为当前正在追加转换节点的 ONNX 块
```