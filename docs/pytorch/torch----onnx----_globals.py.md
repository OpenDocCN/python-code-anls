# `.\pytorch\torch\onnx\_globals.py`

```py
# mypy: allow-untyped-defs
"""
Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""
import torch._C._onnx as _C_onnx
# 导入 _C_onnx 模块，用于访问 ONNX 相关的 C++ 实现

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
# 此模块应仅依赖于 _constants，在 torch.onnx 中不依赖于其他内容，以保持依赖关系的清晰性。
from torch.onnx import _constants

class _InternalGlobals:
    """
    Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self):
        # 设置默认的导出 ONNX 操作集版本为 ONNX_DEFAULT_OPSET
        self._export_onnx_opset_version = _constants.ONNX_DEFAULT_OPSET
        # 设置默认的训练模式为 EVAL
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        # 设置初始状态不在 ONNX 导出过程中
        self._in_onnx_export: bool = False
        # 用户模型在导出期间是否处于训练状态，默认为 False
        self.export_training: bool = False
        # 设置默认的操作符导出类型为 ONNX
        self.operator_export_type: _C_onnx.OperatorExportTypes = (
            _C_onnx.OperatorExportTypes.ONNX
        )
        # 设置默认启用 ONNX 形状推断
        self.onnx_shape_inference: bool = True
        # 设置默认启用自动求导内联
        self._autograd_inlining: bool = True

    @property
    def training_mode(self):
        """The training mode for the exporter."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode):
        if not isinstance(training_mode, _C_onnx.TrainingMode):
            raise TypeError(
                "training_mode must be of type 'torch.onnx.TrainingMode'. This is "
                "likely a bug in torch.onnx."
            )
        self._training_mode = training_mode

    @property
    def export_onnx_opset_version(self) -> int:
        """Opset version used during export."""
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int):
        # 检查所设置的 ONNX opset 版本是否在支持范围内
        supported_versions = range(
            _constants.ONNX_MIN_OPSET, _constants.ONNX_MAX_OPSET + 1
        )
        if value not in supported_versions:
            raise ValueError(f"Unsupported ONNX opset version: {value}")
        self._export_onnx_opset_version = value

    @property
    def in_onnx_export(self) -> bool:
        """Whether it is in the middle of ONNX export."""
        return self._in_onnx_export

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool):
        if type(value) is not bool:
            raise TypeError("in_onnx_export must be a boolean")
        self._in_onnx_export = value

    @property
    def autograd_inlining(self) -> bool:
        """Whether Autograd must be inlined."""
        return self._autograd_inlining

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool):
        if type(value) is not bool:
            raise TypeError("autograd_inlining must be a boolean")
        self._autograd_inlining = value


# 创建 _InternalGlobals 的一个实例，命名为 GLOBALS，用于在 ONNX 导出器内部使用的全局变量管理
GLOBALS = _InternalGlobals()
```