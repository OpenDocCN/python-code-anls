# `.\models\kosmos2\__init__.py`

```
# coding=utf-8
# 声明文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 进行许可
# 除非适用法律要求或书面同意，否则按“原样”分发此软件
# 无论是明示的还是暗示的，都没有任何担保或条件
# 请参阅许可证获取许可的特定语言和权限
from typing import TYPE_CHECKING
# 引入类型检查相关模块

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)
# 从工具模块中引入依赖项检查和懒加载模块

# 定义需要导入的模块结构
_import_structure = {
    "configuration_kosmos2": ["KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Kosmos2Config"],
    "processing_kosmos2": ["Kosmos2Processor"],
}

try:
    # 如果 Torch 不可用，则引发可选依赖项不可用异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果发生可选依赖项不可用异常，则不执行任何操作
    pass
else:
    # 如果 Torch 可用，则添加模型相关的模块到导入结构中
    _import_structure["modeling_kosmos2"] = [
        "KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Kosmos2ForConditionalGeneration",
        "Kosmos2Model",
        "Kosmos2PreTrainedModel",
    ]


if TYPE_CHECKING:
    # 如果处于类型检查模式下，则导入配置和处理相关模块
    from .configuration_kosmos2 import KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP, Kosmos2Config
    from .processing_kosmos2 import Kosmos2Processor

    try:
        # 如果 Torch 不可用，则引发可选依赖项不可用异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果发生可选依赖项不可用异常，则不执行任何操作
        pass
    else:
        # 如果 Torch 可用，则导入模型相关模块
        from .modeling_kosmos2 import (
            KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Kosmos2ForConditionalGeneration,
            Kosmos2Model,
            Kosmos2PreTrainedModel,
        )

else:
    # 如果不处于类型检查模式下，则进行懒加载
    import sys

    # 用 LazyModule 替换当前模块，使其支持懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```