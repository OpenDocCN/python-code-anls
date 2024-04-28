# `.\models\fuyu\__init__.py`

```py
# 导入必要的模块和依赖
from typing import TYPE_CHECKING  # 导入类型检查模块

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available  # 导入必要的工具和函数


_import_structure = {
    "configuration_fuyu": ["FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP", "FuyuConfig"],  # 从 configuration_fuyu 模块导入指定内容
}


try:
    if not is_vision_available():  # 如果视觉模块不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常
except OptionalDependencyNotAvailable:  # 处理可选依赖不可用的异常
    pass  # 什么也不做
else:  # 如果视觉模块可用
    _import_structure["image_processing_fuyu"] = ["FuyuImageProcessor"]  # 从 image_processing_fuyu 模块导入指定内容
    _import_structure["processing_fuyu"] = ["FuyuProcessor"]  # 从 processing_fuyu 模块导入指定内容


try:
    if not is_torch_available():  # 如果 torch 模块不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常
except OptionalDependencyNotAvailable:  # 处理可选依赖不可用的异常
    pass  # 什么也不做
else:  # 如果 torch 模块可用
    _import_structure["modeling_fuyu"] = [  # 从 modeling_fuyu 模块导入指定内容
        "FuyuForCausalLM",
        "FuyuPreTrainedModel",
    ]


if TYPE_CHECKING:  # 如果是类型检查模式
    from .configuration_fuyu import FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP, FuyuConfig  # 导入 configuration_fuyu 模块指定内容

    try:
        if not is_vision_available():  # 如果视觉模块不可用
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常
    except OptionalDependencyNotAvailable:  # 处理可选依赖不可用的异常
        pass  # 什么也不做
    else:  # 如果视觉模块可用
        from .image_processing_fuyu import FuyuImageProcessor  # 导入 image_processing_fuyu 模块指定内容
        from .processing_fuyu import FuyuProcessor  # 导入 processing_fuyu 模块指定内容

    try:
        if not is_torch_available():  # 如果 torch 模块不可用
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常
    except OptionalDependencyNotAvailable:  # 处理可选依赖不可用的异常
        pass  # 什么也不做
    else:  # 如果 torch 模块可用
        from .modeling_fuyu import (  # 导入 modeling_fuyu 模块指定内容
            FuyuForCausalLM,
            FuyuPreTrainedModel,
        )


else:  # 如果不是类型检查模式
    import sys  # 导入 sys 模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)  # 将当前模块设置为懒加载模块
```