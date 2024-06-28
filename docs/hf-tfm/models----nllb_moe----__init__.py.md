# `.\models\nllb_moe\__init__.py`

```
# 版权声明和许可信息，指明代码版权和许可的使用条款
# The HuggingFace Team 版权声明和保留所有权利

# 引入类型检查模块中的 TYPE_CHECKING 类型
from typing import TYPE_CHECKING

# 引入可选依赖未安装时的异常处理和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典
_import_structure = {
    "configuration_nllb_moe": [
        "NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "NllbMoeConfig",
    ]
}

# 检查是否有必要导入 Torch 库，若未安装则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则将相关模型定义添加到导入结构字典中
    _import_structure["modeling_nllb_moe"] = [
        "NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NllbMoeForConditionalGeneration",
        "NllbMoeModel",
        "NllbMoePreTrainedModel",
        "NllbMoeTop2Router",
        "NllbMoeSparseMLP",
    ]

# 如果是类型检查阶段，则导入相应的配置和模型类
if TYPE_CHECKING:
    from .configuration_nllb_moe import (
        NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        NllbMoeConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_nllb_moe import (
            NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST,
            NllbMoeForConditionalGeneration,
            NllbMoeModel,
            NllbMoePreTrainedModel,
            NllbMoeSparseMLP,
            NllbMoeTop2Router,
        )

# 如果不是类型检查阶段，则将当前模块设为延迟加载模块
else:
    import sys

    # 使用 LazyModule 类在 sys.modules 中注册当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```