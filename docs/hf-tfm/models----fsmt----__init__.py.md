# `.\models\fsmt\__init__.py`

```
# 导入类型检查工具
from typing import TYPE_CHECKING

# 导入自定义的异常和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置、标记化和建模组件
_import_structure = {
    "configuration_fsmt": ["FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FSMTConfig"],
    "tokenization_fsmt": ["FSMTTokenizer"],
}

# 检查是否支持 Torch，如果不支持则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Torch，则添加建模组件到导入结构中
    _import_structure["modeling_fsmt"] = ["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从配置、标记化和建模模块导入特定类和类型
    from .configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
    from .tokenization_fsmt import FSMTTokenizer

    # 再次检查 Torch 是否可用，并在可用时导入建模类和类型
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块重新定义为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```