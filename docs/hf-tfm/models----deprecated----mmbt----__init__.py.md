# `.\models\deprecated\mmbt\__init__.py`

```
# 版权声明和许可证信息，声明版权归 HuggingFace Team 所有，遵循 Apache License, Version 2.0。
# 可以在符合许可证的前提下使用此文件。
# 获取完整许可证内容，请访问指定的 URL。
# 
# 如果当前环境不支持 Torch（PyTorch），则引发 OptionalDependencyNotAvailable 异常。
from typing import TYPE_CHECKING

# 导入可选依赖异常和懒加载模块
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包含待导入的模块和类名
_import_structure = {"configuration_mmbt": ["MMBTConfig"]}

# 检查当前环境是否支持 Torch，若不支持则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Torch，则添加额外的导入结构
    _import_structure["modeling_mmbt"] = ["MMBTForClassification", "MMBTModel", "ModalEmbeddings"]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 导入配置模块中的 MMBTConfig 类
    from .configuration_mmbt import MMBTConfig

    # 再次检查当前环境是否支持 Torch，若不支持则忽略异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入建模模块中的 MMBTForClassification, MMBTModel, ModalEmbeddings 类
        from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings

# 如果当前不是类型检查模式
else:
    import sys

    # 使用懒加载模块来替代当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```