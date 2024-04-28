# `.\models\deprecated\mmbt\__init__.py`

```py
# 版权声明及许可声明信息
# 版权声明及许可声明信息
# 版权声明及许可声明信息
# 版权声明及许可声明信息
# 根据给定的类型检查模块导入相关依赖
from typing import TYPE_CHECKING

# 导入必要的依赖
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_mmbt": ["MMBTConfig"]}

# 尝试导入 torch 模块，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 可用，则扩展模块导入结构
    _import_structure["modeling_mmbt"] = ["MMBTForClassification", "MMBTModel", "ModalEmbeddings"]

# 若类型检查模块存在，则从相应的配置模块导入 MMBTConfig 类
if TYPE_CHECKING:
    from .configuration_mmbt import MMBTConfig
    
    # 尝试导入 torch 模块，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 torch 可用，则从模型构建模块导入相应类和函数
        from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings

# 若没有进行类型检查，则将 LazyModule 对象赋值给当前模块
else:
    # 导入 sys 模块
    import sys

    # 将 LazyModule 对象赋值给当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```