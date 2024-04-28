# `.\transformers\models\upernet\__init__.py`

```
# 版权声明及许可证信息

# 是否为类型检查环境
from typing import TYPE_CHECKING

# 导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_upernet": ["UperNetConfig"],
}

# 检查是否存在 torch 包，若不存在则引发依赖不可用的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型导入结构
    _import_structure["modeling_upernet"] = [
        "UperNetForSemanticSegmentation",
        "UperNetPreTrainedModel",
    ]

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 导入 UperNetConfig 配置
    from .configuration_upernet import UperNetConfig
    
    # 再次检查 torch 包是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
        from .modeling_upernet import UperNetForSemanticSegmentation, UperNetPreTrainedModel

# 如果不在类型检查环境下
else:
    import sys
    
    # 将当前模块定义为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```