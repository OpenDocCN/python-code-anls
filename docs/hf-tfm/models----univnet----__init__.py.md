# `.\transformers\models\univnet\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,  # 导入自定义异常类
    _LazyModule,  # 导入懒加载模块
    is_torch_available,  # 导入检查是否可用 torch 的函数
)


# 定义模块导入结构
_import_structure = {
    "configuration_univnet": [  # 导入 UNIVNET 的配置模块
        "UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件的映射字典
        "UnivNetConfig",  # UNIVNET 的配置类
    ],
    "feature_extraction_univnet": [  # 导入 UNIVNET 的特征提取模块
        "UnivNetFeatureExtractor",  # UNIVNET 的特征提取器类
    ],
}

# 检查是否可用 torch，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 UNIVNET 的建模模块到导入结构
    _import_structure["modeling_univnet"] = [
        "UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型文件的列表
        "UnivNetModel",  # UNIVNET 的模型类
    ]


# 如果是类型检查模式，导入特定的类型
if TYPE_CHECKING:
    from .configuration_univnet import (
        UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件的映射字典
        UnivNetConfig,  # UNIVNET 的配置类
    )
    from .feature_extraction_univnet import UnivNetFeatureExtractor  # UNIVNET 的特征提取器类

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_univnet import (
            UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型文件的列表
            UnivNetModel,  # UNIVNET 的模型类
        )

# 如果不是类型检查模式，将当前模块指定为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```