# `.\transformers\models\autoformer\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入懒加载模块和必要的依赖异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_autoformer": [
        "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 自动编码器预训练配置文件映射
        "AutoformerConfig",  # 自动编码器配置类
    ],
}

# 检查是否导入了 Torch 库，若未导入则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 Torch 不可用，直接忽略
else:
    # 如果 Torch 可用，则扩展导入结构，包括自动编码器相关模块
    _import_structure["modeling_autoformer"] = [
        "AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 自动编码器预训练模型存档列表
        "AutoformerForPrediction",  # 自动编码器预测类
        "AutoformerModel",  # 自动编码器模型类
        "AutoformerPreTrainedModel",  # 自动编码器预训练模型基类
    ]

# 如果正在进行类型检查，则执行下面的代码块
if TYPE_CHECKING:
    # 从自动编码器配置模块中导入必要的内容
    from .configuration_autoformer import (
        AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 自动编码器预训练配置文件映射
        AutoformerConfig,  # 自动编码器配置类
    )

    # 再次检查 Torch 是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 如果 Torch 不可用，直接忽略
    else:
        # 如果 Torch 可用，则从自动编码器建模模块中导入必要的内容
        from .modeling_autoformer import (
            AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 自动编码器预训练模型存档列表
            AutoformerForPrediction,  # 自动编码器预测类
            AutoformerModel,  # 自动编码器模型类
            AutoformerPreTrainedModel,  # 自动编码器预训练模型基类
        )

# 如果不是类型检查，则执行下面的代码块
else:
    # 导入系统模块
    import sys

    # 将当前模块注册为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```