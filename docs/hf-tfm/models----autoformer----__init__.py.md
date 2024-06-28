# `.\models\autoformer\__init__.py`

```py
# 版权声明和许可信息
# 该模块是 HuggingFace 团队的代码，版权归其所有
# 根据 Apache 许可证 2.0 版本进行许可
# 如果不遵循许可证，除非适用法律要求或书面同意，否则不得使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本

# 引入类型检查模块
from typing import TYPE_CHECKING

# 从工具模块中引入异常和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构字典
_import_structure = {
    "configuration_autoformer": [
        "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 自动化配置的预训练配置映射
        "AutoformerConfig",  # Autoformer 的配置类
    ],
}

# 尝试检查是否存在 Torch 可用，若不存在则抛出异常 OptionalDependencyNotAvailable
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则更新导入结构字典以包含建模组件
    _import_structure["modeling_autoformer"] = [
        "AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 自动化模型的预训练模型档案列表
        "AutoformerForPrediction",  # 用于预测的 Autoformer 模型
        "AutoformerModel",  # Autoformer 模型
        "AutoformerPreTrainedModel",  # Autoformer 预训练模型
    ]

# 如果处于类型检查模式
if TYPE_CHECKING:
    # 从自动化配置模块中导入相关内容
    from .configuration_autoformer import (
        AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 自动化配置的预训练配置映射
        AutoformerConfig,  # Autoformer 的配置类
    )

    # 尝试检查是否存在 Torch 可用，若不存在则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从建模模块中导入相关内容
        from .modeling_autoformer import (
            AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 自动化模型的预训练模型档案列表
            AutoformerForPrediction,  # 用于预测的 Autoformer 模型
            AutoformerModel,  # Autoformer 模型
            AutoformerPreTrainedModel,  # Autoformer 预训练模型
        )

# 如果不处于类型检查模式
else:
    import sys

    # 将当前模块替换为懒加载模块，实现按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```