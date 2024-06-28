# `.\models\wavlm\__init__.py`

```
# 引入所需的类型检查模块
from typing import TYPE_CHECKING

# 引入必要的依赖异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_wavlm": ["WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "WavLMConfig"]}

# 检查是否存在 torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加以下模型相关的导入结构
    _import_structure["modeling_wavlm"] = [
        "WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "WavLMForAudioFrameClassification",
        "WavLMForCTC",
        "WavLMForSequenceClassification",
        "WavLMForXVector",
        "WavLMModel",
        "WavLMPreTrainedModel",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 从配置文件中导入所需的配置映射和配置类
    from .configuration_wavlm import WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP, WavLMConfig

    # 再次检查是否存在 torch 库，如果不存在则捕获异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型文件中导入所需的模型类
        from .modeling_wavlm import (
            WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            WavLMForAudioFrameClassification,
            WavLMForCTC,
            WavLMForSequenceClassification,
            WavLMForXVector,
            WavLMModel,
            WavLMPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 使用延迟加载模块，定义当前模块为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```