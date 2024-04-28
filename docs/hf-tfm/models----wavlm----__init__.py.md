# `.\transformers\models\wavlm\__init__.py`

```
# 版权声明
# 版权声明，保留所有权利
# 根据 Apache 许可证 2.0 版（“许可证”）授权
# 除非在遵守许可证的情况下，您不得使用此文件
# 您可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律或书面同意要求，在“AS IS”基础上分发软件
# 没有任何担保或条件，无论是明示的还是默示的
# 详见许可证，规定了特定语言的权限和限制

# 引入必要的依赖库
from typing import TYPE_CHECKING
# 引入依赖工具模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {"configuration_wavlm": ["WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "WavLMConfig"]}

# 尝试引入 torch，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果引入成功，则加入到导入结构中
else:
    _import_structure["modeling_wavlm"] = [
        "WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "WavLMForAudioFrameClassification",
        "WavLMForCTC",
        "WavLMForSequenceClassification",
        "WavLMForXVector",
        "WavLMModel",
        "WavLMPreTrainedModel",
    ]

# 如果是类型检查时
if TYPE_CHECKING:
    # 从配置模块中引入所需内容
    from .configuration_wavlm import WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP, WavLMConfig
    # 尝试引入 torch，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果引入成功，则从建模模块中引入所需内容
    else:
        from .modeling_wavlm import (
            WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            WavLMForAudioFrameClassification,
            WavLMForCTC,
            WavLMForSequenceClassification,
            WavLMForXVector,
            WavLMModel,
            WavLMPreTrainedModel,
        )

# 如果不是类型检查时
else:
    # 引入 sys 模块
    import sys
    # 将当前模块指定为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```