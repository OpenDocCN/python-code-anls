# `.\transformers\models\patchtsmixer\__init__.py`

```
# 版权声明及许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 在遵守许可证的条件下可以使用本文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 LazyModule 和 is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_patchtsmixer": [
        "PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PatchTSMixerConfig",
    ],
}

# 检查是否有 torch 库可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若有 torch 库可用，则添加对应模型的导入结构
    _import_structure["modeling_patchtsmixer"] = [
        "PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PatchTSMixerPreTrainedModel",
        "PatchTSMixerModel",
        "PatchTSMixerForPretraining",
        "PatchTSMixerForPrediction",
        "PatchTSMixerForTimeSeriesClassification",
        "PatchTSMixerForRegression",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置相关内容
    from .configuration_patchtsmixer import (
        PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PatchTSMixerConfig,
    )
    
    # 再次检查是否有 torch 库可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
        from .modeling_patchtsmixer import (
            PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST,
            PatchTSMixerForPrediction,
            PatchTSMixerForPretraining,
            PatchTSMixerForRegression,
            PatchTSMixerForTimeSeriesClassification,
            PatchTSMixerModel,
            PatchTSMixerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块作为懒加载模块进行导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```