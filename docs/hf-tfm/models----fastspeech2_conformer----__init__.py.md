# `.\models\fastspeech2_conformer\__init__.py`

```py
# 版权声明和许可条款，声明代码归 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则禁止使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的担保或条件。详见许可证条款
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需的依赖项
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，包含模块名称及其导出的成员列表
_import_structure = {
    "configuration_fastspeech2_conformer": [
        "FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "tokenization_fastspeech2_conformer": ["FastSpeech2ConformerTokenizer"],
}

# 检查是否可用 torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_fastspeech2_conformer 模块的导入结构
    _import_structure["modeling_fastspeech2_conformer"] = [
        "FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FastSpeech2ConformerWithHifiGan",
        "FastSpeech2ConformerHifiGan",
        "FastSpeech2ConformerModel",
        "FastSpeech2ConformerPreTrainedModel",
    ]

# 如果是类型检查模式，从各自模块导入相关类和常量
if TYPE_CHECKING:
    from .configuration_fastspeech2_conformer import (
        FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    from .tokenization_fastspeech2_conformer import FastSpeech2ConformerTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_fastspeech2_conformer import (
            FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
            FastSpeech2ConformerWithHifiGan,
        )

# 如果不是类型检查模式，则导入 sys 模块并用 _LazyModule 替换当前模块
else:
    import sys

    # 用 _LazyModule 替换当前模块，延迟导入模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```