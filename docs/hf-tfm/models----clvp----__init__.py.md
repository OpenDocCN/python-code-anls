# `.\models\clvp\__init__.py`

```py
# 版权声明及版权许可信息
#
# 版权所有2023年HuggingFace团队保留所有权利。
# 
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按“原样”分发，不提供任何明示或暗示的担保或条件。
# 有关具体语言的详情，请参阅许可证。
from typing import TYPE_CHECKING

# 导入必要的依赖模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_clvp": [
        "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
    ],
    "feature_extraction_clvp": ["ClvpFeatureExtractor"],
    "processing_clvp": ["ClvpProcessor"],
    "tokenization_clvp": ["ClvpTokenizer"],
}

# 检查是否有torch可用，如果没有则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，添加modeling_clvp模块到导入结构
    _import_structure["modeling_clvp"] = [
        "CLVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ClvpModelForConditionalGeneration",
        "ClvpForCausalLM",
        "ClvpModel",
        "ClvpPreTrainedModel",
        "ClvpEncoder",
        "ClvpDecoder",
    ]

# 如果是类型检查模式，导入具体的类和方法
if TYPE_CHECKING:
    from .configuration_clvp import (
        CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ClvpConfig,
        ClvpDecoderConfig,
        ClvpEncoderConfig,
    )
    from .feature_extraction_clvp import ClvpFeatureExtractor
    from .processing_clvp import ClvpProcessor
    from .tokenization_clvp import ClvpTokenizer

    # 再次检查torch是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入modeling_clvp模块下的类和方法
        from .modeling_clvp import (
            CLVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ClvpDecoder,
            ClvpEncoder,
            ClvpForCausalLM,
            ClvpModel,
            ClvpModelForConditionalGeneration,
            ClvpPreTrainedModel,
        )

# 如果不是类型检查模式，将模块注册为_LazyModule以延迟导入
else:
    import sys

    # 将当前模块注册为_LazyModule，用于按需导入模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```