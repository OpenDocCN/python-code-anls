# `.\models\bigbird_pegasus\__init__.py`

```py
# 版权声明及许可信息
#
# 版权所有 2021 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件分发根据"按原样"的基础分发，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入可选依赖未找到异常和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典
_import_structure = {
    "configuration_bigbird_pegasus": [
        "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BigBirdPegasusConfig",
        "BigBirdPegasusOnnxConfig",
    ],
}

# 检查是否存在 Torch 可用，如果不可用则抛出可选依赖未找到异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则扩展导入结构字典
    _import_structure["modeling_bigbird_pegasus"] = [
        "BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BigBirdPegasusForCausalLM",
        "BigBirdPegasusForConditionalGeneration",
        "BigBirdPegasusForQuestionAnswering",
        "BigBirdPegasusForSequenceClassification",
        "BigBirdPegasusModel",
        "BigBirdPegasusPreTrainedModel",
    ]

# 如果是类型检查阶段，导入配置和模型模块的具体内容
if TYPE_CHECKING:
    from .configuration_bigbird_pegasus import (
        BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BigBirdPegasusConfig,
        BigBirdPegasusOnnxConfig,
    )

    # 再次检查 Torch 是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的内容
        from .modeling_bigbird_pegasus import (
            BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigBirdPegasusForCausalLM,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForQuestionAnswering,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusModel,
            BigBirdPegasusPreTrainedModel,
        )

# 如果不是类型检查阶段，则注册懒加载模块
else:
    import sys

    # 将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```