# `.\models\falcon\__init__.py`

```py
# coding=utf-8
# 版权声明
# 本代码版权归 Falcon 作者和 HuggingFace 公司团队所有。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的明示或暗示的保证或条件。
# 请参阅许可证了解特定的语言管辖权和权限。
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需的依赖
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义导入结构
_import_structure = {
    "configuration_falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],  # 导入配置相关内容
}

# 检查是否有 torch 可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若有 torch 可用，则导入模型相关内容
    _import_structure["modeling_falcon"] = [
        "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FalconForCausalLM",
        "FalconModel",
        "FalconPreTrainedModel",
        "FalconForSequenceClassification",
        "FalconForTokenClassification",
        "FalconForQuestionAnswering",
    ]


if TYPE_CHECKING:
    # 如果是类型检查阶段，则导入相关内容
    from .configuration_falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_falcon import (
            FALCON_PRETRAINED_MODEL_ARCHIVE_LIST,
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )


else:
    # 如果不是类型检查阶段，则进行懒加载
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

```  
```