# `.\transformers\models\yoso\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明：2022 年 HuggingFace 团队。保留所有权利。
# 根据 Apache 许可证 2.0 版（"许可证"）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据"原样"的基础分发，没有任何担保或条件，
# 明示或暗示。查看许可证以了解具体语言的权限
# 以及限制。
from typing import TYPE_CHECKING  # 导入 TYPE_CHECKING 类型提示

# 从 ...utils 中导入 OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构，包括 configuration_yoso 模块中的一些变量和类名
_import_structure = {"configuration_yoso": ["YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP", "YosoConfig"]}

# 检查 torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 可用，则将 modeling_yoso 模块中的一些变量和类名加入导入结构
    _import_structure["modeling_yoso"] = [
        "YOSO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "YosoForMaskedLM",
        "YosoForMultipleChoice",
        "YosoForQuestionAnswering",
        "YosoForSequenceClassification",
        "YosoForTokenClassification",
        "YosoLayer",
        "YosoModel",
        "YosoPreTrainedModel",
    ]

# 如果是类型检查环境，则导入一些类型相关的模块
if TYPE_CHECKING:
    from .configuration_yoso import YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig  # 导入 YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig 类

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从 modeling_yoso 模块中导入一些类
        from .modeling_yoso import (
            YOSO_PRETRAINED_MODEL_ARCHIVE_LIST,
            YosoForMaskedLM,
            YosoForMultipleChoice,
            YosoForQuestionAnswering,
            YosoForSequenceClassification,
            YosoForTokenClassification,
            YosoLayer,
            YosoModel,
            YosoPreTrainedModel,
        )

# 如果不是类型检查环境，则将当前模块替换为 _LazyModule 实例，以实现惰性加载
else:
    import sys  # 导入 sys 模块

    # 将当前模块替换为 _LazyModule 实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```  
```