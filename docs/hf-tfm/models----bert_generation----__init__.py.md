# `.\transformers\models\bert_generation\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_bert_generation": ["BertGenerationConfig"]}

# 检查是否存在 sentencepiece 库，如果不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 sentencepiece 库，则添加 tokenization_bert_generation 到导入结构中
    _import_structure["tokenization_bert_generation"] = ["BertGenerationTokenizer"]

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加 modeling_bert_generation 到导入结构中
    _import_structure["modeling_bert_generation"] = [
        "BertGenerationDecoder",
        "BertGenerationEncoder",
        "BertGenerationPreTrainedModel",
        "load_tf_weights_in_bert_generation",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入 BertGenerationConfig 类
    from .configuration_bert_generation import BertGenerationConfig

    # 检查是否存在 sentencepiece 库，如果不存在则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 BertGenerationTokenizer 类
        from .tokenization_bert_generation import BertGenerationTokenizer

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关类和函数
        from .modeling_bert_generation import (
            BertGenerationDecoder,
            BertGenerationEncoder,
            BertGenerationPreTrainedModel,
            load_tf_weights_in_bert_generation,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```