# `.\transformers\models\realm\__init__.py`

```py
# 版权声明和许可证信息
# 此文件版权归 2022 年 HuggingFace 团队所有，保留所有权利
# 根据 Apache 许可证版本 2.0 许可
# 只有在符合许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或以书面形式同意，软件按“原样”分发
# 没有任何种类的担保或条件，无论是明示的还是暗示的
# 有关权限限制和许可证的详细信息，请参见许可证
from typing import TYPE_CHECKING  # 引入类型检查

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available  # 导入必要的依赖项

# 声明模块的导入结构
_import_structure = {
    "configuration_realm": ["REALM_PRETRAINED_CONFIG_ARCHIVE_MAP", "RealmConfig"],  # 配置领域
    "tokenization_realm": ["RealmTokenizer"],  # 令牌化领域
}

# 检查是否存在 tokenizers 库，若不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 未找到 tokenizers 库

else:
    _import_structure["tokenization_realm_fast"] = ["RealmTokenizerFast"]  # 快速令牌化领域

# 检查是否存在 torch 库，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 未找到 torch 库

else:
    # 设置模型领域的导入结构
    _import_structure["modeling_realm"] = [
        "REALM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RealmEmbedder",
        "RealmForOpenQA",
        "RealmKnowledgeAugEncoder",
        "RealmPreTrainedModel",
        "RealmReader",
        "RealmScorer",
        "load_tf_weights_in_realm",
    ]
    _import_structure["retrieval_realm"] = ["RealmRetriever"]  # 检索领域

# 如果为类型检查模式，则进一步导入
if TYPE_CHECKING:
    from .configuration_realm import REALM_PRETRAINED_CONFIG_ARCHIVE_MAP, RealmConfig
    from .tokenization_realm import RealmTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_realm import RealmTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_realm import (
            REALM_PRETRAINED_MODEL_ARCHIVE_LIST,
            RealmEmbedder,
            RealmForOpenQA,
            RealmKnowledgeAugEncoder,
            RealmPreTrainedModel,
            RealmReader,
            RealmScorer,
            load_tf_weights_in_realm,
        )
        from .retrieval_realm import RealmRetriever

# 如果不是类型检查模式，则设置懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```