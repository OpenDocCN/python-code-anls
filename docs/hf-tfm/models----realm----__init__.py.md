# `.\models\realm\__init__.py`

```py
# 版权声明和许可声明，说明该文件的版权归 HuggingFace 团队所有，使用 Apache License 2.0 进行许可
#
# 如果不符合许可协议的规定，除非法律另有要求或书面同意，否则不得使用此文件
from typing import TYPE_CHECKING

# 从 utils 模块中导入 OptionalDependencyNotAvailable 异常类和 LazyModule 类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义要导入的模块结构，包括配置、tokenization、modeling 和 retrieval 的相关内容
_import_structure = {
    "configuration_realm": ["REALM_PRETRAINED_CONFIG_ARCHIVE_MAP", "RealmConfig"],
    "tokenization_realm": ["RealmTokenizer"],
}

# 检查是否有 tokenizers 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 "tokenization_realm_fast" 到导入结构中，包含 "RealmTokenizerFast"
    _import_structure["tokenization_realm_fast"] = ["RealmTokenizerFast"]

# 检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 "modeling_realm" 和 "retrieval_realm" 到导入结构中，包含相关类和函数
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
    _import_structure["retrieval_realm"] = ["RealmRetriever"]

# 如果是类型检查模式，则从相应模块中导入所需内容
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

# 如果不是类型检查模式，则将当前模块设置为 LazyModule 的实例，导入 _import_structure 中定义的内容
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```