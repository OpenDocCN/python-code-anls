# `.\transformers\models\big_bird\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入必要的工具函数和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig", "BigBirdOnnxConfig"],
}

# 检查是否有 SentencePiece 可用，若无则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加相应的 tokenization_big_bird 模块到导入结构中
    _import_structure["tokenization_big_bird"] = ["BigBirdTokenizer"]

# 检查是否有 Tokenizers 可用，若无则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加相应的 tokenization_big_bird_fast 模块到导入结构中
    _import_structure["tokenization_big_bird_fast"] = ["BigBirdTokenizerFast"]

# 检查是否有 PyTorch 可用，若无则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加相应的 modeling_big_bird 模块到导入结构中
    _import_structure["modeling_big_bird"] = [
        "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BigBirdForCausalLM",
        "BigBirdForMaskedLM",
        "BigBirdForMultipleChoice",
        "BigBirdForPreTraining",
        "BigBirdForQuestionAnswering",
        "BigBirdForSequenceClassification",
        "BigBirdForTokenClassification",
        "BigBirdLayer",
        "BigBirdModel",
        "BigBirdPreTrainedModel",
        "load_tf_weights_in_big_bird",
    ]

# 检查是否有 Flax 可用，若无则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加相应的 modeling_flax_big_bird 模块到导入结构中
    _import_structure["modeling_flax_big_bird"] = [
        "FlaxBigBirdForCausalLM",
        "FlaxBigBirdForMaskedLM",
        "FlaxBigBirdForMultipleChoice",
        "FlaxBigBirdForPreTraining",
        "FlaxBigBirdForQuestionAnswering",
        "FlaxBigBirdForSequenceClassification",
        "FlaxBigBirdForTokenClassification",
        "FlaxBigBirdModel",
        "FlaxBigBirdPreTrainedModel",
    ]

# 如果是类型检查模式，则进行额外的导入
if TYPE_CHECKING:
    # 引入配置相关的内容
    from .configuration_big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig, BigBirdOnnxConfig

    # 检查是否有 SentencePiece 可用，若无则抛出异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入相应的 tokenization_big_bird 模块
        from .tokenization_big_bird import BigBirdTokenizer
    # 检查 tokenizers 库是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行后续代码
        pass
    # 若未引发异常，则表示 tokenizers 库可用
    else:
        # 导入 BigBirdTokenizerFast 类
        from .tokenization_big_bird_fast import BigBirdTokenizerFast

    # 检查 torch 库是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行后续代码
        pass
    # 若未引发异常，则表示 torch 库可用
    else:
        # 导入 BigBird 模型相关类和函数
        from .modeling_big_bird import (
            BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigBirdForCausalLM,
            BigBirdForMaskedLM,
            BigBirdForMultipleChoice,
            BigBirdForPreTraining,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
            BigBirdLayer,
            BigBirdModel,
            BigBirdPreTrainedModel,
            load_tf_weights_in_big_bird,
        )

    # 检查 flax 库是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行后续代码
        pass
    # 若未引发异常，则表示 flax 库可用
    else:
        # 导入 FlaxBigBird 模型相关类
        from .modeling_flax_big_bird import (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
            FlaxBigBirdModel,
            FlaxBigBirdPreTrainedModel,
        )
# 如果前面的条件不成立，则执行以下操作
else:
    # 导入 sys 模块，用于系统相关操作
    import sys
    # 使用 sys.modules 字典，将当前模块替换为一个懒加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```