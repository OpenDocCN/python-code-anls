# `.\models\big_bird\__init__.py`

```py
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入相关的依赖项和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包含需要导入的模块和对应的标识符
_import_structure = {
    "configuration_big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig", "BigBirdOnnxConfig"],
}

# 检查是否可用句子分割模块，若不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_big_bird 模块到导入结构中
    _import_structure["tokenization_big_bird"] = ["BigBirdTokenizer"]

# 检查是否可用 Tokenizers 库，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_big_bird_fast 模块到导入结构中
    _import_structure["tokenization_big_bird_fast"] = ["BigBirdTokenizerFast"]

# 检查是否可用 Torch 库，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_big_bird 模块到导入结构中
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

# 检查是否可用 Flax 库，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_flax_big_bird 模块到导入结构中
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

# 如果在类型检查模式下，导入额外的配置和模块
if TYPE_CHECKING:
    # 导入具体的配置、模型和标识符
    from .configuration_big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig, BigBirdOnnxConfig

    # 再次检查句子分割模块是否可用，若不可用则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 tokenization_big_bird 模块导入 BigBirdTokenizer 类
        from .tokenization_big_bird import BigBirdTokenizer
    # 尝试检查是否安装了 tokenizers 库，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 tokenizers 库不可用
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行
        pass
    else:
        # 若 tokenizers 可用，则从本地目录导入 BigBirdTokenizerFast
        from .tokenization_big_bird_fast import BigBirdTokenizerFast

    # 尝试检查是否安装了 torch 库，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 torch 库不可用
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行
        pass
    else:
        # 若 torch 可用，则从本地目录导入以下模块：
        # BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST,
        # BigBirdForCausalLM,
        # BigBirdForMaskedLM,
        # BigBirdForMultipleChoice,
        # BigBirdForPreTraining,
        # BigBirdForQuestionAnswering,
        # BigBirdForSequenceClassification,
        # BigBirdForTokenClassification,
        # BigBirdLayer,
        # BigBirdModel,
        # BigBirdPreTrainedModel,
        # load_tf_weights_in_big_bird
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

    # 尝试检查是否安装了 flax 库，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 flax 库不可用
    except OptionalDependencyNotAvailable:
        # 忽略异常，继续执行
        pass
    else:
        # 若 flax 可用，则从本地目录导入以下模块：
        # FlaxBigBirdForCausalLM,
        # FlaxBigBirdForMaskedLM,
        # FlaxBigBirdForMultipleChoice,
        # FlaxBigBirdForPreTraining,
        # FlaxBigBirdForQuestionAnswering,
        # FlaxBigBirdForSequenceClassification,
        # FlaxBigBirdForTokenClassification,
        # FlaxBigBirdModel,
        # FlaxBigBirdPreTrainedModel
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
# 如果上述条件不满足，即导入模块失败，则执行以下操作
else:
    # 导入 sys 模块
    import sys
    # 将当前模块更新到 sys.modules 中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```