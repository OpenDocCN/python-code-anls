# `.\transformers\models\pop2piano\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的依赖项和实用函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_pop2piano": ["POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Pop2PianoConfig"],
}

# 检查是否导入了torch模块，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了torch模块，则添加以下模块到导入结构中
    _import_structure["modeling_pop2piano"] = [
        "POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Pop2PianoForConditionalGeneration",
        "Pop2PianoPreTrainedModel",
    ]

# 检查是否导入了librosa、essentia、scipy和torch模块，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not (is_librosa_available() and is_essentia_available() and is_scipy_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了上述模块，则添加以下模块到导入结构中
    _import_structure["feature_extraction_pop2piano"] = ["Pop2PianoFeatureExtractor"]

# 检查是否导入了pretty_midi和torch模块，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not (is_pretty_midi_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了上述模块，则添加以下模块到导入结构中
    _import_structure["tokenization_pop2piano"] = ["Pop2PianoTokenizer"]

# 检查是否导入了pretty_midi、torch、librosa、essentia和scipy模块，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not (
        is_pretty_midi_available()
        and is_torch_available()
        and is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了上述模块，则添加以下模块到导入结构中
    _import_structure["processing_pop2piano"] = ["Pop2PianoProcessor"]

# 如果类型检查模块可用，则导入配置和模型相关模块
if TYPE_CHECKING:
    from .configuration_pop2piano import POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP, Pop2PianoConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pop2piano import (
            POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Pop2PianoForConditionalGeneration,
            Pop2PianoPreTrainedModel,
        )

    try:
        if not (is_librosa_available() and is_essentia_available() and is_scipy_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件为真，则导入 Pop2PianoFeatureExtractor 模块
    else:
        from .feature_extraction_pop2piano import Pop2PianoFeatureExtractor

    # 尝试检查是否 pretty_midi 和 torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not (is_pretty_midi_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若发生异常则什么也不做
        pass
    # 若无异常发生，则导入 Pop2PianoTokenizer 模块
    else:
        from .tokenization_pop2piano import Pop2PianoTokenizer

    # 尝试检查是否 pretty_midi、torch、librosa、essentia、scipy 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not (
            is_pretty_midi_available()
            and is_torch_available()
            and is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若发生异常则什么也不做
        pass
    # 若无异常发生，则导入 Pop2PianoProcessor 模块
    else:
        from .processing_pop2piano import Pop2PianoProcessor
# 如果从该模块进行导入时，当前环境没有定义 _LazyModule 模块
else:
    # 导入 sys 模块
    import sys
    # 将当前模块的 __name__ 属性（即模块名）作为 __name__ 参数，
    # 将 globals()["__file__"] 作为 __file__ 参数，
    # 将 _import_structure 作为 _import_structure 参数，
    # 将 __spec__ 作为 module_spec 参数，创建一个 _LazyModule 对象
    # 并将其设置为当前模块的 __module__ 属性
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```