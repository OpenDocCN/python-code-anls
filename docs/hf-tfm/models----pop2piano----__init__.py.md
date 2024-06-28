# `.\models\pop2piano\__init__.py`

```py
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入依赖项检查函数和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_torch_available,
)

# 定义导入结构字典，用于组织不同模块的导入
_import_structure = {
    "configuration_pop2piano": ["POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Pop2PianoConfig"],
}

# 检查是否存在 torch 库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 存在，则添加相关模块到导入结构字典
    _import_structure["modeling_pop2piano"] = [
        "POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Pop2PianoForConditionalGeneration",
        "Pop2PianoPreTrainedModel",
    ]

# 检查是否存在 librosa、essentia、scipy 和 torch 库，若有任一依赖项缺失则抛出异常
try:
    if not (is_librosa_available() and is_essentia_available() and is_scipy_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若所有依赖项都存在，则添加相关模块到导入结构字典
    _import_structure["feature_extraction_pop2piano"] = ["Pop2PianoFeatureExtractor"]

# 检查是否存在 pretty_midi 和 torch 库，若任一依赖项缺失则抛出异常
try:
    if not (is_pretty_midi_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若依赖项都存在，则添加相关模块到导入结构字典
    _import_structure["tokenization_pop2piano"] = ["Pop2PianoTokenizer"]

# 检查是否存在 pretty_midi、torch、librosa、essentia 和 scipy 库，若有任一依赖项缺失则抛出异常
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
    # 若所有依赖项都存在，则添加相关模块到导入结构字典
    _import_structure["processing_pop2piano"] = ["Pop2PianoProcessor"]

# 如果在类型检查模式下，则从相应模块导入所需的类和变量
if TYPE_CHECKING:
    from .configuration_pop2piano import POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP, Pop2PianoConfig

    # 检查是否存在 torch 库，若不存在则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 torch 存在，则从 modeling_pop2piano 模块中导入相关类和变量
        from .modeling_pop2piano import (
            POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Pop2PianoForConditionalGeneration,
            Pop2PianoPreTrainedModel,
        )

    # 检查是否存在 librosa、essentia、scipy 和 torch 库，若有任一依赖项缺失则抛出异常
    try:
        if not (is_librosa_available() and is_essentia_available() and is_scipy_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果前面的导入失败，则尝试从本地模块导入 Pop2PianoFeatureExtractor
        from .feature_extraction_pop2piano import Pop2PianoFeatureExtractor

    try:
        # 检查必要的依赖是否都可用，否则抛出 OptionalDependencyNotAvailable 异常
        if not (is_pretty_midi_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则不进行后续操作
        pass
    else:
        # 如果依赖可用，则从本地模块导入 Pop2PianoTokenizer
        from .tokenization_pop2piano import Pop2PianoTokenizer

    try:
        # 检查多个依赖是否都可用，否则抛出 OptionalDependencyNotAvailable 异常
        if not (
            is_pretty_midi_available()
            and is_torch_available()
            and is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则不进行后续操作
        pass
    else:
        # 如果依赖可用，则从本地模块导入 Pop2PianoProcessor
        from .processing_pop2piano import Pop2PianoProcessor
else:
    # 导入 sys 模块，用于操作解释器相关的功能
    import sys

    # 将当前模块（__name__）的引用指向 _LazyModule 类的实例，这是一种惰性加载模块的方法
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```