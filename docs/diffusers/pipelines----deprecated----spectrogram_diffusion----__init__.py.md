# `.\diffusers\pipelines\deprecated\spectrogram_diffusion\__init__.py`

```py
# flake8: noqa  # 忽略 flake8 的检查
from typing import TYPE_CHECKING  # 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from ....utils import (  # 从 utils 模块导入必要的工具函数和常量
    DIFFUSERS_SLOW_IMPORT,  # 导入 DIFFUSERS_SLOW_IMPORT 常量
    _LazyModule,  # 导入 _LazyModule 类
    is_note_seq_available,  # 导入检测 note_seq 可用性的函数
    OptionalDependencyNotAvailable,  # 导入表示可选依赖不可用的异常
    is_torch_available,  # 导入检测 PyTorch 可用性的函数
    is_transformers_available,  # 导入检测 transformers 可用性的函数
    get_objects_from_module,  # 导入从模块获取对象的函数
)

_dummy_objects = {}  # 初始化一个空字典，用于存储虚拟对象
_import_structure = {}  # 初始化一个空字典，用于存储导入结构

try:  # 尝试执行以下代码块
    if not (is_transformers_available() and is_torch_available()):  # 检查 transformers 和 torch 是否可用
        raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常
except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
    from ....utils import dummy_torch_and_transformers_objects  # 导入虚拟对象模块

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))  # 更新虚拟对象字典
else:  # 如果没有异常，执行以下代码
    _import_structure["continous_encoder"] = ["SpectrogramContEncoder"]  # 添加连续编码器到导入结构
    _import_structure["notes_encoder"] = ["SpectrogramNotesEncoder"]  # 添加音符编码器到导入结构
    _import_structure["pipeline_spectrogram_diffusion"] = [  # 添加谱图扩散管道到导入结构
        "SpectrogramContEncoder",  # 添加连续编码器
        "SpectrogramDiffusionPipeline",  # 添加谱图扩散管道
        "T5FilmDecoder",  # 添加 T5 电影解码器
    ]
try:  # 再次尝试执行代码块
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):  # 检查所有三个依赖项
        raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常
except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
    from ....utils import dummy_transformers_and_torch_and_note_seq_objects  # 导入所有虚拟对象模块

    _dummy_objects.update(get_objects_from_module(dummy_transformers_and_torch_and_note_seq_objects))  # 更新虚拟对象字典
else:  # 如果没有异常，执行以下代码
    _import_structure["midi_utils"] = ["MidiProcessor"]  # 添加 MIDI 工具到导入结构


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:  # 如果正在类型检查或慢导入标志为真
    try:  # 尝试执行以下代码块
        if not (is_transformers_available() and is_torch_available()):  # 检查 transformers 和 torch 是否可用
            raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常

    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from ....utils.dummy_torch_and_transformers_objects import *  # 导入所有虚拟对象
    else:  # 如果没有异常，执行以下代码
        from .pipeline_spectrogram_diffusion import SpectrogramDiffusionPipeline  # 从谱图扩散管道导入 SpectrogramDiffusionPipeline
        from .pipeline_spectrogram_diffusion import SpectrogramContEncoder  # 从谱图扩散管道导入 SpectrogramContEncoder
        from .pipeline_spectrogram_diffusion import SpectrogramNotesEncoder  # 从谱图扩散管道导入 SpectrogramNotesEncoder
        from .pipeline_spectrogram_diffusion import T5FilmDecoder  # 从谱图扩散管道导入 T5FilmDecoder

    try:  # 再次尝试执行代码块
        if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):  # 检查所有三个依赖项
            raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from ....utils.dummy_transformers_and_torch_and_note_seq_objects import *  # 导入所有虚拟对象

    else:  # 如果没有异常，执行以下代码
        from .midi_utils import MidiProcessor  # 从 MIDI 工具模块导入 MidiProcessor

else:  # 如果不是类型检查且慢导入标志为假
    import sys  # 导入 sys 模块

    sys.modules[__name__] = _LazyModule(  # 将当前模块替换为懒加载模块
        __name__,  # 模块名称
        globals()["__file__"],  # 模块文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )

    for name, value in _dummy_objects.items():  # 遍历虚拟对象字典
        setattr(sys.modules[__name__], name, value)  # 将虚拟对象设置到当前模块
```