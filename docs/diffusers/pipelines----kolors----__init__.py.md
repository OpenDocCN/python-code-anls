# `.\diffusers\pipelines\kolors\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，以支持类型检查
from typing import TYPE_CHECKING

# 从上层模块导入一系列工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 表示是否需要慢速导入
    OptionalDependencyNotAvailable,  # 当可选依赖不可用时引发的异常
    _LazyModule,  # 延迟加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具函数
    is_sentencepiece_available,  # 检查 SentencePiece 是否可用
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 初始化一个空字典，用于存放虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存放导入结构
_import_structure = {}

# 尝试检查可选依赖是否可用
try:
    # 如果 Transformers 和 PyTorch 不可用，但 SentencePiece 可用，抛出异常
    if not (is_transformers_available() and is_torch_available()) and is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入虚拟对象，避免依赖错误
    from ...utils import dummy_torch_and_transformers_and_sentencepiece_objects  # noqa F403

    # 更新虚拟对象字典，获取 dummy 对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_sentencepiece_objects))
else:
    # 如果依赖可用，更新导入结构，包含相应的管道和模型
    _import_structure["pipeline_kolors"] = ["KolorsPipeline"]
    _import_structure["pipeline_kolors_img2img"] = ["KolorsImg2ImgPipeline"]
    _import_structure["text_encoder"] = ["ChatGLMModel"]
    _import_structure["tokenizer"] = ["ChatGLMTokenizer"]

# 如果在类型检查中或需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 再次检查可选依赖是否可用
        if not (is_transformers_available() and is_torch_available()) and is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从工具模块导入所有虚拟对象
        from ...utils.dummy_torch_and_transformers_and_sentencepiece_objects import *

    else:
        # 导入实际的管道和模型
        from .pipeline_kolors import KolorsPipeline
        from .pipeline_kolors_img2img import KolorsImg2ImgPipeline
        from .text_encoder import ChatGLMModel
        from .tokenizer import ChatGLMTokenizer

else:
    # 如果不在类型检查中，执行懒加载模块的操作
    import sys

    # 使用懒加载模块初始化当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名
        globals()["__file__"],  # 当前文件名
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```