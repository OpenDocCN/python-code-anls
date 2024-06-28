# `.\models\gpt_sw3\__init__.py`

```
# 版权声明和许可证信息，指明代码版权归 The HuggingFace Team 所有，使用 Apache License 2.0 许可
#
# 导入必要的模块和函数声明
from typing import TYPE_CHECKING

# 导入可选依赖相关的异常和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义一个空的导入结构
_import_structure = {}

# 尝试检查是否存在 SentencePiece 模块，如果不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 GPTSw3Tokenizer 添加到导入结构中
    _import_structure["tokenization_gpt_sw3"] = ["GPTSw3Tokenizer"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    try:
        # 再次检查是否存在 SentencePiece 模块，如果不存在则引发异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 tokenization_gpt_sw3 模块中导入 GPTSw3Tokenizer 类
        from .tokenization_gpt_sw3 import GPTSw3Tokenizer

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块定义为 _LazyModule 的延迟加载模块，使用当前模块的名称和文件路径
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```