# `.\models\bartpho\__init__.py`

```
# 版权声明和许可信息，声明版权归 HuggingFace 团队所有，授权遵循 Apache License 2.0
#
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块，包括自定义的异常 OptionalDependencyNotAvailable，_LazyModule，以及检查 SentencePiece 是否可用的函数 is_sentencepiece_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义一个空的导入结构字典
_import_structure = {}

# 尝试检查 SentencePiece 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果检测到 OptionalDependencyNotAvailable 异常，则忽略并继续执行
    pass
else:
    # 如果 SentencePiece 可用，则添加 BartphoTokenizer 到导入结构字典
    _import_structure["tokenization_bartpho"] = ["BartphoTokenizer"]

# 如果正在进行类型检查（Type Checking）
if TYPE_CHECKING:
    try:
        # 再次检查 SentencePiece 是否可用
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果不可用，则忽略异常
        pass
    else:
        # 如果可用，从 tokenization_bartpho 模块中导入 BartphoTokenizer 类
        from .tokenization_bartpho import BartphoTokenizer

# 如果不是在类型检查模式下运行
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule，延迟加载导入结构字典中的内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```