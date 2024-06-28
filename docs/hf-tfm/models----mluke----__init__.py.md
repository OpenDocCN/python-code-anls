# `.\models\mluke\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义的异常类和延迟加载模块的工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义一个空的导入结构字典
_import_structure = {}

# 尝试检测是否存在 SentencePiece 库，如果不存在则引发自定义的异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果依赖项不可用，则忽略异常继续执行
    pass
else:
    # 如果依赖项可用，则将 MLukeTokenizer 添加到导入结构中
    _import_structure["tokenization_mluke"] = ["MLukeTokenizer"]

# 如果类型检查开启
if TYPE_CHECKING:
    try:
        # 再次检测是否存在 SentencePiece 库，如果不存在则引发自定义的异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖项不可用，则忽略异常继续执行
        pass
    else:
        # 如果依赖项可用，则从 tokenization_mluke 模块导入 MLukeTokenizer 类
        from .tokenization_mluke import MLukeTokenizer

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为一个延迟加载模块，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```