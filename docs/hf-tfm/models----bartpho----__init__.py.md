# `.\transformers\models\bartpho\__init__.py`

```
# 导入 TYPE_CHECKING，用于在类型检查时使用
from typing import TYPE_CHECKING
# 导入自定义的异常 OptionalDependencyNotAvailable，用于处理可选依赖未安装的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available

# 定义一个空的导入结构字典
_import_structure = {}

# 尝试检查是否 sentencepiece 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
# 捕获 OptionalDependencyNotAvailable 异常
except OptionalDependencyNotAvailable:
    pass
# 若 sentencepiece 可用，则执行以下代码块
else:
    # 将 BartphoTokenizer 添加到导入结构字典中
    _import_structure["tokenization_bartpho"] = ["BartphoTokenizer"]

# 如果在类型检查时
if TYPE_CHECKING:
    # 尝试检查是否 sentencepiece 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 tokenization_bartpho 模块导入 BartphoTokenizer 类
        from .tokenization_bartpho import BartphoTokenizer

# 如果不在类型检查时
else:
    # 导入 sys 模块
    import sys

    # 将当前模块的名称注册为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```