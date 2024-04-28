# `.\transformers\models\nllb\__init__.py`

```
# 引入必要的类型检查模块
from typing import TYPE_CHECKING
# 引入可选依赖未安装异常模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构字典
_import_structure = {}

# 尝试检查并导入句子拆分模块
try:
    # 如果句子拆分模块不可用，则引发可选依赖未安装异常
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未安装异常
except OptionalDependencyNotAvailable:
    pass
# 如果没有异常，则执行以下操作
else:
    # 在导入结构字典中添加对应的模块和类
    _import_structure["tokenization_nllb"] = ["NllbTokenizer"]

# 尝试检查并导入分词器模块
try:
    # 如果分词器模块不可用，则引发可选依赖未安装异常
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未安装异常
except OptionalDependencyNotAvailable:
    pass
# 如果没有异常，则执行以下操作
else:
    # 在导入结构字典中添加对应的模块和类
    _import_structure["tokenization_nllb_fast"] = ["NllbTokenizerFast"]

# 如果是类型检查模式
if TYPE_CHECKING:
    try:
        # 如果句子拆分模块不可用，则引发可选依赖未安装异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖未安装异常
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入句子拆分模块中的类
        from .tokenization_nllb import NllbTokenizer

    try:
        # 如果分词器模块不可用，则引发可选依赖未安装异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖未安装异常
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入分词器模块中的类
        from .tokenization_nllb_fast import NllbTokenizerFast

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys
    # 将当前模块替换为 LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```