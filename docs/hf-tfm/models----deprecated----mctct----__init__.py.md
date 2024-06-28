# `.\models\deprecated\mctct\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常类和模块惰性加载工具函数
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构字典
_import_structure = {
    "configuration_mctct": ["MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MCTCTConfig"],
    "feature_extraction_mctct": ["MCTCTFeatureExtractor"],
    "processing_mctct": ["MCTCTProcessor"],
}

# 检查是否存在 torch 库，若不存在则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加模型相关的导入内容到结构字典
    _import_structure["modeling_mctct"] = [
        "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MCTCTForCTC",
        "MCTCTModel",
        "MCTCTPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的类和变量
    from .configuration_mctct import MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP, MCTCTConfig
    # 导入特征提取相关的类
    from .feature_extraction_mctct import MCTCTFeatureExtractor
    # 导入处理相关的类
    from .processing_mctct import MCTCTProcessor

    # 再次检查 torch 库是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和变量
        from .modeling_mctct import MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST, MCTCTForCTC, MCTCTModel, MCTCTPreTrainedModel

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块替换为懒加载模块对象，支持按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```