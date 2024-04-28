# `.\models\fsmt\__init__.py`

```
# 版权声明和许可信息
# 从类型提示中导入 TYPE_CHECKING
from typing import TYPE_CHECKING
# 从 utils 模块中导入 OptionalDependencyNotAvailable, _LazyModule, is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义 _import_structure 字典，包含模块名称和对应的导入列表
_import_structure = {
    # 配置模块 fsmt 的导入信息
    "configuration_fsmt": ["FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FSMTConfig"],
    # 分词模块 fsmt 的导入信息
    "tokenization_fsmt": ["FSMTTokenizer"],
}

# 尝试导入 torch，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 若导入成功，则将建立模型模块 fsmt 的导入信息
else:
    _import_structure["modeling_fsmt"] = ["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"]

# 若类型检查为真，则进行类型检查相关的导入操作
if TYPE_CHECKING:
    # 从配置模块 fsmt 中导入指定内容
    from .configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
    # 从分词模块 fsmt 中导入指定内容
    from .tokenization_fsmt import FSMTTokenizer

    # 尝试导入 torch，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 若导入成功，则从模型模块 fsmt 中导入指定内容
    else:
        from .modeling_fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel

# 若不是类型检查，则进行懒加载模块导入的操作
else:
    import sys
    # 将当前模块设为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```