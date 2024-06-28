# `.\models\codegen\__init__.py`

```
# 版权声明及许可信息，指出本代码的所有权和使用许可
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证，否则禁止使用本文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得以任何形式分发本软件
# 本软件基于"按原样"提供，没有任何明示或暗示的担保或条件
# 请参阅许可证，了解详细的法律条文和限制条件
from typing import TYPE_CHECKING

# 导入所需模块和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构，包含各个子模块和类的映射关系
_import_structure = {
    "configuration_codegen": ["CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP", "CodeGenConfig", "CodeGenOnnxConfig"],
    "tokenization_codegen": ["CodeGenTokenizer"],
}

# 尝试导入 tokenizers，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入成功，添加对应的快速 tokenization_codegen_fast 模块
    _import_structure["tokenization_codegen_fast"] = ["CodeGenTokenizerFast"]

# 尝试导入 torch，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入成功，添加 modeling_codegen 模块及其内容
    _import_structure["modeling_codegen"] = [
        "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CodeGenForCausalLM",
        "CodeGenModel",
        "CodeGenPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置、tokenizer 及模型相关的类和映射
    from .configuration_codegen import CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP, CodeGenConfig, CodeGenOnnxConfig
    from .tokenization_codegen import CodeGenTokenizer

    # 尝试导入 tokenizers，若不可用则不进行导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若导入成功，导入 tokenization_codegen_fast 模块中的类
        from .tokenization_codegen_fast import CodeGenTokenizerFast

    # 尝试导入 torch，若不可用则不进行导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若导入成功，导入 modeling_codegen 模块中的类
        from .modeling_codegen import (
            CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
            CodeGenForCausalLM,
            CodeGenModel,
            CodeGenPreTrainedModel,
        )

# 非类型检查模式下，使用 LazyModule 实现懒加载模块
else:
    import sys

    # 将当前模块设置为 LazyModule，以便按需加载各个子模块和类
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```