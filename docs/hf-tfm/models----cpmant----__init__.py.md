# `.\models\cpmant\__init__.py`

```
# 禁止 flake8 对当前模块进行检查
# 没有办法在此模块中忽略 "F401 '...' imported but unused" 警告，除非放弃其他警告。所以完全不检查此模块。

# 版权声明
# 2022年，HuggingFace 团队和 OpenBMB 团队保留所有权利
#
# 根据 Apache 许可证，版本 2.0 (the "License") 授权
# 你不得使用本文件，除非遵守该许可证
# 你可以在以下网址获取该许可证的一份副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则以"原样"分发软件
# 无任何形式的担保或条件，明示或暗示
# 请参阅许可证了解详细的授权规定和限制
from typing import TYPE_CHECKING

# 依赖 isort 来合并导入
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available


# 导入结构
_import_structure = {
    "configuration_cpmant": ["CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CpmAntConfig"],
    "tokenization_cpmant": ["CpmAntTokenizer"],
}


# 尝试导入 torch，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_cpmant"] = [
        "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CpmAntForCausalLM",
        "CpmAntModel",
        "CpmAntPreTrainedModel",
    ]


# 如果类型检查为真
if TYPE_CHECKING:
    from .configuration_cpmant import CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP, CpmAntConfig
    from .tokenization_cpmant import CpmAntTokenizer

    # 尝试导入 torch，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
        from .modeling_cpmant import (
            CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CpmAntForCausalLM,
            CpmAntModel,
            CpmAntPreTrainedModel,
        )


# 如果类型检查为假
else:
    # 导入 sys
    import sys

    # 使用 _LazyModule 创建模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```