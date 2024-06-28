# `.\models\mamba\__init__.py`

```
# 版权声明和许可证信息
# 版权 2024 年 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 不提供任何形式的担保或条件，无论是明示的还是默示的。
# 有关特定语言的权限，请参阅许可证。

# 导入所需的类型检查模块
from typing import TYPE_CHECKING

# 导入工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_mamba": ["MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MambaConfig", "MambaOnnxConfig"],
}

# 检查是否可以导入 Torch，如果不能则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果能导入 Torch，则添加以下模块到导入结构中
    _import_structure["modeling_mamba"] = [
        "MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MambaForCausalLM",
        "MambaModel",
        "MambaPreTrainedModel",
    ]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的类型
    from .configuration_mamba import MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP, MambaConfig, MambaOnnxConfig

    # 再次检查 Torch 是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入建模相关的类型
        from .modeling_mamba import (
            MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MambaForCausalLM,
            MambaModel,
            MambaPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块用于注册当前模块
    import sys

    # 将当前模块替换为延迟加载模块 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```