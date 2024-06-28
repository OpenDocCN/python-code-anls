# `.\models\rwkv\__init__.py`

```
# 版权声明和许可证信息，指出此代码版权归HuggingFace团队所有，并遵循Apache License, Version 2.0。
#
# 如果不满足许可证的要求，禁止使用此文件。可以从以下链接获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

from typing import TYPE_CHECKING

# 从utils模块中导入所需的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义了模块的导入结构
_import_structure = {
    "configuration_rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig", "RwkvOnnxConfig"],
}

# 检查是否有torch库可用，如果不可用，则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，将modeling_rwkv模块添加到导入结构中
    _import_structure["modeling_rwkv"] = [
        "RWKV_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RwkvForCausalLM",
        "RwkvModel",
        "RwkvPreTrainedModel",
    ]

# 如果当前是类型检查阶段，导入所需的类型定义
if TYPE_CHECKING:
    from .configuration_rwkv import RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP, RwkvConfig, RwkvOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv import (
            RWKV_PRETRAINED_MODEL_ARCHIVE_LIST,
            RwkvForCausalLM,
            RwkvModel,
            RwkvPreTrainedModel,
        )
# 如果不是类型检查阶段，则在sys.modules中注册一个LazyModule
else:
    import sys

    # 使用_LazyModule类将当前模块注册到sys.modules中，以实现惰性加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```