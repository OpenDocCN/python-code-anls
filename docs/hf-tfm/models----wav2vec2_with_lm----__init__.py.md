# `.\transformers\models\wav2vec2_with_lm\__init__.py`

```
# 版权声明
# 版权所有 2021 年 HuggingFace 团队保留所有权利。
# 根据 Apache 许可证第 2.0 版授权;
# 您可能不得使用本文件，除非遵守许可证。
# 您可以在以下网址获取许可证副本:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得根据此许可证分发软件是基于“原样”基础，
# 没有任何形式的保证或条件，不论是明示的或默示的。
# 请见许可证，了解特定语言管理权限和
# 许可证下的限制
from typing import TYPE_CHECKING

from ...utils import _LazyModule

# 导入结构定义
_import_structure = {"processing_wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"]}

# 如果是类型检查，则导入 Wav2Vec2ProcessorWithLM 类型
if TYPE_CHECKING:
    from .processing_wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
# 如果不是类型检查，则导入 sys 模块，将当前模块设为 _LazyModule
else:
    import sys

    # 将当前模块设为延迟加载模块，并传入模块结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```