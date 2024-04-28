# `.\transformers\utils\bitsandbytes.py`

```
# 版权声明及许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入警告模块
import warnings

# 发出未来警告，提示 transformers.utils.bitsandbytes 模块已弃用，并将在将来版本中移除
warnings.warn(
    "transformers.utils.bitsandbytes module is deprecated and will be removed in a future version. Please import bitsandbytes modules directly from transformers.integrations",
    FutureWarning,
)

# 从 transformers.integrations 导入以下模块
from ..integrations import (  # noqa
    get_keys_to_not_convert,
    replace_8bit_linear,
    replace_with_bnb_linear,
    set_module_8bit_tensor_to_device,
    set_module_quantized_tensor_to_device,
)
```