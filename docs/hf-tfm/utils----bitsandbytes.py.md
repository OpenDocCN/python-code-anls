# `.\utils\bitsandbytes.py`

```py
# 引入警告模块，用于发出警告信息
import warnings

# 发出一个将来版本移除模块的警告，提醒用户更新相关导入语句
warnings.warn(
    "transformers.utils.bitsandbytes module is deprecated and will be removed in a future version. Please import bitsandbytes modules directly from transformers.integrations",
    FutureWarning,
)

# 从 integrations 模块中导入以下函数和类，其中 get_keys_to_not_convert, replace_8bit_linear, replace_with_bnb_linear,
# set_module_8bit_tensor_to_device, set_module_quantized_tensor_to_device 被导入，忽略导入时的警告
from ..integrations import (  # noqa
    get_keys_to_not_convert,
    replace_8bit_linear,
    replace_with_bnb_linear,
    set_module_8bit_tensor_to_device,
    set_module_quantized_tensor_to_device,
)
```