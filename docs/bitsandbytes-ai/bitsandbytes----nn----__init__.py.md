# `bitsandbytes\bitsandbytes\nn\__init__.py`

```py
# 版权声明，声明代码使用 MIT 许可证
#
# 导入所需的模块
from .modules import (
    Embedding,  # 导入 Embedding 模块
    Int8Params,  # 导入 Int8Params 模块
    Linear4bit,  # 导入 Linear4bit 模块
    Linear8bitLt,  # 导入 Linear8bitLt 模块
    LinearFP4,  # 导入 LinearFP4 模块
    LinearNF4,  # 导入 LinearNF4 模块
    OutlierAwareLinear,  # 导入 OutlierAwareLinear 模块
    Params4bit,  # 导入 Params4bit 模块
    StableEmbedding,  # 导入 StableEmbedding 模块
    SwitchBackLinearBnb,  # 导入 SwitchBackLinearBnb 模块
)
# 导入基于 Triton 的模块
from .triton_based_modules import (
    StandardLinear,  # 导入 StandardLinear 模块
    SwitchBackLinear,  # 导入 SwitchBackLinear 模块
    SwitchBackLinearGlobal,  # 导入 SwitchBackLinearGlobal 模块
    SwitchBackLinearVectorwise,  # 导入 SwitchBackLinearVectorwise 模块
)
```