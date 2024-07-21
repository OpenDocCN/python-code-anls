# `.\pytorch\torch\masked\__init__.py`

```py
# 导入 torch.masked._ops 模块中的函数和变量
from torch.masked._ops import (
    _canonical_dim,          # 导入 _canonical_dim 函数，用于规范化维度
    _combine_input_and_mask, # 导入 _combine_input_and_mask 函数，用于将输入和掩码结合
    _generate_docstring,     # 导入 _generate_docstring 函数，用于生成文档字符串
    _input_mask,             # 导入 _input_mask 变量，可能是输入掩码相关
    _output_mask,            # 导入 _output_mask 变量，可能是输出掩码相关
    _reduction_identity,     # 导入 _reduction_identity 变量，可能是规约操作的标识
    _where,                  # 导入 _where 函数，用于条件查找
    amax,                    # 导入 amax 函数，用于计算最大值
    amin,                    # 导入 amin 函数，用于计算最小值
    argmax,                  # 导入 argmax 函数，用于计算最大值的索引
    argmin,                  # 导入 argmin 函数，用于计算最小值的索引
    cumprod,                 # 导入 cumprod 函数，用于计算累积乘积
    cumsum,                  # 导入 cumsum 函数，用于计算累积和
    log_softmax,             # 导入 log_softmax 函数，用于计算对数 softmax
    logaddexp,               # 导入 logaddexp 函数，用于计算对数加指数
    logsumexp,               # 导入 logsumexp 函数，用于计算对数和指数
    mean,                    # 导入 mean 函数，用于计算均值
    median,                  # 导入 median 函数，用于计算中位数
    norm,                    # 导入 norm 函数，用于计算范数
    normalize,               # 导入 normalize 函数，用于标准化
    prod,                    # 导入 prod 函数，用于计算乘积
    softmax,                 # 导入 softmax 函数，用于计算 softmax
    softmin,                 # 导入 softmin 函数，用于计算 softmin
    std,                     # 导入 std 函数，用于计算标准差
    sum,                     # 导入 sum 函数，用于计算总和
    var,                     # 导入 var 函数，用于计算方差
)

# 导入 torch.masked.maskedtensor.core 模块中的函数和类
from torch.masked.maskedtensor.core import is_masked_tensor, MaskedTensor
# 导入 torch.masked.maskedtensor.creation 模块中的函数
from torch.masked.maskedtensor.creation import as_masked_tensor, masked_tensor

# 模块中导出的公共接口列表
__all__ = [
    "as_masked_tensor",  # 将 as_masked_tensor 添加到公共接口列表中
    "is_masked_tensor",  # 将 is_masked_tensor 添加到公共接口列表中
    "masked_tensor",     # 将 masked_tensor 添加到公共接口列表中
    "MaskedTensor",      # 将 MaskedTensor 添加到公共接口列表中
]
```