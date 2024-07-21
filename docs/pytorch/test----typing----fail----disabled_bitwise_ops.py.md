# `.\pytorch\test\typing\fail\disabled_bitwise_ops.py`

```py
# 导入 torch 库，用于科学计算和机器学习任务
import torch

# 定义一个浮点数张量 a，所有元素初始化为 1，数据类型为 torch.float64
a = torch.ones(3, dtype=torch.float64)

# 定义一个整数 i，未初始化，值为 0
i = int()

# 执行按位或操作 i | a，此处会导致错误：不支持的操作数类型
i | a  # E: Unsupported operand types
```