# `.\PaddleOCR\test_tipc\supplementary\slim\slim_fpgm.py`

```
# 导入 paddleslim 模块
import paddleslim
# 导入 paddle 模块
import paddle
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 从 paddleslim.dygraph 模块中导入 FPGMFilterPruner 类
from paddleslim.dygraph import FPGMFilterPruner

# 定义函数 prune_model，用于对模型进行剪枝
def prune_model(model, input_shape, prune_ratio=0.1):

    # 计算模型的 FLOPs
    flops = paddle.flops(model, input_shape)
    # 创建 FPGMFilterPruner 对象，传入模型和输入形状
    pruner = FPGMFilterPruner(model, input_shape)

    # 创建一个空字典 params_sensitive 用于存储敏感参数
    params_sensitive = {}
    # 遍历模型的参数
    for param in model.parameters():
        # 如果参数名称中不包含 'transpose' 和 'linear'
        if 'transpose' not in param.name and 'linear' not in param.name:
            # 设置剪枝比例为 10%。值越大，剪枝的卷积权重越多
            params_sensitive[param.name] = prune_ratio

    # 使用 params_sensitive 中的敏感参数进行剪枝，[0] 表示第一个 epoch
    plan = pruner.prune_vars(params_sensitive, [0])

    # 重新计算模型的 FLOPs
    flops = paddle.flops(model, input_shape)
    # 返回剪枝后的模型
    return model
```