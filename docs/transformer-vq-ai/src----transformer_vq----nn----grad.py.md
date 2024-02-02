# `transformer_vq\src\transformer_vq\nn\grad.py`

```py
# 导入 flax.linen 库中的 nn 模块，并重命名为 nn
import flax.linen as nn
# 导入 jax 库
import jax

# 定义函数 sg，返回输入的梯度停止版本
def sg(x):
    return jax.lax.stop_gradient(x)

# 定义函数 st，返回输入减去其梯度停止版本的结果
def st(x):
    return x - sg(x)

# 定义函数 maybe_remat，根据 enabled 参数决定是否启用重组计算
def maybe_remat(module, enabled):
    # 如果 enabled 为真，则返回 module 的重组计算版本
    if enabled:
        return nn.remat(module)
    # 如果 enabled 为假，则返回原始的 module
    else:
        return module
```