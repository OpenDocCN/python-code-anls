# `transformer_vq\src\transformer_vq\nn\act.py`

```
# 导入 jax 库
import jax
# 导入 jax 库中的 numpy 模块并重命名为 jnp
import jax.numpy as jnp

# 定义 glu 函数，参数为 x
def glu(x):
    # 将输入 x 按照最后一个维度分割成两部分，分别赋值给 xf 和 xg
    xf, xg = jnp.split(x, 2, axis=-1)
    # 返回 xf 与 jax.nn.sigmoid(xg) 的乘积
    return xf * jax.nn.sigmoid(xg)

# 定义 swiglu 函数，参数为 x
def swiglu(x):
    # 将输入 x 按照最后一个维度分割成两部分，分别赋值给 xf 和 xg
    xf, xg = jnp.split(x, 2, axis=-1)
    # 返回 xf 与 jax.nn.silu(xg) 的乘积
    return xf * jax.nn.silu(xg)

# 定义 sqrelu 函数，参数为 x
def sqrelu(x):
    # 返回 jax.nn.relu(x) 的平方
    return jnp.square(jax.nn.relu(x))

# 定义 get_activation 函数，参数为 name
def get_activation(name):
    # 如果 name 等于 "relu"，则执行以下代码
    # 如果激活函数名称为 "relu"，返回 jax.nn.relu 函数
    if name == "relu":
        return jax.nn.relu
    # 如果激活函数名称为 "gelu"，返回 jax.nn.gelu 函数
    if name == "gelu":
        return jax.nn.gelu
    # 如果激活函数名称为 "silu"，返回 jax.nn.silu 函数
    if name == "silu":
        return jax.nn.silu
    # 如果激活函数名称为 "swiglu"，返回 swiglu 函数
    if name == "swiglu":
        return swiglu
    # 如果激活函数名称为 "sqrelu"，返回 sqrelu 函数
    if name == "sqrelu":
        return sqrelu
    # 如果激活函数名称不在以上列表中，抛出 NotImplementedError
    raise NotImplementedError
```