# `transformer_vq\src\transformer_vq\nn\types.py`

```py
# 导入必要的模块和类型
from dataclasses import fields
from typing import Any
from typing import Callable
from typing import List

# 导入 JAX 库中的初始化器和数组类型
import jax.nn.initializers as inits
import jax.numpy as jnp
from flax import struct
from jax import Array

# 定义类型别名
PRNGKey = Any
Shape = List[int]
Dtype = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

# 定义一个数据类，表示 Transformer 模型的配置
@struct.dataclass
class TransformerConfig:
    # 定义配置的各种参数
    param_dtype: Dtype
    dtype: Dtype
    global_batch_size: int
    sequence_len: int
    update_len: int
    block_len: int
    mem_len: int
    grad_thru_cache: bool
    agg_cache: bool
    d_model: int
    d_k: int
    d_v: int
    d_ff: int
    n_head: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: str
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    r_init: Initializer
    b_init: Initializer
    no_emb: bool = False

    # 定义一个类方法，用于创建配置对象
    @classmethod
    def create(cls, **kwargs):
        # 获取配置类的字段名和类型
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        # 过滤掉不在字段中的参数
        filtered = {k: v for k, v in kwargs.items() if k in signature}

        # 如果参数类型是字符串，则转换为 JAX 的数据类型
        if isinstance(filtered["param_dtype"], str):
            filtered["param_dtype"] = jnp.dtype(filtered["param_dtype])

        if isinstance(filtered["dtype"], str):
            filtered["dtype"] = jnp.dtype(filtered["dtype"])

        # 将整数 0 和 1 转换为布尔类型
        for k, v in filtered.items():
            if signature[k] is bool and v in {0, 1}:
                filtered[k] = bool(v)

        # 初始化一些参数的默认值
        filtered["e_init"] = inits.normal(1.0)
        filtered["w_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["r_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["b_init"] = inits.zeros

        # 使用过滤后的参数创建配置对象并返回
        return cls(**filtered)
```