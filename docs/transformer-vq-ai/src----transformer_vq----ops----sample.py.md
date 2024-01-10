# `transformer_vq\src\transformer_vq\ops\sample.py`

```
# 导入 functools 模块
import functools

# 导入 jax 模块
import jax
# 导入 jax.numpy 模块并重命名为 jnp
import jax.numpy as jnp

# 从 transformer_vq.nn.model 模块中导入 Transformer 类
from transformer_vq.nn.model import Transformer
# 从 transformer_vq.nn.prob 模块中导入 nucleus 函数

# 使用 jax.pmap 函数创建一个新的函数，并使用 functools.partial 函数对其进行部分参数化
@functools.partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(0, 1),
)
# 定义 sample_op 函数，接受 config, eos_id, params, rng 四个参数
def sample_op(config, eos_id, params, rng):
    # 计算每个设备上的批处理大小
    device_batch_size = config.global_batch_size // jax.device_count()

    # 定义 body 函数，接受 carry, discard 两个参数
    def body(carry, discard):
        # 使用 jax.random.split 函数将随机数生成器 rng 分割成两个新的随机数生成器 rng_new, rng_sample
        rng_new, rng_sample = jax.random.split(carry["rng"])
        # 使用 Transformer 类创建一个模型，并对其应用
        outputs = Transformer(config).apply(
            {"params": params},
            inputs=carry["token_prev"],
            doc_ids=carry["doc_ids"],
            state=carry["attn_state"],
            vq_spec=None,
            rngs=None,
        )
        # 使用 nucleus 函数对输出的 logprobs 进行筛选，得到新的概率分布
        nucleus_logits = nucleus(outputs["logprobs"][:, -1:, :], p=config.p_nucleus)
        # 使用 jax.random.categorical 函数根据给定的概率分布进行采样，得到新的 token
        token_new = jax.random.categorical(rng_sample, logits=nucleus_logits)
        # 创建新的 carry 字典，更新其中的 attn_state, token_prev, doc_ids, rng 字段
        carry_new = dict(
            attn_state=outputs["attn_state"],
            token_prev=token_new,
            doc_ids=carry["doc_ids"] + jnp.equal(token_new, eos_id).astype(jnp.int32),
            rng=rng_new,
        )
        return carry_new, token_new

    # 使用 jax.lax.scan 函数对 body 函数进行扫描，得到最终的 tokens_all
    _, tokens_all = jax.lax.scan(
        f=body,
        init=dict(
            attn_state=Transformer.initial_state(config, device_batch_size),
            doc_ids=jnp.zeros(shape=[device_batch_size, 1], dtype=jnp.int32),
            token_prev=eos_id * jnp.ones(shape=[device_batch_size, 1], dtype=jnp.int32),
            rng=rng,
        ),
        xs=jnp.arange(config.sequence_len),
        length=config.sequence_len,
        unroll=1,
    )
    # 去除 tokens_all 的一个维度
    tokens_all = jnp.squeeze(tokens_all, -1)
    # 转置 tokens_all 的维度
    tokens_all = jnp.transpose(tokens_all, (1, 0))
    # 返回 tokens_all
    return tokens_all
```