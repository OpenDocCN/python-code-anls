# `transformer_vq\src\transformer_vq\ops\sample.py`

```
# 导入必要的模块
import functools  # 用于创建偏函数
import jax  # JAX 深度学习库
import jax.numpy as jnp  # JAX 的 NumPy 接口
from transformer_vq.nn.model import Transformer  # 导入 Transformer 模型
from transformer_vq.nn.prob import nucleus  # 导入 nucleus 模块

# 使用 jax.pmap 创建一个偏函数，用于在多个设备上并行执行操作
@functools.partial(
    jax.pmap,  # 部分应用 jax.pmap 函数
    axis_name="devices",  # 指定轴名称为 "devices"
    static_broadcasted_argnums=(0, 1),  # 指定静态广播参数的索引
)
# 定义一个并行采样操作，接受配置、结束符 ID、参数和随机数生成器作为输入
def sample_op(config, eos_id, params, rng):
    # 计算每个设备上的批大小
    device_batch_size = config.global_batch_size // jax.device_count()

    # 定义采样操作的循环体
    def body(carry, discard):
        # 切分随机数生成器
        rng_new, rng_sample = jax.random.split(carry["rng"])
        # 使用 Transformer 模型进行前向传播
        outputs = Transformer(config).apply(
# 传入参数为 params，inputs 为上一个 token，doc_ids 为文档 id，state 为注意力状态，vq_spec 为 None，rngs 为 None
{"params": params},
inputs=carry["token_prev"],
doc_ids=carry["doc_ids"],
state=carry["attn_state"],
vq_spec=None,
rngs=None,
)
# 通过 nucleus 函数获取下一个 token 的概率分布
nucleus_logits = nucleus(outputs["logprobs"][:, -1:, :], p=config.p_nucleus)
# 通过随机抽样获取下一个 token
token_new = jax.random.categorical(rng_sample, logits=nucleus_logits)
# 更新状态和 token
carry_new = dict(
attn_state=outputs["attn_state"],
token_prev=token_new,
doc_ids=carry["doc_ids"] + jnp.equal(token_new, eos_id).astype(jnp.int32),
rng=rng_new,
)
# 返回更新后的状态和 token
return carry_new, token_new

# 使用 jax.lax.scan 函数执行 body 函数
_, tokens_all = jax.lax.scan(
f=body,
init=dict(
# 初始化注意力机制的状态
attn_state = Transformer.initial_state(config, device_batch_size)
# 创建一个全零的文档 ID 数组
doc_ids = jnp.zeros(shape=[device_batch_size, 1], dtype=jnp.int32)
# 创建一个全为结束标记的 token_prev 数组
token_prev = eos_id * jnp.ones(shape=[device_batch_size, 1], dtype=jnp.int32)
# 生成一个随机数种子
rng = rng,
# 使用 Transformer 模型生成 tokens_all
tokens_all = Transformer.generate(
    params=params,
    attn_state=attn_state,
    doc_ids=doc_ids,
    token_prev=token_prev,
    rng=rng,
    xs=jnp.arange(config.sequence_len),
    length=config.sequence_len,
    unroll=1,
)
# 压缩 tokens_all 的最后一个维度
tokens_all = jnp.squeeze(tokens_all, -1)
# 调换 tokens_all 的维度顺序
tokens_all = jnp.transpose(tokens_all, (1, 0))
# 返回 tokens_all
return tokens_all
```