# `transformer_vq\src\transformer_vq\ops\loss.py`

```
# 导入 jax 库中的 numpy 模块，并重命名为 jnp
import jax.numpy as jnp

# 从 transformer_vq.nn.grad 模块中导入 sg 函数
from transformer_vq.nn.grad import sg
# 从 transformer_vq.nn.model 模块中导入 Transformer 类

# 定义损失函数，接受多个参数
def loss_fn(
    params,  # 模型参数
    config,  # 模型配置
    batch,   # 输入数据批次
    attn_state,  # 注意力状态
    vq_spec,  # VQ 特征
    rngs,  # 随机数生成器
):
    # 定义调用参数字典
    call_kwargs = dict(
        inputs=batch["inputs"],  # 输入数据
        doc_ids=batch["doc_ids"],  # 文档 ID
        state=attn_state,  # 注意力状态
        vq_spec=vq_spec,  # VQ 特征
    )
    # 如果 rngs 不为空，则将其添加到 call_kwargs 字典中
    if rngs is not None:
        call_kwargs["rngs"] = rngs
    # 使用 Transformer 对象应用参数和关键字参数
    outputs = Transformer(config).apply({"params": params}, **call_kwargs)
    # 获取输出结果中的 logprobs
    logprobs = outputs["logprobs"]
    # 计算未经缩放的语言模型损失
    l_lm_premask = -jnp.take_along_axis(logprobs, batch["targets"][..., None], axis=-1)
    l_lm_premask = jnp.squeeze(l_lm_premask, axis=-1)
    l_lm_unscaled = (batch["loss_mask"] * l_lm_premask).mean()
    # 创建一个值为零的 jnp.float32 类型的变量
    zero = jnp.zeros([], dtype=jnp.float32)
    # 如果输出结果中包含 l_commit，则将其赋值给 l_commit，否则赋值为零
    l_commit = outputs["l_commit"] if "l_commit" in outputs else zero
    # 如果输出结果中包含 l_codebook，则将其赋值给 l_codebook，否则赋值为零
    l_codebook = outputs["l_codebook"] if "l_codebook" in outputs else zero
    # 将 l_lm_unscaled 添加到输出结果中
    outputs["l_lm_unscaled"] = l_lm_unscaled
    # 计算复合损失
    composite_loss = l_lm_unscaled + config.c_beta * l_commit + l_codebook
    # 如果输出结果中不包含 metrics，则添加一个空字典
    if "metrics" not in outputs:
        outputs["metrics"] = dict()
    # 将各种损失指标添加到 metrics 中
    outputs["metrics"]["loss_lm_per_token_unscaled"] = sg(l_lm_unscaled)
    outputs["metrics"]["loss_mask_per_token"] = sg(batch["loss_mask"].mean())
    outputs["metrics"]["loss_commit_per_token"] = sg(l_commit)
    # 返回复合损失和输出结果
    return composite_loss, outputs
```