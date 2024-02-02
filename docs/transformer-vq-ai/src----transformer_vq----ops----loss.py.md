# `transformer_vq\src\transformer_vq\ops\loss.py`

```py
# 导入 jax 库中的 numpy 模块，并重命名为 jnp
import jax.numpy as jnp

# 从 transformer_vq.nn.grad 模块中导入 sg 函数
from transformer_vq.nn.grad import sg
# 从 transformer_vq.nn.model 模块中导入 Transformer 类
from transformer_vq.nn.model import Transformer

# 定义损失函数，接受多个参数
def loss_fn(
    params,  # 模型参数
    config,  # 模型配置
    batch,   # 输入数据批次
    attn_state,  # 注意力状态
    vq_spec,  # VQ 特征
    rngs,     # 随机数生成器
):
    # 准备调用 Transformer 模型的参数
    call_kwargs = dict(
        inputs=batch["inputs"],  # 输入数据
        doc_ids=batch["doc_ids"],  # 文档 ID
        state=attn_state,  # 注意力状态
        vq_spec=vq_spec,  # VQ 特征
    )
    # 如果随机数生成器不为空，则添加到调用参数中
    if rngs is not None:
        call_kwargs["rngs"] = rngs
    # 调用 Transformer 模型，获取输出结果
    outputs = Transformer(config).apply({"params": params}, **call_kwargs)
    # 计算平均的 lm 损失，包括每个设备的批次位置和段落标记
    logprobs = outputs["logprobs"]  # 获取输出结果中的对数概率
    l_lm_premask = -jnp.take_along_axis(logprobs, batch["targets"][..., None], axis=-1)  # 计算 lm 损失
    l_lm_premask = jnp.squeeze(l_lm_premask, axis=-1)  # 去除多余的维度
    l_lm_unscaled = (batch["loss_mask"] * l_lm_premask).mean()  # 计算未缩放的 lm 损失
    zero = jnp.zeros([], dtype=jnp.float32)  # 创建一个值为 0 的张量
    l_commit = outputs["l_commit"] if "l_commit" in outputs else zero  # 获取输出结果中的 l_commit，如果不存在则使用 zero
    l_codebook = outputs["l_codebook"] if "l_codebook" in outputs else zero  # 获取输出结果中的 l_codebook，如果不存在则使用 zero
    outputs["l_lm_unscaled"] = l_lm_unscaled  # 将未缩放的 lm 损失添加到输出结果中
    composite_loss = l_lm_unscaled + config.c_beta * l_commit + l_codebook  # 计算综合损失
    if "metrics" not in outputs:  # 如果输出结果中不存在 metrics
        outputs["metrics"] = dict()  # 则创建一个空的 metrics 字典
    outputs["metrics"]["loss_lm_per_token_unscaled"] = sg(l_lm_unscaled)  # 将未缩放的 lm 损失添加到 metrics 中
    outputs["metrics"]["loss_mask_per_token"] = sg(batch["loss_mask"].mean())  # 将每个标记的损失掩码添加到 metrics 中
    outputs["metrics"]["loss_commit_per_token"] = sg(l_commit)  # 将每个标记的 l_commit 添加到 metrics 中
    return composite_loss, outputs  # 返回综合损失和输出结果
```