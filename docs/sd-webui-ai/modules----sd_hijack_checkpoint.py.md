# `stable-diffusion-webui\modules\sd_hijack_checkpoint.py`

```
# 从 torch.utils.checkpoint 模块中导入 checkpoint 函数
from torch.utils.checkpoint import checkpoint

# 导入 ldm.modules.attention 和 ldm.modules.diffusionmodules.openaimodel 模块
import ldm.modules.attention
import ldm.modules.diffusionmodules.openaimodel

# 定义 BasicTransformerBlock_forward 函数，使用 checkpoint 函数对 _forward 方法进行检查点操作
def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context)

# 定义 AttentionBlock_forward 函数，使用 checkpoint 函数对 _forward 方法进行检查点操作
def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x)

# 定义 ResBlock_forward 函数，使用 checkpoint 函数对 _forward 方法进行检查点操作
def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb)

# 存储变量 stored，用于存储函数引用
stored = []

# 定义 add 函数，用于添加函数引用到 stored 变量中
def add():
    # 如果 stored 中已经有函数引用，则直接返回
    if len(stored) != 0:
        return

    # 将函数引用添加到 stored 中
    stored.extend([
        ldm.modules.attention.BasicTransformerBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.ResBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward
    ])

    # 重写函数引用
    ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward

# 定义 remove 函数，用于移除函数引用
def remove():
    # 如果 stored 中没有函数引用，则直接返回
    if len(stored) == 0:
        return

    # 恢复原始函数引用
    ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    # 清空 stored 变量
    stored.clear()
```