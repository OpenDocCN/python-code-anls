# `.\lucidrains\triton-transformer\assert.py`

```py
# 导入 PyTorch 库
import torch
# 从 triton_transformer 模块中导入 Transformer 类
from triton_transformer import Transformer

# 检查是否有可用的 CUDA 设备
assert torch.cuda.is_available()

# 实例化模型和数据

# 创建 Transformer 模型对象，设置参数：标记数量为 256，最大序列长度为 1024，维度为 512，深度为 6，头数为 8，头维度为 64，使用因果性，不使用 Triton
model = Transformer(
    num_tokens = 256,
    max_seq_len = 1024,
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    causal = True,
    use_triton = False
).cuda()

# 生成一个大小为 (1, 1024) 的张量，填充随机整数，放在 CUDA 设备上
x = torch.randint(0, 256, (1, 1024)).cuda()
# 生成一个大小为 (1, 1024) 的张量，填充随机整数，放在 CUDA 设备上
labels = torch.randint(0, 256, (1, 1024)).cuda()

# 无 Triton 的前向传播和反向传播

# 计算模型输出和损失
loss = model(x, labels = labels)
# 反向传播计算梯度
loss.backward()

# 复制损失值
loss = loss.clone()
# 复制 token embeddings 的梯度
emb_grad = model.token_emb.weight.grad.clone()
# 复制 LayerNorm 层的权重梯度
ln_weight_grad = model.norm.weight.grad.clone()
# 复制 LayerNorm 层的偏置梯度
ln_bias_grad = model.norm.bias.grad.clone()

# 清零所有梯度
model.zero_grad()

# Triton 的前向传播和反向传播

# 使用 Triton 进行前向传播和反向传播
triton_loss = model(x, labels = labels, use_triton = True)
# Triton 反向传播计算梯度
triton_loss.backward()

# 复制 Triton 下的 token embeddings 的梯度
triton_emb_grad = model.token_emb.weight.grad.clone()
# 复制 Triton 下的 LayerNorm 层的权重梯度
triton_ln_weight_grad = model.norm.weight.grad.clone()
# 复制 Triton 下的 LayerNorm 层的偏置梯度
triton_ln_bias_grad = model.norm.bias.grad.clone()

# 应该相等，对输出和 token embeddings 的梯度进行检查

# 检查输出是否相等
assert torch.allclose(loss.cpu(), triton_loss.cpu(), atol=1e-6), 'output is the same'
# 检查 token embeddings 的梯度是否相等
assert torch.allclose(emb_grad.cpu(), triton_emb_grad.cpu(), atol=2e-6), 'grad is the same'
# 检查 LayerNorm 层的权重梯度是否相等
assert torch.allclose(ln_weight_grad.cpu(), triton_ln_weight_grad.cpu(), atol=2e-6), 'layernorm weight grad is the same'
# 检查 LayerNorm 层的偏置梯度是否相等
assert torch.allclose(ln_bias_grad.cpu(), triton_ln_bias_grad.cpu(), atol=2e-6), 'layernorm bias grad is the same'

# 打印成功信息
print('succeeded')
```