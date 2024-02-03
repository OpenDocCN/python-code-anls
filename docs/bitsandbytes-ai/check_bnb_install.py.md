# `bitsandbytes\check_bnb_install.py`

```py
# 导入 torch 库
import torch

# 导入 bitsandbytes 库，重命名为 bnb
import bitsandbytes as bnb

# 创建一个需要梯度的张量 p，并将其移动到 GPU 上
p = torch.nn.Parameter(torch.rand(10,10).cuda())
# 创建一个需要梯度的张量 a，并将其移动到 GPU 上
a = torch.rand(10,10).cuda()

# 计算张量 p 的所有元素之和，并转换为 Python 数值
p1 = p.data.sum().item()

# 使用 bitsandbytes 库中的 Adam 优化器来优化参数 p
adam = bnb.optim.Adam([p])

# 计算张量 a 与张量 p 的乘积
out = a*p
# 计算乘积张量的所有元素之和
loss = out.sum()
# 反向传播计算梯度
loss.backward()
# 使用 Adam 优化器更新参数
adam.step()

# 再次计算张量 p 的所有元素之和，并转换为 Python 数值
p2 = p.data.sum().item()

# 断言 p1 与 p2 不相等
assert p1 != p2
# 打印成功信息
print('SUCCESS!')
print('Installation was successful!')
```