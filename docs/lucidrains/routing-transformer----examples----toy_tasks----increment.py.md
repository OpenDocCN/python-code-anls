# `.\lucidrains\routing-transformer\examples\toy_tasks\increment.py`

```
# 导入所需的库
import torch
import numpy as np
import math
import time
import random
from torch.optim import Adam
from routing_transformer.routing_transformer import RoutingTransformerLM
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper

# 创建 RoutingTransformerLM 模型实例
s = RoutingTransformerLM(
    num_tokens = 256 + 4,
    dim = 1024,
    depth = 2,
    heads = 8,
    max_seq_len = 256,
    causal = True,
    window_size = 128
).cuda()

# 使用 AutoregressiveWrapper 对模型进行包装
s = AutoregressiveWrapper(s, ignore_index = 0, pad_value = 0)
# 使用 Adam 优化器对模型参数进行优化
opt = Adam(s.parameters(), lr=1e-4)

# 定义批次大小、源序列长度和目标序列长度
N_BATCH = 32
SRC_SEQ_LEN = 128
TGT_SEQ_LEN = 128

# 定义起始符、结束符和位置编码
bos = 1*torch.ones(N_BATCH, 1).long()
eos = 2*torch.ones(N_BATCH, 1).long()
pos = 3*torch.ones(N_BATCH, 1).long()

# 进行训练循环
for i in range(10000):
    # 生成随机的训练输入序列
    train_seq_in = torch.randint(4, 6, (N_BATCH, SRC_SEQ_LEN - 2)).long()
    # 训练输出序列为输入序列加一
    train_seq_out = train_seq_in + 1

    # 构建完整的训练序列，包括起始符、位置编码、输入序列、输出序列和结束符
    train_seq = torch.cat([bos, train_seq_in, pos, pos, pos, train_seq_out, eos], dim=1).cuda()

    # 计算模型的损失
    loss = s(train_seq, return_loss = True)
    # 反向传播计算梯度
    loss.backward()
    # 根据梯度更新模型参数
    opt.step()
    # 清空梯度
    opt.zero_grad()
    # 打印当前迭代次数和损失值
    print(i, loss.item())
```