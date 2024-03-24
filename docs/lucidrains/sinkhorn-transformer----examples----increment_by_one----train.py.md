# `.\lucidrains\sinkhorn-transformer\examples\increment_by_one\train.py`

```py
# 导入 torch 库
import torch
# 从 sinkhorn_transformer 库中导入 SinkhornTransformerLM 类
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformerLM
# 从 sinkhorn_transformer 库中导入 AutoregressiveWrapper 类
from sinkhorn_transformer.autoregressive_wrapper import AutoregressiveWrapper

# 定义批量大小
N_BATCH = 16
# 定义源序列长度
SRC_SEQ_LEN = 512
# 定义目标序列长度
TGT_SEQ_LEN = 512

# 创建 SinkhornTransformerLM 编码器对象
enc = SinkhornTransformerLM(
    num_tokens = 64,
    dim = 512,
    depth = 1,
    heads = 8,
    max_seq_len = SRC_SEQ_LEN,
    bucket_size = 64,
    return_embeddings = True
).cuda()

# 创建 SinkhornTransformerLM 解码器对象
dec = SinkhornTransformerLM(
    num_tokens = 64,
    dim = 512,
    depth = 2,
    heads = 8,
    max_seq_len = TGT_SEQ_LEN,
    bucket_size = 64,
    causal = True,
    receives_context = True
).cuda()

# 将解码器包装在 AutoregressiveWrapper 中，设置忽略索引和填充值
dec = AutoregressiveWrapper(dec, ignore_index = 0, pad_value = 0)
# 使用 Adam 优化器，传入编码器和解码器的参数，设置学习率
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-4)

# 定义起始符、结束符和位置符
bos = 1 * torch.ones(N_BATCH, 1).long()
eos = 2 * torch.ones(N_BATCH, 1).long()
pos = 3 * torch.ones(N_BATCH, 1).long()

# 循环训练
for i in range(10000):
    # 生成随机训练序列
    train_seq_in = torch.randint(4, 63, (N_BATCH, SRC_SEQ_LEN-2)).long()
    # 目标序列为输入序列加一
    train_seq_out = train_seq_in + 1

    # 在序列开头和结尾添加起始符和结束符，并转移到 GPU
    x = torch.cat([bos, train_seq_in, eos], dim=1).cuda()
    y = torch.cat([bos, train_seq_out, eos], dim=1).cuda()

    # 编码输入序列，得到上下文信息
    context = enc(x)
    # 计算解码器的损失
    loss = dec(y, context = context, return_loss = True)
    # 反向传播计算梯度
    loss.backward()

    # 更新优化器参数
    opt.step()
    # 梯度清零
    opt.zero_grad()
    # 打印当前迭代次数和损失值
    print(i, loss.item())
```