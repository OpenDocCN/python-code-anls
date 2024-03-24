# `.\lucidrains\routing-transformer\examples\toy_tasks\enc_dec_copy_task.py`

```
# 导入必要的库
import tqdm
import torch
import torch.optim as optim

# 导入自定义的模型类RoutingTransformerEncDec

from routing_transformer import RoutingTransformerEncDec

# 定义常量

NUM_BATCHES = int(1e5)  # 总批次数
BATCH_SIZE = 32  # 每批次的样本数量
LEARNING_RATE = 1e-4  # 学习率
GENERATE_EVERY  = 100  # 每隔多少批次生成一次输出
NUM_TOKENS = 256 + 2  # 标记的数量
ENC_SEQ_LEN = 128  # 编码器序列长度
DEC_SEQ_LEN = 256  # 解码器序列长度

# 定义辅助函数

def cycle():
    # 生成器函数，无限循环生成数据
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# 实例化模型

model = RoutingTransformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=3,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_window_size=32,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = 3,
    dec_heads = 8,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_window_size=32,
).cuda()

# 定义优化器

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练过程

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    # 获取下一个数据批次
    src, tgt, src_mask, tgt_mask = next(cycle())
    # 计算损失
    loss, _ = model(src, tgt, enc_input_mask=src_mask, dec_input_mask=tgt_mask, return_loss = True, randomly_truncate_sequence = True)
    # 反向传播
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # 更新参数
    optim.step()
    # 梯度清零
    optim.zero_grad()

    # 每GENERATE_EVERY个批次生成一次输出
    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[0:1], src_mask[0:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        # 生成输出
        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_input_mask=src_mask)
        # 计算错误数量
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
```