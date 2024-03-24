# `.\lucidrains\x-transformers\examples\toy_tasks\enc_dec_copy.py`

```
# 导入必要的库
import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer

# 定义常量
NUM_BATCHES = int(1e5)  # 总批次数
BATCH_SIZE = 32  # 每批次的样本数量
LEARNING_RATE = 3e-4  # 学习率
GENERATE_EVERY  = 100  # 每隔多少批次生成输出
NUM_TOKENS = 16 + 2  # 标记的数量
ENC_SEQ_LEN = 32  # 编码器序列长度
DEC_SEQ_LEN = 64 + 1  # 解码器序列长度

# 定义辅助函数
def cycle():
    # 生成器函数，无限循环生成数据
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().cuda()
        yield (src, tgt, src_mask)

# 实例化模型
model = XTransformer(
    dim = 512,
    tie_token_emb = True,
    return_tgt_loss = True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth = 3,
    enc_heads = 8,
    enc_max_seq_len = ENC_SEQ_LEN,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = 3,
    dec_heads = 8,
    dec_max_seq_len = DEC_SEQ_LEN
).cuda()

# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask = next(cycle())

    # 计算损失并反向传播
    loss = model(src, tgt, mask=src_mask)
    loss.backward()
    print(f'{i}: {loss.item()}')

    optim.step()
    optim.zero_grad()

    # 每隔一定批次生成输出
    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask = next(cycle())
        src, src_mask = src[:1], src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        # 生成输出并计算错误数量
        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask = src_mask)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
```