# `.\lucidrains\performer-pytorch\examples\toy_tasks\enc_dec_copy.py`

```py
# 导入必要的库
import tqdm
import torch
import torch.optim as optim
from performer_pytorch import PerformerEncDec
from torch.cuda.amp import autocast, GradScaler

# 定义常量
NUM_BATCHES = int(1e5)  # 总批次数
BATCH_SIZE = 32  # 每批次大小
LEARNING_RATE = 1e-4  # 学习率
GENERATE_EVERY  = 100  # 每隔多少批次生成输出
NUM_TOKENS = 16 + 2  # 标记的数量
ENC_SEQ_LEN = 32  # 编码器序列长度
DEC_SEQ_LEN = 64 + 1  # 解码器序列长度

# 定义生成数据的辅助函数
def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# 实例化模型
model = PerformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=1,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_reversible=True,
    enc_feature_redraw_interval=1000,
    enc_nb_features = 64,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=3,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_reversible=True,
    dec_feature_redraw_interval=1000,
    dec_nb_features=64
).cuda()

# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask, tgt_mask = next(cycle())

    with autocast():
        loss = model(src, tgt, enc_mask=src_mask, dec_mask=tgt_mask)

    scaler.scale(loss).backward()
    print(f'{i}: {loss.item()}')

    scaler.step(optim)
    scaler.update()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[:1], src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_mask=src_mask)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
```