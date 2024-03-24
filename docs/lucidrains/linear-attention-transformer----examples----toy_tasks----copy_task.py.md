# `.\lucidrains\linear-attention-transformer\examples\toy_tasks\copy_task.py`

```py
# 导入必要的库
import tqdm
import torch
import torch.optim as optim

# 导入自定义模块
from linear_attention_transformer import LinearAttentionTransformerLM
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64

# 定义生成数据的辅助函数
def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1] - 1).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# 实例化编码器和解码器模型
enc = LinearAttentionTransformerLM(
    num_tokens = NUM_TOKENS,
    dim = 512,
    heads = 8,
    depth = 1,
    max_seq_len = ENC_SEQ_LEN,
    shift_tokens = True,
    return_embeddings = True
).cuda()

dec = LinearAttentionTransformerLM(
    num_tokens = NUM_TOKENS,
    dim = 512,
    heads = 8,
    depth = 3,
    causal = True,
    shift_tokens = True,
    blindspot_size = 2,             # a small blindspot greatly saves on memory
    max_seq_len = DEC_SEQ_LEN,
    receives_context = True
).cuda()

# 将解码器包装为自回归模型
dec = AutoregressiveWrapper(dec)

# 定义优化器
optim = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=LEARNING_RATE)

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    enc.train(), dec.train()
    src, tgt, src_mask, tgt_mask = next(cycle())

    # 编码器生成上下文信息
    context = enc(src, input_mask = src_mask)
    # 解码器计算损失
    loss = dec(tgt, context = context, input_mask = tgt_mask, context_mask = src_mask, return_loss = True)
    loss.backward()
    print(loss.item())

    optim.step()
    optim.zero_grad()

    if i % GENERATE_EVERY == 0:
        enc.eval(), dec.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[0:1], src_mask[0:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        # 生成预测结果
        context = enc(src)
        sample = dec.generate(start_tokens, ENC_SEQ_LEN, context = context)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
```