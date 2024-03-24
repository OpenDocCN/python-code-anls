# `.\lucidrains\flash-cosine-sim-attention\benchmark.py`

```py
# 导入必要的库
import argparse
from itertools import product

import torch
from torch import einsum
assert torch.cuda.is_available(), 'cuda must be available to run benchmark'

# 导入自定义模块
from flash_cosine_sim_attention.benchmark import benchmark
from flash_cosine_sim_attention import flash_cosine_sim_attention, l2norm_tensors

# 定义辅助函数

# 检查变量是否存在
def exists(t):
    return t is not None

# 将输入转换为元组
def cast_tuple(t):
    return t if isinstance(t, tuple) else (t,)

# 解析命令行参数

parser = argparse.ArgumentParser()
parser.add_argument('--causal', default = False, action = 'store_true')
parser.add_argument('--mask-prob', type = float, default = 0.)
parser.add_argument('--only-forwards', default = False, action = 'store_true')
parser.add_argument('--only-backwards', default = False, action = 'store_true')
parser.add_argument('--num-times', default = 20, type = int)
args = parser.parse_args()

# 定义常量

BATCH_SIZES = 4
HEADS = 8
DIM = 64

CAUSAL = args.causal
SHOULD_MASK = args.mask_prob > 0.

assert args.mask_prob >= 0 and args.mask_prob < 1.
assert not (args.only_forwards and args.only_backwards)
assert not (CAUSAL and SHOULD_MASK)

TEST_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)

TEST_FORWARDS = not args.only_backwards
TEST_BACKWARDS = not args.only_forwards

# 简化的余弦相似度注意力机制用于基准测试

def simplified_cosine_sim_attention(
    q,
    k,
    v,
    scale = 10,
    l2norm_qk = True,
    causal_mask = None,
    mask = None
):
    if l2norm_qk:
        q, k = l2norm_tensors(q, k)

    sim = einsum(f'b h i d, b h j d -> b h i j', q, k)
    sim = sim * scale

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(sim.dtype).max)

    if exists(causal_mask):
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    attn = sim.softmax(dim = -1)
    return einsum(f'b h i j, b h j d -> b h i d', attn, v)

# 创建基准测试函数

fused_attention_fn = benchmark(
    flash_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS,
    num_times = args.num_times
)

attention_fn = benchmark(
    simplified_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS,
    num_times = args.num_times
)

# 所有排列组合

params = dict((
    ('batch size', BATCH_SIZES),
    ('heads', HEADS),
    ('feature dimension', DIM)
))

permutations = list(product(*map(cast_tuple, params.values())))

for name, dtype in (('float32', torch.float32), ('float16', torch.float16)):
    # 对于每个批次大小、头数和维度的排列组合
    for batch, heads, dim in permutations:
        # 打印分隔线
        print('-' * 60)
        # 打印当前排列组合的名称、批次大小、头数和维度
        print(f'{name}\t\tbatch: {batch}\theads: {heads}\tdim {dim}')
        # 打印分隔线
        print('-' * 60)

        # 对于测试序列长度中的每个序列长度
        for seq in TEST_SEQUENCE_LENGTHS:
            # 生成随机的查询、键和值张量，设置为需要梯度计算，并移动到 GPU 上
            q = torch.randn(batch, heads, seq, dim, dtype=dtype).cuda().requires_grad_()
            k = torch.randn(batch, heads, seq, dim, dtype=dtype).cuda().requires_grad_()
            v = torch.randn(batch, heads, seq, dim, dtype=dtype).cuda().requires_grad_()

            # 生成一个上三角矩阵作为因果掩码
            causal_mask = torch.ones((seq, seq), dtype=torch.bool).cuda().triu(1)

            # 初始化融合注意力函数参数和基准函数参数
            fused_args = dict(causal=CAUSAL)
            baseline_args = dict()

            # 如果使用因果掩码
            if CAUSAL:
                baseline_args = {**baseline_args, 'causal_mask': causal_mask}

            # 如果需要进行掩码
            if SHOULD_MASK:
                # 生成一个掩码张量
                mask = torch.zeros((batch, seq)).float().cuda().uniform_(0, 1) > args.mask_prob

                # 更新融合注意力函数参数和基准函数参数
                fused_args = {**fused_args, 'mask': mask}
                baseline_args = {**baseline_args, 'mask': mask}

            # 运行基准函数并考虑内存溢出
            fused_time = fused_attention_fn(q, k, v, **fused_args)

            try:
                baseline_time = attention_fn(q, k, v, **baseline_args)
            except:
                # 清空 GPU 缓存
                torch.cuda.empty_cache()
                baseline_time = -1

            # 计算融合函数相对于基准函数的速度差异
            times_slower = (fused_time / baseline_time) if baseline_time != -1 else 0.
            baseline_time_str = 'oom' if baseline_time == -1 else f"{baseline_time:.2f}ms"

            # 打印结果信息：序列长度、速度差异倍数、融合函数时间、基准函数时间
            print(f'seq_len: {seq}\tslower: {times_slower:.2f}x\tkernel: {fused_time:.2f}ms\tbaseline: {baseline_time_str}')
```