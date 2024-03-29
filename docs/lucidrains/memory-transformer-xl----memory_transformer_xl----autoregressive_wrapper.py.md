# `.\lucidrains\memory-transformer-xl\memory_transformer_xl\autoregressive_wrapper.py`

```py
# 导入数学库
import math
# 导入partial函数
from functools import partial
# 导入namedtuple
from collections import namedtuple

# 导入torch库
import torch
# 导入torch的nn模块
from torch import nn
# 导入torch的functional模块
import torch.nn.functional as F
# 导入torch的rnn模块
from torch.nn.utils.rnn import pad_sequence

# 定义一个命名元组Return，包含loss和is_last_batch两个字段
Return = namedtuple('Return', ['loss', 'is_last_batch'])

# 定义top_p函数，用于根据概率阈值过滤logits
def top_p(logits, thres = 0.9):
    # 对logits进行降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # 计算累积概率
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 根据阈值确定需要移除的索引
    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # 将需要移除的logits设置为负无穷
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# 定义top_k函数，用于根据概率阈值过滤logits
def top_k(logits, thres = 0.9):
    # 计算需要保留的top k个元素
    k = int((1 - thres) * logits.shape[-1])
    # 获取top k的值和索引
    val, ind = torch.topk(logits, k)
    # 创建与logits相同形状的tensor，并填充为负无穷
    probs = torch.full_like(logits, float('-inf'))
    # 将top k的值填充到对应位置
    probs.scatter_(1, ind, val)
    return probs

# 定义AutoregressiveWrapper类，继承自nn.Module
class AutoregressiveWrapper(nn.Module):
    # 初始化函数
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.seq_len = net.seq_len

    # 生成函数，用于生成序列
    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        # 保存模型当前是否为训练状态
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()

        out = start_tokens

        # 处理默认的masking

        full_mask_like = lambda x: torch.full_like(x, True, dtype=torch.bool, device=x.device)

        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = full_mask_like(out)

        # 处理任意长度的primed序列

        mem = None
        *primes, out = out.split(self.seq_len, dim=1)
        *prime_masks, mask = mask.split(self.seq_len, dim=1)

        for prime, prime_mask in zip(primes, prime_masks):
            _, mem = self.net(prime, memories = mem, mask = prime_mask, **kwargs)

        # 生成直到达到序列长度

        input_len = out.shape[1]

        for _ in range(seq_len):
            logits, mem = self.net(out[:, -input_len:], memories = mem, mask = mask[:, -input_len:], **kwargs)
            logits = logits[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            # 不同于大多数模型，一旦填满完整序列长度，输入从序列长度为1开始

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            # 将样本追加到累��输出

            input_len = input_len % self.seq_len
            input_len += 1

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out
    # 定义前向传播函数，接受输入 x，最大批处理大小 max_batch_size，默认不返回损失，截断长度 truncate_every，以及其他关键字参数
    def forward(self, x, max_batch_size = None, return_loss = False, truncate_every = None, **kwargs):
        # 定义一个填充函数，用于在序列维度上进行填充，保证批处理时序列长度一致
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        # 如果不需要返回损失
        if not return_loss:
            # 如果输入不是张量，则进行填充
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            # 调用网络进行前向传播
            return self.net(x, **kwargs)

        # 如果输入是张量
        if isinstance(x, torch.Tensor):
            # 将输入序列切片为输入和输出序列
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            # 对输入序列进行填充和切片，得到输入和输出序列
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        # 处理输入掩码，解决自回归模型中输入掩码与源序列长度不一致的问题
        mask = kwargs.pop('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]

        # 定义分段函数，用于将序列分段
        segment_fn = lambda x: x.split(self.seq_len, dim=1)
        # 将输入和输出序列分段
        (xi, xo) = map(segment_fn, (xi, xo))

        # 获取分段数量
        num_segments = len(xi)
        # 如果存在掩码，则对掩码进行分段
        mask = segment_fn(mask) if mask is not None else ((None,) * num_segments)

        # 如果未指定最大批处理大小，则使用输入序列的大小
        max_batch_size = x.shape[0] if max_batch_size is None else max_batch_size
        # 定义分批函数，用于将序列按照最大批处理大小进行分批
        split_batch_fn = lambda x: x.split(max_batch_size, dim=0)

        # 计算梯度累积次数
        grad_accumulate_every = math.ceil(x.shape[0] / max_batch_size)
        # 初始化记忆列表
        mems = [None] * grad_accumulate_every

        # 遍历每个分段
        for ind, (xi_seg, xo_seg, mask_seg) in enumerate(zip(xi, xo, mask)):
            # 将输入和输出序列按照最大批处理大小进行分批
            xi_seg, xo_seg = map(split_batch_fn, (xi_seg, xo_seg))
            mask_seg = split_batch_fn(mask_seg) if mask_seg is not None else ((None,) * grad_accumulate_every)
            # 判断是否需要截断
            truncate = truncate_every is not None and ((ind + 1) % truncate_every) == 0

            new_mems = []
            # 遍历每个分批
            for ind, (xi_seg_b, xo_seg_b, mask_seg_b, mem) in enumerate(zip(xi_seg, xo_seg, mask_seg, mems)):
                is_last = ind == (grad_accumulate_every - 1)

                # 调用网络进行前向传播，获取输出和新的记忆
                logits, new_mem = self.net(xi_seg_b, mask = mask_seg_b, memories = mem, detach_lmem = truncate, **kwargs)
                new_mems.append(new_mem)

                # 计算交叉熵损失
                loss = F.cross_entropy(logits.transpose(1, 2), xo_seg_b, ignore_index = self.ignore_index)
                # 返回损失和是否为最后一个分批
                yield Return(loss, is_last)

            mems = new_mems
```