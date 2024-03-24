# `.\lucidrains\sinkhorn-transformer\sinkhorn_transformer\autoregressive_wrapper.py`

```
# 导入必要的库
from functools import partial
import torch
from random import randint
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformerLM
from sinkhorn_transformer.autopadder import Autopadder

# 定义一个函数，如果值为None，则返回默认值
def default(value, default):
    return value if value is not None else default

# 从logits中选择概率最高的元素，保留概率总和大于阈值的元素
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# 从logits中选择概率最高的k个元素，其余设置为负无穷
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 将序列左侧填充到相同长度
def pad_sequence_left(seqs, value):
    m = max([len(s) for s in seqs])
    return torch.stack([F.pad(s, (m - len(s), 0)) for s in seqs])

# 随机截断输入序列
def random_truncate_inputs(inputs, mask = None, pad_value=0):
    b, t, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    mask = default(mask, torch.ones_like(inputs))
    rand_lengths = torch.randint(2, t, (b, 1))
    rand_mask = (torch.arange(t) < rand_lengths).to(device)
    target_seqs = [t.masked_select(mask) for mask, t in zip(rand_mask, inputs)]
    mask_seqs = [m.masked_select(mask) for mask, m in zip(rand_mask, rand_mask)]
    return pad_sequence_left(target_seqs, pad_value), pad_sequence_left(mask_seqs, False)

# 自回归包装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = None, pad_value = 0, pad_left = True):
        super().__init__()
        assert isinstance(net, SinkhornTransformerLM), 'generative trainer wrapper can only accept SinkhornTransformerLM class'
        self.pad_value = pad_value
        self.ignore_index = default(ignore_index, pad_value)

        self.net = Autopadder(net, pad_left = pad_left)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]
            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)
            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out
    # 定义前向传播函数，接受输入 x，是否返回损失值，是否随机截断序列等参数
    def forward(self, x, return_loss = False, randomly_truncate_sequence = False, **kwargs):
        # 定义一个填充函数，用于在序列维度上进行填充，保证输入数据维度一致
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        # 如果不需要返回损失值
        if not return_loss:
            # 如果输入不是张量，则进行填充
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            # 返回网络处理后的结果
            return self.net(x, **kwargs)

        # 从参数中弹出输入掩码
        m = kwargs.pop('input_mask', None)

        # 如果需要随机截断序列
        if randomly_truncate_sequence:
            # 对输入进行随机截断
            x, m = random_truncate_inputs(x, m, pad_value = self.pad_value)

        # 如果输入是张量
        if isinstance(x, torch.Tensor):
            # 分别获取输入序列和目标序列
            xi, xo = x[:, :-1], x[:, 1:]
        else:
            # 对输入进行填充和截断，获取输入序列和目标序列
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        # 如果存在输入掩码
        if m is not None:
            # 确保输入掩码形状与输入序列的形状一致
            assert m.shape == x.shape[0:2], 'input mask must be the same shape as the input of the auto-regressive wrapper to automatically handle'
            # 更新参数中的输入掩码
            kwargs.update(input_mask = m[:, :-1])

        # 将输入序列传入网络得到输出
        out = self.net(xi, **kwargs)

        # 计算交叉熵损失
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        # 返回损失值
        return loss
```