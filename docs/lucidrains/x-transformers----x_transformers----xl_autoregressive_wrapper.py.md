# `.\lucidrains\x-transformers\x_transformers\xl_autoregressive_wrapper.py`

```
# 从 math 模块中导入 ceil 函数
from math import ceil

# 导入 torch 模块及相关子模块
import torch
from torch import nn
import torch.nn.functional as F

# 导入 einops 模块中的 rearrange, pack, unpack 函数
from einops import rearrange, pack, unpack
# 导入 x_transformers 模块中的 autoregressive_wrapper 模块中的 top_p, top_k, eval_decorator 函数

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 判断一个数是否能被另一个数整除的函数
def divisible_by(numer, denom):
    return (numer % denom) == 0 

# xl 自回归包装器类

class XLAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    # 生成序列的方法，使用 torch.no_grad() 修饰，eval_decorator 修饰
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        mems = None,
        **kwargs
    ):
        device, max_seq_len = start_tokens.device, self.max_seq_len

        start_tokens, ps = pack([start_tokens], '* n')

        b, t = start_tokens.shape

        *all_leading_tokens, _ = start_tokens.split(max_seq_len, dim = -1)

        # 捕获当前段的记忆

        for leading_tokens in all_leading_tokens:
            _, mems = self.net(
                leading_tokens,
                mems = mems,
                return_mems = True,
                **kwargs
            )

        # 现在开始从当前段进行采样

        curr_pos = len(all_leading_tokens) * max_seq_len
        curr_mems = mems

        cache = None
        out = start_tokens

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits, cache = self.net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            mems = cache.mems

            logits = logits[:, -1]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # 掩盖掉 eos 之后的所有内容
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        **kwargs
        ):
        # 从 self 中获取 ignore_index 和 max_seq_len 的值
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        # 将输入 x 的每一行的最后一个元素作为 labels，其余作为 x
        x, labels = x[:, :-1], x[:, 1:]

        # 获取 x 的序列长度
        seq_len = x.shape[1]

        # 准备分块数据

        # 将 x 和 labels 按照 max_seq_len 进行分块
        split_x = x.split(max_seq_len, dim = -1)
        split_labels = labels.split(max_seq_len, dim = -1)
        # 计算每个分块的损失权重
        loss_weights = tuple(map(lambda t: t.shape[-1] / seq_len, split_x))

        # 遍历每个分块并计算加权损失

        # 初始化总损失
        total_loss = 0.        

        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):

            # 在网络中传入当前分块数据，获取输出 logits 和记忆 mems
            logits, mems = self.net(
                chunk,
                mems = mems,
                return_mems = True,
                **kwargs
            )

            # 计算交叉熵损失
            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = ignore_index
            )

            # 累加加权损失
            total_loss = total_loss + loss * loss_weight

        # 返回总损失
        return total_loss
```