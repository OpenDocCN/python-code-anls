# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\utils\utils.py`

```
import torch
from einops import repeat, rearrange

def average_over_durations(values, durs):
    """
        - in:
            - values: B, 1, T_de
            - durs: B, T_en
        - out:
            - avg: B, 1, T_en
    """
    # 计算累积持续时间的结束位置
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    # 计算累积持续时间的开始位置
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    # 计算非零值的累积
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    # 计算值的累积
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    # 重复持续时间的开始位置
    dcs = repeat(durs_cums_starts, 'bs l -> bs n l', n=n_formants)
    # 重复持续时间的结束位置
    dce = repeat(durs_cums_ends, 'bs l -> bs n l', n=n_formants)

    # 计算值的总和
    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).to(values.dtype)
    # 计算值的元素个数
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).to(values.dtype)

    # 计算平均值
    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems).to(values.dtype)
    return avg

def create_mask(sequence_length, max_len):
    dtype, device = sequence_length.dtype, sequence_length.device
    # 创建一个序列范围
    seq_range = torch.arange(max_len, dtype=dtype, device=device)
    sequence_length = rearrange(sequence_length, 'b -> b 1')
    seq_range = rearrange(seq_range, 't -> 1 t')
    return seq_range < sequence_length
```