# `.\lucidrains\self-rewarding-lm-pytorch\self_rewarding_lm_pytorch\sampling_utils.py`

```py
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import Tensor  # 导入 PyTorch 中的张量
from torch.nn import Module  # 导入 PyTorch 中的神经网络模块
from torch.nn.utils.rnn import pad_sequence  # 导入 PyTorch 中的序列填充函数

from beartype import beartype  # 导入 beartype 库中的类型检查装饰器
from beartype.typing import Optional, Callable, List, Tuple  # 导入 beartype 库中的类型注解

from einops import rearrange  # 导入 einops 库中的重排函数

def exists(v):  # 定义函数，判断变量是否存在
    return v is not None

def default(v, d):  # 定义函数，如果变量存在则返回变量，否则返回默认值
    return v if exists(v) else d

# 采样辅助函数

def log(t, eps = 1e-20):  # 定义函数，计算张量的对数
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):  # 定义函数，生成 Gumbel 噪声
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True, eps = 1e-10):  # 定义函数，进行 Gumbel 采样
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# nucleus

def top_p(logits, thres = 0.9):  # 定义函数，根据 top-p 算法进行筛选
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k: Optional[int] = None):  # 定义函数，根据 top-k 算法进行筛选
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 解码

@torch.no_grad()  # 禁用梯度计算
@beartype  # 使用 beartype 类型检查装饰器
def sample(  # 定义采样函数
    net: Module,  # 神经网络模型
    prompts,  # 输入的提示序列
    seq_len: int,  # 生成序列的长度
    temperature = 1.,  # 温度参数
    filter_fn: Callable = top_p,  # 筛选函数，默认为 top_p
    filter_kwargs: dict = dict(),  # 筛选函数的参数
    pad_id: int = -1,  # 填充标识���
    eos_id: Optional[int] = None,  # 结束标识符
    output_keep_prompt = False  # 是否保留提示序列
):
    device = next(net.parameters()).device  # 获取神经网络模型的设备
    net.eval()  # 设置神经网络模型为评估模式

    if isinstance(prompts, (tuple, list)):  # 如果提示序列是元组或列表
        prompts = pad_sequence(prompts, batch_first = True, padding_value = pad_id)  # 对提示序列进行填充

    batch, prompts_tensor_len = prompts.shape  # 获取提示序列的批次和长度

    batch_arange = torch.arange(batch, device = device)[..., None]  # 生成批次索引

    prompt_lens = (prompts != pad_id).sum(dim = -1)  # 计算提示序列的长度
    curr_seq_indices = prompt_lens[..., None]  # 当前序列索引

    out = prompts.clone()  # 克隆提示序列

    while (curr_seq_indices < seq_len).any():  # 当当前序列索引小于生成序列长度时循环
        out = F.pad(out, (0, 1), value = pad_id)  # 对输出序列进行填充

        net_input = out.masked_fill(out == pad_id, 0)  # 将填充部分替换为零

        logits = net(net_input)  # 输入神经网络模型，获取输出 logits

        logits = logits[batch_arange, curr_seq_indices]  # 获取当前序列的 logits
        logits = rearrange(logits, 'b 1 d -> b d')  # 重排 logits 的维度

        logits = filter_fn(logits, **filter_kwargs)  # 使用筛选函数对 logits 进行筛选
        sampled_tokens = gumbel_sample(logits, temperature = temperature, dim = -1)  # 使用 Gumbel 采样获取生成的 token

        out[batch_arange, curr_seq_indices] = sampled_tokens  # 将生成的 token 放入输出序列

        curr_seq_indices += 1  # 当前序列索引加一
        curr_seq_indices.clamp_(max = seq_len)  # 限制当前序列索引的最大值为生成序列长度

        if not exists(eos_id):  # 如果不存在结束标识符
            continue  # 继续循环

        is_eos_mask = out == eos_id  # 获取结束标识符的掩码
        all_eos = is_eos_mask.any(dim = -1).all()  # 判断是否所有序列都包含结束标识符

        if all_eos:  # 如果所有序列都包含结束标识符
            break  # 跳出循环

    out = out[:, :seq_len]  # 截取生成的序列长度为指定长度

    if exists(eos_id):  # 如果存在结束标识符
        is_eos_mask = out == eos_id  # 获取结束标识符的掩码
        after_eos_mask = F.pad(is_eos_mask.cumsum(dim = -1) > 0, (1, -1), value = False)  # 获取结束标识符后的掩码
        out = out.masked_fill_(after_eos_mask, pad_id)  # 将结束标识符后的部分替换为填充标识符

    if output_keep_prompt:  # 如果需要保留提示序列
        return out  # 返回生成的序列

    prompt_mask = torch.arange(out.shape[-1], device = device) < prompt_lens[..., None]  # 生成提示序列的掩码

    generated_seq_mask = out != pad_id & ~prompt_mask  # 生成序列的掩码
    seq_lens = generated_seq_mask.sum(dim = -1).tolist()  # 计算生成序列的长度

    return out[generated_seq_mask].split(seq_lens)  # 返回拆分后的生成序列
```