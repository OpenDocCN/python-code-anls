# `.\lucidrains\CALM-pytorch\CALM_pytorch\sampling_utils.py`

```
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import Tensor  # 导入 PyTorch 中的张量
from torch.nn import Module  # 导入 PyTorch 中的神经网络模块
from torch.nn.utils.rnn import pad_sequence  # 导入 PyTorch 中的序列填充函数

from beartype import beartype  # 导入 beartype 库中的类型检查装饰器
from beartype.typing import Optional, Callable, List, Tuple  # 导入 beartype 库中的类型注解

from einops import rearrange  # 导入 einops 库中的重排函数

from tqdm import tqdm  # 导入 tqdm 库中的进度条显示函数

def exists(v):  # 定义函数，判断变量是否存在
    return v is not None  # 返回变量是否不为 None

def default(v, d):  # 定义函数，返回变量或默认值
    return v if exists(v) else d  # 如果变量存在则返回变量，否则返回默认值

# 采样辅助函数

def log(t, eps = 1e-20):  # 定义函数，计算张量的对数
    return torch.log(t.clamp(min = eps))  # 返回张量的对数，避免小于 eps 的值

def gumbel_noise(t):  # 定义函数，生成 Gumbel 噪声
    noise = torch.zeros_like(t).uniform_(0, 1)  # 生成与输入张量相同大小的均匀分布噪声
    return -log(-log(noise))  # 返回 Gumbel 噪声

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True, eps = 1e-10):  # 定义函数，使用 Gumbel 分布进行采样
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)  # 返回 Gumbel 采样结果

# nucleus

def top_p(logits, thres = 0.9):  # 定义函数，根据 top-p 策略进行筛选
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)  # 对 logits 进行降序排序
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)  # 计算累积概率

    sorted_indices_to_remove = cum_probs > thres  # 根据阈值筛选需要移除的索引
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)  # 对需要移除的索引进行填充

    sorted_logits[sorted_indices_to_remove] = float('-inf')  # 将需要移除的 logits 置为负无穷
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)  # 返回根据 top-p 策略筛选后的 logits

# topk

def top_k(logits, frac_num_tokens = 0.1, k: Optional[int] = None):  # 定义函数，根据 top-k 策略进行筛选
    num_tokens = logits.shape[-1]  # 获取 logits 的最后一个维度大小

    k = default(k, ceil(frac_num_tokens * num_tokens))  # 计算 k 值
    k = min(k, num_tokens)  # 取 k 和 num_tokens 中的较小值

    val, ind = torch.topk(logits, k)  # 获取 top-k 的值和索引
    probs = torch.full_like(logits, float('-inf'))  # 创建与 logits 相同大小的全为负无穷的张量
    probs.scatter_(1, ind, val)  # 根据 top-k 的索引和值填充 probs
    return probs  # 返回根据 top-k 策略筛选后的 logits

# 解码

@torch.no_grad()  # 禁用梯度计算
@beartype  # 使用 beartype 类型检查装饰器
def sample(  # 定义函数，生成序列样本
    net: Module,  # 神经网络模型
    prompts,  # 输入的提示序列
    seq_len: int,  # 生成序列的长度
    temperature = 1.,  # 温度参数
    filter_fn: Callable = top_p,  # 筛选函数，默认为 top-p
    filter_kwargs: dict = dict(),  # 筛选函数的参数
    pad_id: int = -1,  # 填充标识符
    eos_id: Optional[int] = None,  # 结束标识符
    output_keep_prompt = False  # 是否保留提示序列
):
    device = next(net.parameters()).device  # 获取模型参数所在的设备
    net.eval()  # 设置模型为评估模式

    if isinstance(prompts, (tuple, list)):  # 如果提示序列是元组或列表
        prompts = pad_sequence(prompts, batch_first = True, padding_value = pad_id)  # 对提示序列进行填充

    batch, prompts_tensor_len = prompts.shape  # 获取提示序列的形状信息

    batch_arange = torch.arange(batch, device = device)[..., None]  # 创建批次索引张量

    prompt_lens = (prompts != pad_id).sum(dim = -1)  # 计算提示序列的长度
    curr_seq_indices = prompt_lens[..., None]  # 当前序列索引

    out = prompts.clone()  # 克隆提示序列作为输出序列

    pbar = tqdm(  # 创建进度条
        initial = out.shape[-1],  # 初始值
        total = seq_len,  # 总步数
        desc = 'sampling'  # 描述
    )

    while (curr_seq_indices < seq_len).any():  # 当当前序列索引小于生成序列长度时循环
        out = F.pad(out, (0, 1), value = pad_id)  # 对输出序列进行填充

        net_input = out.masked_fill(out == pad_id, 0)  # 将填充值替换为 0

        logits = net(net_input)  # 输入网络获取 logits

        logits = logits[batch_arange, curr_seq_indices]  # 根据当前序列索引获取 logits
        logits = rearrange(logits, 'b 1 d -> b d')  # 重排 logits 的维度

        logits = filter_fn(logits, **filter_kwargs)  # 根据筛选函数筛选 logits
        sampled_tokens = gumbel_sample(logits, temperature = temperature, dim = -1)  # 使用 Gumbel 采样获取 tokens

        out[batch_arange, curr_seq_indices] = sampled_tokens  # 更新输出序列

        curr_seq_indices += 1  # 当前序列索引加一
        curr_seq_indices.clamp_(max = seq_len)  # 限制当前序列索引的最大值为生成序列长度
        pbar.update(1)  # 更新进度条

        if not exists(eos_id):  # 如果结束标识符不存在
            continue  # 继续下一次循环

        is_eos_mask = out == eos_id  # 获取结束标识符的掩码
        all_eos = is_eos_mask.any(dim = -1).all()  # 判断是否所有序列都包含结束标识符

        if all_eos:  # 如果所有序列都包含结束标识符
            break  # 跳出循环

    pbar.close()  # 关闭进度条

    out = out[:, :seq_len]  # 截取生成序列的长度

    if exists(eos_id):  # 如果结束标识符存在
        is_eos_mask = out == eos_id  # 获取结束标识符的掩码
        after_eos_mask = F.pad(is_eos_mask.cumsum(dim = -1) > 0, (1, -1), value = False)  # 获取结束标识符后的掩码
        out = out.masked_fill_(after_eos_mask, pad_id)  # 将结束标识符后的位置填充为填充标识符

    if output_keep_prompt:  # 如果需要保留提示序列
        return out  # 返回输出序列

    prompt_mask = torch.arange(out.shape[-1], device = device) < prompt_lens[..., None]  # 创建提示序列的掩码

    generated_seq_mask = out != pad_id & ~prompt_mask  # 生成序列的掩码
    seq_lens = generated_seq_mask.sum(dim = -1).tolist()  # 计算生成序列的长度

    return out[generated_seq_mask].split(seq_lens)  # 返回根据生成序列掩码拆分后的结果
```