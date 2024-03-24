# `.\lucidrains\toolformer-pytorch\toolformer_pytorch\toolformer_pytorch.py`

```
# 导入所需的库
import re

from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, reduce

# 导入自定义模块
from toolformer_pytorch.palm import PaLM
from toolformer_pytorch.optimizer import get_optimizer
from toolformer_pytorch.prompts import DEFAULT_PROMPT_INPUT_TAG

# 导入类型提示相关库
from beartype import beartype
from beartype.typing import Callable, Optional, Union, List, Tuple

from tqdm import tqdm
from x_clip.tokenizer import tokenizer

# 设置 pad_sequence 函数的 batch_first 参数为 True
pad_sequence = partial(pad_sequence, batch_first = True)

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回输入值
def identity(t):
    return t

# 返回固定值的函数
def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

# 尝试执行函数，捕获异常并执行回调函数
def try_except(fn, callback = identity):
    @wraps(fn)
    def inner(*args):
        try:
            return fn(*args)
        except Exception as e:
            return callback(e)
    return inner

# 张量操作函数

# 对数函数
def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 生成 Gumbel 分布采样
def gumbel_sample(t, temperature = 1., dim = -1, eps = 1e-10):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim)

# 保留前 k 个最大值，其余设为负无穷
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, indices = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, indices, val)
    return probs

# 检查张量是否包含指定值
def all_contains_id(t: torch.Tensor, token_id: int):
    mask = t == token_id
    return mask.any(dim = -1).all()

# 查找指定值在张量中的索引
def find_indices_of(t: torch.Tensor, token_id: int, occurrence = 1):
    assert occurrence > 0
    mask = (t == token_id)

    has_occurred = mask.cumsum(dim = -1)
    has_occurred = F.pad(has_occurred, (1, 0), value = 0.)

    return (has_occurred < occurrence).sum(dim = -1).long()

# 调用 API 调用函数

# 检查字符串是否为有效格式
def is_valid_string(s):
    return exists(re.fullmatch(r"'[^']*'|\"[^\"]*\"", s))

# 检查整数是否为有效格式
def is_valid_integer(s):
    return exists(re.fullmatch(r"[+-]?\d+", s))

# 检查浮点数是否为有效格式
def is_valid_float(s):
    return exists(re.fullmatch(r"[+-]?\d+(\.\d+)?", s))

# 解析参数字符串为整数、浮点数或字符串
def parse_param(s: str) -> Optional[Union[int, float, str]]:
    if is_valid_string(s):
        return str(s)
    elif is_valid_integer(s):
        return int(s)
    elif is_valid_float(s):
        return float(s)

    return None

# 替换函数，根据注册的函数执行相应的 API 调用
@beartype
def replace_fn(
    registry: dict[str, Callable],
    matches,
    delimiter = '→'
):
    orig_text = matches.group(0)

    text_without_end_api_token = matches.group(1)
    end_api_token = matches.group(4)
    function_name = matches.group(2)

    # 如果注册表中找不到函数，则返回原始文本
    if function_name not in registry:
        return orig_text

    fn = registry[function_name]

    params = matches.group(3).split(',')
    params = list(map(lambda s: s.strip(), params))
    params = list(filter(len, params))
    params = list(map(parse_param, params))

    # 如果参数中有无法解析的部分，则返回原始文本
    if any([(not exists(p)) for p in params]):
        return orig_text

    # 尝试执行函数，如果出现错误则返回 None
    out = try_except(fn, always(None))(*params)

    # 如果输出为 None，则返回原始文本
    if not exists(out):
        return orig_text

    # 返回带有输出分隔符和字符串化输出的原始文本
    return f'{text_without_end_api_token} {delimiter} {str(out)} {end_api_token}'

# 主函数，接受函数注册表、文本和进行 API 调用并附加输出
def create_function_regex(
    api_start = ' [',
    api_stop = ']'
):
    # 将 api_start 和 api_stop 进行转义，得到转义后的正则表达式字符串
    api_start_regex, api_stop_regex = map(re.escape, (api_start, api_stop))
    # 返回一个包含转义后的 api_start 和 api_stop 的正则表达式字符串
    return rf'({api_start_regex}(\w+)\(([^)]*)\))({api_stop_regex})'
# 计算子字符串在文本中出现的次数
def num_matches(substr: str, text: str):
    return len(re.findall(re.escape(substr), text))

# 检查文本中是否存在 API 调用
def has_api_calls(
    text,
    api_start = ' [',
    api_stop = ']'
):
    # 创建 API 调用的正则表达式
    regex = create_function_regex(api_start, api_stop)
    # 查找匹配的 API 调用
    matches = re.findall(regex, text)
    return len(matches) > 0

# 替换除第一个外的所有 API 调用
def replace_all_but_first(
    text: str,
    api_start = ' [',
    api_stop = ']'
) -> str:
    # 创建 API 调用的正则表达式
    regex = create_function_regex(api_start, api_stop)

    count = 0

    def replace_(matches):
        orig_text = matches.group(0)
        nonlocal count
        count += 1
        if count > 1:
            return ''
        return orig_text

    return re.sub(regex, replace_, text)

# 在文本中调用工具函数
def invoke_tools(
    registry: dict[str, Callable],
    text: str,
    delimiter: str = '→',
    api_start = ' [',
    api_stop = ' ]'
) -> str:
    # 创建 API 调用的正则表达式
    regex = create_function_regex(api_start, api_stop)
    replace_ = partial(replace_fn, registry, delimiter = delimiter)
    return re.sub(regex, replace_, text)

# 在批量序列上调用工具函数
def invoke_tools_on_batch_sequences(
    registry: dict[str, Callable],
    token_ids: torch.Tensor,
    *,
    encode: Callable,
    decode: Callable,
    delimiter: str = '→',
    api_start = ' [',
    api_stop = ']'
) -> torch.Tensor:
    regex = create_function_regex(api_start_regex, api_stop_regex)
    all_texts = [decode(one_seq_token_ids) for one_seq_token_ids in token_ids]

    invoke_tools_ = partial(invoke_tools, api_start = api_start, api_stop = api_stop)
    all_texts_with_api_calls = [invoke_tools_(registry, text, delimiter) for text in all_texts]

    return encode(all_texts_with_api_calls)

# 采样 API 相关函数
# 它们进行贪婪采样，但通过在前 k = 10 中自动选择 <api> 标记来鼓励采样 API 调用

@beartype
@torch.no_grad()
def sample(
    model: nn.Module,
    *,
    seq_len,
    prime: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    batch_size = 1,
    eos_token_id = None,
    sos_token_id = 1,
    temperature = 0.,
    pad_id = 0,
    call_api_only_once = False,
    api_start_token_id = None,
    auto_select_api_start_token_when_topk = False,
    select_api_start_id_top_k = 10,
):
    device = next(model.parameters()).device
    max_seq_len = seq_len + 1

    # 验证

    if call_api_only_once:
        assert exists(api_start_token_id)

    # 初始化

    if exists(prime):
        batch_size, prime_length = prime.shape
    else:
        prime_length = 1
        prime = torch.full((batch_size, 1), sos_token_id, device = device, dtype = torch.long)

    prime = prime.to(device)

    # 采样位置 - 不同序列有不同的游标

    if exists(positions):
        positions = positions.clone()
    else:
        positions = torch.zeros((batch_size,), device = device, dtype = torch.long)

    assert (positions <= (prime_length + 1)).all() and (positions <= max_seq_len).all(), '所有位置必须小于初始主长度以及总序列长度 + 1（如果一个序列在另一个序列之前完成采样，则加一）'

    # 评估模型

    model.eval()

    # 将主长度延长到整个序列长度

    remain_iterations = seq_len - prime_length
    output = F.pad(prime, (0, max_seq_len - prime_length), value = 0.)

    batch_indices = torch.arange(batch_size, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    position_indices = rearrange(positions, 'b -> b 1')

    # 确定 <api> 标记掩码，以确保只调用一次 API，屏蔽对数以防止它被选择为那些已经包含 <api> 标记的行

    api_token_mask = None # 懒惰创建，因为不知道对数维度

    def create_api_token_mask(num_tokens, api_start_token_id):
        mask = torch.zeros((1, 1, num_tokens), dtype = torch.bool)
        assert api_start_token_id < num_tokens
        mask[..., api_start_token_id] = True
        return mask

    # 开始迭代
    # 对于剩余的迭代次数，循环执行以下操作
    for iteration in tqdm(range(remain_iterations):
        # 使用模型预测输出
        logits = model(output)
        # 获取最后一个位置的logits
        last_logits = logits[batch_indices, position_indices]

        # 确保每个批次的令牌序列最多只有一个<api>令牌
        if call_api_only_once:
            # 如果api_token_mask不存在，则创建一个
            if not exists(api_token_mask):
                num_tokens = last_logits.shape[-1]
                api_token_mask = create_api_token_mask(num_tokens, api_start_token_id)
                api_token_mask = api_token_mask.to(device)

            # 检查是否调用了api
            api_called = (output == api_start_token_id).any(dim=-1)

            # 创建logit_mask，用于标记需要被替换的位置
            logit_mask = api_token_mask & rearrange(api_called, 'b -> b 1 1')
            last_logits = last_logits.masked_fill(logit_mask, -torch.finfo(last_logits.dtype).max)

        # 使用贪婪采样（也可以选择非贪婪）
        sampled = gumbel_sample(last_logits, temperature=temperature)

        # 对于那些没有api调用的序列，如果api_start_token_id在logits的前k个（设置为10）中，则自动选择
        if auto_select_api_start_token_when_topk:
            top_token_ids = last_logits.topk(select_api_start_id_top_k, dim=-1).indices
            has_api_token_in_topk = (top_token_ids == api_start_token_id).any(dim=-1)
            should_auto_select_api_token = has_api_token_in_topk & ~rearrange(api_called, 'b -> b 1')

            sampled = sampled.masked_fill(should_auto_select_api_token, api_start_token_id)

        # 将采样的令牌放置在正确的光标位置
        output[batch_indices, position_indices] = sampled

        # 增加位置索引
        position_indices += 1
        position_indices.clamp_(max=seq_len)  # 如果一个序列更靠后且接近结尾，则不执行任何操作

        # 如果使用<eos>令牌，查找所有包含它的序列并终止，<eos>之后的内容将被填充
        if exists(eos_token_id):
            eos_mask = (output == eos_token_id)
            all_rows_have_eos = eos_mask.any(dim=-1).all()

            if all_rows_have_eos:
                keep_mask = eos_mask.cumsum(dim=-1) == 0
                keep_mask = F.pad(keep_mask, (1, 0), value=True)
                output = output.masked_fill(~keep_mask, pad_id)
                break

    # 移除输出中的最后一个令牌（作为无操作占位符）
    output = output[:, :-1]
    return output
# 使用 beartype 装饰器对函数进行类型检查
# 使用 torch.no_grad() 上下文管理器，禁用梯度计算
@beartype
@torch.no_grad()
# 从模型中生成序列，调用 API 并返回结果
def sample_with_api_call(
    model: nn.Module,
    *,
    seq_len,  # 序列长度
    call_apis: Callable,  # 调用 API 的函数
    prime: torch.Tensor,  # 初始张量
    api_end_token_id: int,  # API 结束标记的 ID
    occurrence = 1,  # API 出现次数
    **kwargs  # 其他关键字参数
):
    # 生成初始序列
    sampled = sample(
        model = model,
        prime = prime,
        seq_len = seq_len,
        **kwargs
    )

    # 调用 API 处理生成的序列
    sampled = call_apis(sampled)

    # 获取处理后序列的长度
    sampled_seq_len = sampled.shape[-1]
    null_positions = sampled_seq_len  # 处理不包含 API 调用的序列

    # 查找 API 结束标记的位置
    pos_starting_at_end_of_api = find_indices_of(
        sampled,
        api_end_token_id,
        occurrence = occurrence
    )

    # 重新生成序列，从 API 结束位置开始
    resample_after_api_calls = sample(
        model = model,
        prime = sampled,
        seq_len = sampled_seq_len,
        positions = (pos_starting_at_end_of_api + 1).clamp(max = null_positions),  # 从 </api> 后的位置开始
        **kwargs
    )

    return resample_after_api_calls

# 论文的主要贡献在于第 2 节中提出的过滤方程

# 默认的权重函数
def default_weight_fn(t):
    # 根据第 4.1 节中的公式计算权重，不确定分母中的 w_s 是什么
    # 如果 t 代表每个时间步，则在 5 个标记内会减少到 0？
    return (1. - t * 0.2).clamp(min = 0.)

# 获取预测概率
def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]  # 每个标记的 logits（省略最后一个 logits）
    token_ids = token_ids[:, 1:]  # 预测下一个标记的 ID（省略第一个标记的 ID）

    token_ids = rearrange(token_ids, 'b n -> b n 1')
    probs = logits.softmax(dim = -1)
    correct_token_id_pred_prob = probs.gather(-1, token_ids)
    return rearrange(correct_token_id_pred_prob, 'b n 1 -> b n')

# 获取从特定标记开始的索引
def get_arange_start_at_token_id(
    token_ids: torch.Tensor,
    token_id: int,
    pad_id = -1
):
    is_token_id_mask = token_ids == token_id
    arange = (is_token_id_mask.cumsum(dim = -1) > 0).cumsum(dim = -1)
    before_token_mask = arange == 0
    arange = arange - 1
    arange = arange.masked_fill(before_token_mask, pad_id)
    return arange

# 计算权重和掩码
def weight_and_mask(
    token_ids: torch.Tensor,
    token_id: int,
    pad_id = -1,
    weighting_fn: Callable = default_weight_fn
):
    t = get_arange_start_at_token_id(token_ids, token_id, pad_id)
    weights = weighting_fn(t)
    return weights.masked_fill(t == pad_id, 0.)

# 定义过滤结果的命名元组
FilteredResults = namedtuple('FilteredResults', [
    'num_passed',
    'num_failed',
    'selected_indices',
    'selected_mask',
    'filtered_tokens',
    'filtered_tokens_without_api_response',
    'filtered_tokens_with_api_response'
])

# 过滤带有 API 响应的标记
@beartype
def filter_tokens_with_api_response(
    model: nn.Module,  # 语言模型应接受下面的标记并返回形状为 (batch, seq, num tokens) 的 logits
    *,
    tokens: torch.Tensor,  # 原始段落的标记 ID（不包含 API 调用）
    tokens_without_api_response: torch.Tensor,  # 包含 API 调用但没有填充响应的段落的标记 ID - <api>tool1(x, y)</api>
    tokens_with_api_response: torch.Tensor,  # 包含 API 调用和响应的段落的标记 ID - <api>tool1(x, y) → {response}</api>
    api_start_token_id: int,  # <api> 标记的 ID
    api_end_token_id: int,  # </api> 标记的 ID
    filter_threshold: float = 1.,  # 接受采样的 API 调用的阈值（tokens_with_api_response）用于微调
    weighting_fn: Callable = default_weight_fn  # 权重函数
) -> FilteredResults:

    # 验证

    assert all([*map(lambda t: t.dtype == torch.long, (tokens, tokens_with_api_response, tokens_without_api_response))])

    assert all_contains_id(tokens_without_api_response, api_start_token_id)
    assert all_contains_id(tokens_without_api_response, api_end_token_id)
    # 确保所有的 tokens_with_api_response 中包含 api_start_token_id
    assert all_contains_id(tokens_with_api_response, api_start_token_id)
    # 确保所有的 tokens_with_api_response 中包含 api_end_token_id
    assert all_contains_id(tokens_with_api_response, api_end_token_id)

    # 自动设置设备

    # 获取模型参数的设备
    device = next(model.parameters()).device
    # 将 tokens, tokens_without_api_response, tokens_with_api_response 移动到指定设备上
    tokens, tokens_without_api_response, tokens_with_api_response = map(lambda t: t.to(device), (tokens, tokens_without_api_response, tokens_with_api_response))

    # 获取所有的 logits

    with torch.no_grad():
        # 设置模型为评估模式
        model.eval()
        # 获取 logits, logits_without_api_response, logits_with_api_response
        logits, logits_without_api_response, logits_with_api_response = map(model, (tokens, tokens_without_api_response, tokens_with_api_response))

    # 推导出序列中实际下一个 token id 的所有预测概率

    probs                       = get_pred_prob(tokens, logits)
    probs_without_api_response  = get_pred_prob(tokens_without_api_response, logits_without_api_response)
    probs_with_api_response     = get_pred_prob(tokens_with_api_response, logits_with_api_response)

    weight_and_mask_fn = partial(weight_and_mask, weighting_fn = weighting_fn)

    # 推导权重

    weight_without_api_response = weight_and_mask_fn(tokens_without_api_response[:, :-1], api_end_token_id)
    weight_with_api_response = weight_and_mask_fn(tokens_with_api_response[:, :-1], api_end_token_id)

    # 推导原始 passage 的权重更加复杂
    # 需要从 <api> 开始标记的位置开始计数
    # 这也假设语言模型完美地复制了 passage，并且两个 token id 对齐，除了插入的 API 调用 - 但最终可以通过自定义过滤函数完成

    weight = weight_and_mask_fn(tokens_without_api_response[:, 1:], api_start_token_id) # 左移一个位置，因为原始序列中不存在 <api>
    weight = weight[:, :probs.shape[-1]]

    # 获取所有三种序列的损失 L

    def loss_fn(weight, probs):
        return (weight * -log(probs)).sum(dim = -1)

    loss = loss_fn(weight, probs)
    loss_without_api_response = loss_fn(weight_without_api_response, probs_without_api_response)
    loss_with_api_response = loss_fn(weight_with_api_response, probs_with_api_response)

    # 计算论文中的主要公式

    # loss+ = 带有 api 响应的损失
    # loss- = 最小值(没有 api 响应的损失, 没有 api 的损失)

    loss_plus = loss_with_api_response
    loss_minus = torch.minimum(loss_without_api_response, loss)

    selected_mask = (loss_minus - loss_plus) >= filter_threshold

    # 现在我们可以选择并返回经过过滤阶段幸存下来的条目
    # 同时返回正在处理的批次的选定索引
    # 用于将模型微调为 toolformer

    batch = tokens.shape[0]
    indices = torch.arange(batch, device = tokens.device)

    selected_indices = indices[selected_mask]

    ret = FilteredResults(
        selected_mask.sum().item(),
        (~selected_mask).sum().item(),
        selected_indices,
        selected_mask,
        tokens[selected_mask],
        tokens_without_api_response[selected_mask],
        tokens_with_api_response[selected_mask]
    )

    return ret
# datasets and dataloaders

# 用于通过 API 调用引导初始数据集以及最终微调

# 定义 PromptDataset 类，继承自 Dataset 类
@beartype
class PromptDataset(Dataset):
    # 初始化方法
    def __init__(
        self,
        prompt: str,
        prompt_input_tag: str,
        data: List[str],
        tokenizer_encode: Callable
    ):
        # 初始化数据集、提示、提示输入标签的正则表达式、编码器
        self.data = data
        self.prompt = prompt
        self.prompt_input_tag_regex = re.escape(prompt_input_tag)
        self.tokenizer_encode = tokenizer_encode

    # 返回数据集长度
    def __len__(self):
        return len(self.data)

    # 获取指定索引的数据
    def __getitem__(self, idx):
        data_string = self.data[idx]
        data_with_prompt = re.sub(self.prompt_input_tag_regex, data_string, self.prompt)
        token_ids = self.tokenizer_encode(data_with_prompt)
        return torch.tensor(token_ids).long(), torch.tensor(len(token_ids)).long()

# 定义 prompt_collate_fn 函数，用于数据集的填充
def prompt_collate_fn(data, padding_value = 0):
    prompts, prompt_lengths = zip(*data)
    prompts = pad_sequence(prompts, padding_value = padding_value)
    return prompts, torch.stack(prompt_lengths)

# 定义 PromptDataloader 函数，用于创建数据加载器
def PromptDataloader(ds: Dataset, *args, padding_value = 0, **kwargs):
    collate_fn = partial(prompt_collate_fn, padding_value = padding_value)
    return DataLoader(ds, *args, collate_fn = collate_fn, **kwargs)

# 定义 FinetuneDataset 类，继承自 Dataset 类
class FinetuneDataset(Dataset):
    # 初始化方法
    def __init__(
        self,
        tokens: torch.Tensor
    ):
        # 初始化 tokens
        self.tokens = tokens

    # 返回数据集长度
    def __len__(self):
        return len(self.tokens)

    # 获取指定索引的数据
    def __getitem__(self, idx):
        return self.tokens[idx]

# 定义 FinetuneDataloader 函数，用于创建微调数据加载器
def FinetuneDataloader(ds: Dataset, *args, padding_value = 0, **kwargs):
    return DataLoader(ds, *args, collate_fn = partial(pad_sequence, padding_value = padding_value), **kwargs)

# classes

# 定义 Toolformer 类，继承自 nn.Module 类
@beartype
class Toolformer(nn.Module):
    # 初始化方法
    def __init__(
        self,
        model: nn.Module,
        *,
        tool_id: str,
        tool: Callable,
        api_start_str = ' [',
        api_stop_str = ']',
        api_response_delimiter = '→',
        api_start_id = None,
        api_stop_id = None,
        teach_tool_prompt: str,
        filter_threshold = 1.,
        pad_id = 0,
        prompt_batch_size = 4,
        model_seq_len = 2048,
        tokenizer_encode: Callable = tokenizer.encode,
        tokenizer_decode: Callable = tokenizer.decode,
        post_prompt_callback: Callable = identity,
        prompt_input_tag: str = DEFAULT_PROMPT_INPUT_TAG,
        exclude_filters: dict[str, Callable[[str], bool]] = dict(),
        finetune = False,
        finetune_lr = 1e-4,
        finetune_wd = 1e-2,
        finetune_betas = (0.9, 0.99),
        finetune_eps = 1e-8,
        finetune_epochs = 3,
        finetune_batch_size = 16
    # 初始化函数，设置模型、模型序列长度、教学工具提示、提示批量大小、提示输入标签等参数
    ):
        super().__init__()
        self.model = model
        self.model_seq_len = model_seq_len

        self.teach_tool_prompt = teach_tool_prompt
        self.prompt_batch_size = prompt_batch_size
        self.prompt_input_tag = prompt_input_tag

        self.post_prompt_callback = post_prompt_callback # for easy mocking

        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.tokenizer_encode_to_tensor = lambda s: torch.tensor(tokenizer_encode(s)).long()

        self.filter_threshold = filter_threshold

        self.api_start_str = api_start_str
        self.api_stop_str = api_stop_str
        self.api_response_delimiter = api_response_delimiter

        # 如果不存在api_start_id，则根据api_start_str进行编码
        if not exists(api_start_id):
            api_start_id = tokenizer_encode(api_start_str)
            assert len(api_start_id) == 1
            api_start_id = api_start_id[0]

        self.api_start_id = api_start_id

        # 如果不存在api_stop_id，则根据api_stop_str进行编码
        if not exists(api_stop_id):
            api_stop_id = tokenizer_encode(api_stop_str)
            assert len(api_stop_id) == 1
            api_stop_id = api_stop_id[0]

        self.api_stop_id = api_stop_id

        self.pad_id = pad_id

        self.tool_id = tool_id
        self.tool = tool
        self.registry = {tool_id: tool}

        # 确保在提示中只有一个指定的提示输入标签
        assert num_matches(prompt_input_tag, teach_tool_prompt) == 1, f'there must be exactly one prompt input tag `{prompt_input_tag}` in your prompt to encourage the language model to use the designated tool'

        self.teach_tool_prompt = teach_tool_prompt
        self.exclude_filters = exclude_filters

        self.should_finetune = finetune

        # 如果不需要微调，则直接返回
        if not finetune:
            return

        self.finetune_batch_size = finetune_batch_size
        self.finetune_epochs = finetune_epochs

        # 获取优化器
        self.optimizer = get_optimizer(
            model.parameters(),
            lr = finetune_lr,
            wd = finetune_wd,
            betas = finetune_betas,
            eps = finetune_eps
        )

    # 生成带有API调用的数据
    def generate_data_with_api_calls(
        self,
        data: List[str],
        temperature: float = 0.9
    ) -> List[str]:

        # 创建PromptDataset对象
        dataset = PromptDataset(
            data = data,
            prompt_input_tag = self.prompt_input_tag,
            prompt = self.teach_tool_prompt,
            tokenizer_encode = self.tokenizer_encode
        )

        # 创建PromptDataloader对象
        dl = PromptDataloader(
            dataset,
            batch_size = self.prompt_batch_size
        )

        prompted_outputs = []

        # 遍历数据加载器
        for prime, positions in dl:

            # 对模型进行采样
            sampled_outputs = sample(
                model = self.model,
                prime = prime,
                positions = positions,
                seq_len = self.model_seq_len,
                pad_id = self.pad_id,
                temperature = temperature
            )

            # 解码采样输出并添加到结果列表中
            for sample_output, position in zip(sampled_outputs, positions):
                start_position = position.item()

                prompted_output = self.tokenizer_decode(sample_output[start_position:])
                prompted_outputs.append(prompted_output)

        # 调用后处理回调函数
        return self.post_prompt_callback(prompted_outputs)

    # 过滤并仅保留第一个API调用
    def filter_and_keep_only_first_api_call(
        self,
        data,
        data_with_api_calls: List[str],
        return_excluded = False
    # 初始化包含数据和包含 API 调用数据的空列表
    included_data = []
    included_data_with_api_calls = []

    # 将包含数据和包含 API 调用数据组成元组
    included = (included_data, included_data_with_api_calls)

    # 初始化排除数据和排除 API 调用数据的空列表
    excluded_data = []
    excluded_data_with_api_calls = []

    # 将排除数据和排除 API 调用数据组成元组
    excluded = (excluded_data, excluded_data_with_api_calls)

    # 设置 API 调用开始和结束参数
    api_start_stop_kwargs = dict(api_start=self.api_start_str, api_stop=self.api_stop_str)

    # 创建部分函数，用于检查是否存在 API 调用和替换除第一个外的所有 API 调用
    has_api_calls_ = partial(has_api_calls, **api_start_stop_kwargs)
    replace_all_but_first_ = partial(replace_all_but_first, **api_start_stop_kwargs)

    # 遍历数据和数据中包含 API 调用的元组
    for datum, data_with_api_call in zip(data, data_with_api_calls):
        # 如果数据中包含 API 调用
        if has_api_calls_(data_with_api_call):
            # 替换除第一个外的所有 API 调用
            data_with_api_call = replace_all_but_first_(data_with_api_call)

            # 将数据和数据中包含 API 调用添加到包含列表中
            included_data.append(datum)
            included_data_with_api_calls.append(data_with_api_call)
        else:
            # 将数据和数据中包含 API 调用添加到排除列表中
            excluded_data.append(datum)
            excluded_data_with_api_calls.append(data_with_api_call)

    # 如果不返回排除数据，则返回包含数据
    if not return_excluded:
        return included

    # 返回包含数据和排除数据
    return included, excluded

@torch.no_grad()
def sample_model_with_api_calls(
    self,
    prime: Union[torch.Tensor, str],
    occurrence=1,
    **kwargs
):
    # 将模型设置为评估模式
    self.model.eval()

    # 检查 prime 是否为字符串
    prime_is_str = isinstance(prime, str)

    # 如果 prime 是字符串
    if prime_is_str:
        # 对 prime 进行编码和转换为张量
        prime = self.tokenizer_encode(prime)
        prime = torch.tensor(prime).long()
        prime = rearrange(prime, 'n -> 1 n')

    # 断言 prime 的形状为 (1, n)
    assert prime.shape[0] == 1, 'only one at a time for now'

    # 创建部分函数，用于调用工具函数
    invoke_tools_ = partial(invoke_tools, self.registry)

    # 定义调用 API 函数
    def call_apis(t: torch.Tensor):
        t = self.tokenizer_decode(t[0])
        t = invoke_tools_(t)
        t = self.tokenizer_encode_to_tensor(t)
        return rearrange(t, 'n -> 1 n')

    # 使用带有 API 调用的模型进行采样
    output = sample_with_api_call(
        model=self.model,
        prime=prime,
        seq_len=self.model_seq_len,
        call_apis=call_apis,
        api_end_token_id=self.api_stop_id,
        occurrence=occurrence,
        **kwargs
    )

    # 如果 prime 不是字符串，则返回输出
    if not prime_is_str:
        return output

    # 将输出解码为字符串并返回
    return self.tokenizer_decode(output[0])

# 执行 API 调用
def make_api_calls(
    self,
    filtered_data_with_api_calls: List[str]
):
    # 创建部分函数，用于调用工具函数
    invoke_tools_ = partial(
        invoke_tools,
        self.registry,
        api_start=self.api_start_str,
        api_stop=self.api_stop_str,
        delimiter=self.api_response_delimiter
    )

    # 对过滤后的数据进行 API 调用
    data_with_api_responses = []
    for data in filtered_data_with_api_calls:
        output = invoke_tools_(data)
        data_with_api_responses.append(output)

    # 返回包含 API 响应的数据
    return data_with_api_responses

# 根据 API 响应过滤数据
def filter_by_api_responses(
    self,
    data: List[str],
    data_with_api_calls: List[str],
    data_with_api_responses: List[str]
) -> FilteredResults:

    # 定义将列表转换为张量的函数
    to_token_ids = lambda l: pad_sequence([*map(self.tokenizer_encode_to_tensor, l)], padding_value=self.pad_id)

    # 将数据转换为张量
    tokens, tokens_without_api_response, tokens_with_api_response = map(to_token_ids, (data, data_with_api_calls, data_with_api_responses))

    # 过滤带有 API 响应的结果
    filtered_results = filter_tokens_with_api_response(
        model=self.model,
        tokens=tokens,
        tokens_with_api_response=tokens_with_api_response,
        tokens_without_api_response=tokens_without_api_response,
        filter_threshold=self.filter_threshold,
        api_start_token_id=self.api_start_id,
        api_end_token_id=self.api_stop_id
    )

    # 返回过滤后的结果
    return filtered_results

# 微调模型
def finetune(
    self,
    filtered_results: Union[FilteredResults, torch.Tensor]
    # 设置模型为训练模式
    ):
        self.model.train()

        # 如果filtered_results是FilteredResults类型，则将其转换为没有API响应的过滤后结果
        if isinstance(filtered_results, FilteredResults):
            filtered_results = filtered_results.filtered_tokens_without_api_response

        # 创建用于微调的数据集
        dataset = FinetuneDataset(tokens = filtered_results)
        # 创建用于微调的数据加载器
        dl = FinetuneDataloader(dataset, batch_size = self.finetune_batch_size, shuffle = True)

        # 循环微调epochs次数
        for epoch in tqdm(range(self.finetune_epochs), desc = 'finetune epochs'):
            # 遍历数据加载器中的每个批次
            for batch in dl:
                # 将输入和标签分别赋值为批次中的前n-1列和最后一列
                inp, labels = batch[:, :-1], batch[:, 1:]

                # 使用模型进行前向传播
                logits = self.model(inp)
                # 重新排列logits的维度
                logits = rearrange(logits, 'b n c -> b c n')

                # 计算交叉熵损失
                loss = F.cross_entropy(logits, labels, ignore_index = self.pad_id)
                # 反向传播计算梯度
                loss.backward()

                # 打印损失值
                print(f'loss: {loss.item()}')
                # 更新优化器参数
                self.optimizer.step()
                # 梯度清零
                self.optimizer.zero_grad()

        # 打印微调结束信息
        print(f'finished finetuning on {len(dataset)} filtered samples')

    # 前向传播函数
    def forward(
        self,
        data: List[str],
        return_after_generating_api_calls = False,
        return_after_making_api_calls = False,
        return_after_filtering_api_calls = False,
        return_after_filtering_by_api_response = False
    ):
        # 生成带有API调用的数据
        data_with_api_calls = self.generate_data_with_api_calls(data)

        # 如果需要在生成API调用后返回数据，则直接返回
        if return_after_generating_api_calls:
            return data_with_api_calls

        # 过滤数据并保留第一个API调用
        filtered_data, filtered_data_with_api_calls = self.filter_and_keep_only_first_api_call(data, data_with_api_calls)

        # 如果需要在过滤API调用后返回数据，则直接返回
        if return_after_filtering_api_calls:
            return filtered_data, filtered_data_with_api_calls

        # 断言过滤后的数据中至少有一个API调用
        assert len(filtered_data_with_api_calls) > 0, 'your model failed to follow instructions and make API calls. please try a better model or do some better prompt engineering'

        # 进行API调用
        data_with_responses = self.make_api_calls(filtered_data_with_api_calls)

        # 如果需要在进行API调用后返回数据，则直接返回
        if return_after_making_api_calls:
            return filtered_data, filtered_data_with_api_calls, data_with_responses

        # 根据API响应过滤数据
        filtered_results = self.filter_by_api_responses(filtered_data, filtered_data_with_api_calls, data_with_responses)

        # 如果需要在根据API响应过滤数据后返回数据，则直接返回
        if return_after_filtering_by_api_response:
            return filtered_results

        # 如果需要微调模型
        if self.should_finetune:
            # 断言通过API调用的数据数量大于0
            assert filtered_results.num_passed > 0, f'none of the sequences with API calls passed the filtering criteria with threshold {self.filter_threshold}'

            # 进行��调
            self.finetune(filtered_results)

        # 返回过滤后的结果
        return filtered_results
```