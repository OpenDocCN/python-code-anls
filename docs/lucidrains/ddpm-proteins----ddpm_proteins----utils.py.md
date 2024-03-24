# `.\lucidrains\ddpm-proteins\ddpm_proteins\utils.py`

```py
# 导入所需的库
import os
from PIL import Image
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sidechainnet.utils.sequence import ProteinVocabulary
from einops import rearrange

# 通用函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 广播连接多个张量
def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

# 单例 MSA 转换器

msa_instances = None

def get_msa_transformer():
    global msa_instances
    if not exists(msa_instances):
        msa_model, alphabet = torch.hub.load("facebookresearch/esm", "esm_msa1_t12_100M_UR50S")
        batch_converter = alphabet.get_batch_converter()
        return msa_model, batch_converter
    return msa_instances

# MSA 嵌入相关函数

VOCAB = ProteinVocabulary()

# 将氨基酸 ID 转换为字符串
def ids_to_aa_str(x):
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    is_char = lambda c: isinstance(c, str) and len(c) == 1
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_aa_str(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(is_char, out)):
        return ''.join(out)

    return out

# 将氨基酸字符串转换为嵌入输入
def aa_str_to_embed_input(x):
    assert isinstance(x, list), 'input must be a list'
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(aa_str_to_embed_input(el))
        elif isinstance(el, str):
            out.append((None, el))
        else:
            raise TypeError('type must be either list or string')

    return out

# 对齐位置相关的函数
def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)
    normalized = x - avg
    return normalized

# 使矩阵对称
def symmetrize(x):
    return x + x.transpose(-1, -2)

# 将图像填充到指定大小
def pad_image_to(tensor, size, value = 0.):
    remainder = size - tensor.shape[-1]
    tensor = F.pad(tensor, (0, remainder, 0, remainder), value = value)
    return tensor

# 获取单个 MSA 注意力嵌入，带有缓存

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.ddpm-proteins'))
FETCH_FROM_CACHE = not exists(os.getenv('CLEAR_CACHE'))

os.makedirs(CACHE_PATH, exist_ok = True)

@torch.no_grad()
def get_msa_attention_embedding(
    model,
    batch_converter,
    aa_str,
    id,
    fetch_msas_fn = lambda t: [],
    cache = True
):
    device = next(model.parameters()).device

    cache_full_path = os.path.join(CACHE_PATH, f'{id}.pt')
    if cache and FETCH_FROM_CACHE and os.path.exists(cache_full_path):
        try:
            loaded = torch.load(cache_full_path).to(device)
        except:
            loaded = None

        if exists(loaded):
            return loaded

    msas = default(fetch_msas_fn(aa_str), [])
    seq_with_msas = [aa_str, *msas]

    embed_inputs = aa_str_to_embed_input(seq_with_msas)
    _, _, msa_batch_tokens = batch_converter(embed_inputs)
    # 使用模型对输入的批量 tokens 进行推理，需要获取注意力权重
    results = model(msa_batch_tokens.to(device), need_head_weights = True)

    # 从结果中获取行注意力权重
    attentions = results['row_attentions']
    # 剔除无效的位置信息
    attentions = attentions[..., 1:, 1:]
    # 重新排列注意力权重的维度
    attentions = rearrange(attentions, 'b l h m n -> b (l h) m n')
    # 对注意力权重进行对称化处理
    attentions = apc(symmetrize(attentions))

    # 如果需要缓存结果，则将结果保存到指定路径
    if cache:
        print(f'caching to {cache_full_path}')
        torch.save(attentions, cache_full_path)

    # 返回处理后的注意力权重
    return attentions
# 获取多序列对齐（MSA）的注意力嵌入
def get_msa_attention_embeddings(
    model,
    batch_converter,
    seqs,
    ids,
    fetch_msas_fn = lambda t: [],
    cache = True
):
    # 获取序列的长度
    n = seqs.shape[1]
    # 重新排列序列的维度
    seqs = rearrange(seqs, 'b n -> b () n')
    # 将序列 ID 转换为氨基酸字符串
    aa_strs = ids_to_aa_str(seqs.cpu().tolist())
    # 获取每个序列的注意力嵌入
    embeds_list = [get_msa_attention_embedding(model, batch_converter, aa, seq_id, cache = cache) for aa, seq_id in zip(aa_strs, ids)]
    # 将嵌入填充到相同长度
    embeds_list = [pad_image_to(embed, n) for embed in embeds_list]
    # 拼接所有嵌入
    embeds = torch.cat(embeds_list, dim = 0)
    return embeds

# 循环生成数据加载器
def cycle(loader, thres = 256):
    while True:
        for data in loader:
            # 如果序列长度小于阈值，则生成数据
            if data.seqs.shape[1] <= thres:
                yield data

# 保存热图
def save_heatmap(tensor, filepath, dpi = 200, return_image = False):
    # 生成热图
    heatmap = sn.heatmap(tensor.cpu().numpy())
    # 获取热图的图像
    figure = heatmap.get_figure()    
    # 保存热图到文件
    figure.savefig(filepath, dpi = dpi)
    # 清空图像
    plt.clf()

    # 如果不需要返回图像，则结束函数
    if not return_image:
        return
    # 返回图像对象
    return Image.open(filepath)
```