# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\utils.py`

```py
# 导入必要的库
import os
import sys
import time
import pickle
import gzip
import torch
import contextlib
from functools import wraps, lru_cache
from filelock import FileLock
from einops import rearrange

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回唯一值列表
def uniq(arr):
    return list({el: True for el in arr}.keys())

# 返回指定阶数
def to_order(degree):
    return 2 * degree + 1

# 对字典的值应用函数
def map_values(fn, d):
    return {k: fn(v) for k, v in d.items()}

# 安全地拼接张量
def safe_cat(arr, el, dim):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim=dim)

# 将值转换为元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# 广播张量
def broadcat(tensors, dim=-1):
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
    return torch.cat(tensors, dim=dim)

# 批量索引选择
def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# 掩码均值
def masked_mean(tensor, mask, dim=-1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim=dim)
    mean = tensor.sum(dim=dim) / total_el.clamp(min=1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

# 生成均匀分布的随机张量
def rand_uniform(size, min_val, max_val):
    return torch.empty(size).uniform_(min_val, max_val)

# 快速分割张量
def fast_split(arr, splits, dim=0):
    axis_len = arr.shape[dim]
    splits = min(axis_len, max(splits, 1))
    chunk_size = axis_len // splits
    remainder = axis_len - chunk_size * splits
    s = 0
    for i in range(splits):
        adjust, remainder = 1 if remainder > 0 else 0, remainder - 1
        yield torch.narrow(arr, dim, s, chunk_size + adjust)
        s += chunk_size + adjust

# 傅立叶编码
def fourier_encode(x, num_encodings=4, include_self=True, flatten=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    x = rearrange(x, 'b m n ... -> b m n (...)') if flatten else x
    return x

# 默认数据类型上下文管理器
@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# 转换为 torch 张量的装饰器
def cast_torch_tensor(fn):
    @wraps(fn)
    # 定义一个内部函数 inner，接受一个参数 t
    def inner(t):
        # 如果 t 不是 torch 的张量，则将 t 转换为 torch 的张量，数据类型为默认数据类型
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        # 调用外部函数 fn，并传入处理后的张量 t
        return fn(t)
    # 返回内部函数 inner
    return inner
# benchmark 工具函数，用于计算函数执行时间
def benchmark(fn):
    # 内部函数，记录函数执行时间并返回结果
    def inner(*args, **kwargs):
        # 记录开始时间
        start = time.time()
        # 执行函数
        res = fn(*args, **kwargs)
        # 计算时间差
        diff = time.time() - start
        # 返回时间差和结果
        return diff, res
    return inner

# 缓存函数装饰器
def cache(cache, key_fn):
    # 内部函数，实现缓存功能
    def cache_inner(fn):
        @wraps(fn)
        # 内部函数，检查缓存并返回结果
        def inner(*args, **kwargs):
            # 生成缓存键名
            key_name = key_fn(*args, **kwargs)
            # 如果缓存中存在键名，则直接返回结果
            if key_name in cache:
                return cache[key_name]
            # 否则执行函数并将结果存入缓存
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# 在目录中进行缓存
def cache_dir(dirname, maxsize=128):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''
    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        # 内部函数，实现目录缓存功能
        def wrapper(*args, **kwargs):
            # 如果目录不存在，则直接执行函数
            if not exists(dirname):
                return func(*args, **kwargs)

            # 创建目录
            os.makedirs(dirname, exist_ok=True)

            indexfile = os.path.join(dirname, "index.pkl")
            lock = FileLock(os.path.join(dirname, "mutex"))

            with lock:
                index = {}
                # 如果索引文件存在，则加载索引
                if os.path.exists(indexfile):
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                key = (args, frozenset(kwargs), func.__defaults__)

                # 如果键存在于索引中，则获取文件名
                if key in index:
                    filename = index[key]
                else:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            # 如果文件存在，则加载结果
            if os.path.exists(filepath):
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            print(f"compute {filename}... ", end="", flush=True)
            result = func(*args, **kwargs)
            print(f"save {filename}... ", end="", flush=True)

            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            print("done")

            return result
        return wrapper
    return decorator
```