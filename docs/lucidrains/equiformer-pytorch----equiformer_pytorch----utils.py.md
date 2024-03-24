# `.\lucidrains\equiformer-pytorch\equiformer_pytorch\utils.py`

```
# 导入必要的库
from pathlib import Path

import time
import pickle
import gzip

import torch
import torch.nn.functional as F

import contextlib
from functools import wraps, lru_cache
from filelock import FileLock
from equiformer_pytorch.version import __version__

from einops import rearrange

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t):
    return t

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将度数转换为顺序
def to_order(degree):
    return 2 * degree + 1

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 为了使 y 居中于 x，对 y 进行填充
def pad_for_centering_y_to_x(x, y):
    assert y <= x
    total_pad = x - y
    assert (total_pad % 2) == 0
    return total_pad // 2

# 为了使 y 居中于 x，对 y 进行切片
def slice_for_centering_y_to_x(x, y):
    pad = pad_for_centering_y_to_x(x, y)
    if pad == 0:
        return slice(None)
    return slice(pad, -pad)

# 安全地拼接张量
def safe_cat(arr, el, dim):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim = dim)

# 将值转换为元组
def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

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

# 带掩码的均值计算
def masked_mean(tensor, mask, dim = -1):
    if not exists(mask):
        return tensor.mean(dim = dim)

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

# 生成指定范围内的随机均匀分布张量
def rand_uniform(size, min_val, max_val):
    return torch.empty(size).uniform_(min_val, max_val)

# 默认数据类型上下文管理器

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# 将输入转换为 torch 张量
def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype = torch.get_default_dtype())
        return fn(t)
    return inner

# 基准测试工具

def benchmark(fn):
    def inner(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        diff = time.time() - start
        return diff, res
    return inner

# 缓存函数

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# 在目录中缓存

def cache_dir(dirname, maxsize=128):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''
    # 定义一个装饰器函数，接受一个函数作为参数
    def decorator(func):

        # 使用 lru_cache 装饰器缓存函数的结果，设置最大缓存大小为 maxsize
        @lru_cache(maxsize=maxsize)
        # 使用 wraps 装饰器保留原始函数的元数据
        @wraps(func)
        # 定义一个包装函数，接受任意参数和关键字参数
        def wrapper(*args, **kwargs):
            # 如果目录不存在，则直接调用原始函数并返回结果
            if not exists(dirname):
                return func(*args, **kwargs)

            # 创建目录路径对象
            dirpath = Path(dirname)
            # 创建目录，如果不存在则创建，存在则忽略
            dirpath.mkdir(parents=True, exist_ok=True)

            # 创建索引文件路径对象
            indexfile = dirpath / 'index.pkl'
            # 创建文件锁对象
            lock = FileLock(str(dirpath / 'mutex'))

            # 使用文件锁
            with lock:
                # 初始化索引字典
                index = {}
                # 如果索引文件存在，则读取索引数据
                if indexfile.exists():
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                # 生成键值，用于索引
                key = (args, frozenset(kwargs), func.__defaults__)

                # 如果键值存在于索引中，则获取对应的文件名
                if key in index:
                    filename = index[key]
                else:
                    # 如果键值不存在于索引中，则生成新的文件名，并更新索引
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            # 生成文件路径对象
            filepath = dirpath / filename

            # 如果文件存在，则使用文件锁读取文件数据
            if filepath.exists():
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            # 打印计算过程信息
            print(f"compute {filename}... ", end="", flush=True)
            # 调用原始函数计算结果
            result = func(*args, **kwargs)
            # 打印保存文件信息
            print(f"save {filename}... ", end="", flush=True)

            # 使用文件锁保存计算结果到文件
            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            # 打印完成信息
            print("done")

            return result
        # 返回包装函数
        return wrapper
    # 返回装饰器函数
    return decorator
```