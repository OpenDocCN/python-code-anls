# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\cache_utils.py`

```py
# 导入必要的库
import os
from shutil import rmtree
import torch
import hashlib
from functools import wraps
from pathlib import Path

# 检查值是否存在的辅助函数
def exists(val):
    return val is not None

# 常量定义

# 设置缓存路径，默认为用户主目录下的.cache.tf.bind.transformer文件夹
CACHE_PATH = Path(os.getenv('TF_BIND_CACHE_PATH', os.path.expanduser('~/.cache.tf.bind.transformer')))
# 如果缓存路径不存在，则创建
CACHE_PATH.mkdir(exist_ok=True, parents=True)

# 检查是否需要清除缓存
CLEAR_CACHE = exists(os.getenv('CLEAR_CACHE', None))
# 检查是否需要输出详细信息
VERBOSE = exists(os.getenv('VERBOSE', None))

# 日志输出函数
def log(s):
    if not VERBOSE:
        return
    print(s)

# 计算字符串的 MD5 哈希值
def md5_hash_fn(s):
    encoded = s.encode('utf-8')
    return hashlib.md5(encoded).hexdigest()

# 仅运行一次的函数

# 全局运行记录字典
GLOBAL_RUN_RECORDS = dict()

# 仅运行一次的装饰器函数
def run_once(global_id=None):
    def outer(fn):
        has_ran_local = False
        output = None

        @wraps(fn)
        def inner(*args, **kwargs):
            nonlocal has_ran_local
            nonlocal output

            has_ran = GLOBAL_RUN_RECORDS.get(global_id, False) if exists(global_id) else has_ran_local

            if has_ran:
                return output

            output = fn(*args, **kwargs)

            if exists(global_id):
                GLOBAL_RUN_RECORDS[global_id] = True

            has_ran = True
            return output

        return inner
    return outer

# 缓存函数

# 缓存函数的装饰器
def cache_fn(
    fn,
    path='',
    hash_fn=md5_hash_fn,
    clear=False or CLEAR_CACHE,
    should_cache=True
):
    if not should_cache:
        return fn

    # 创建缓存路径
    (CACHE_PATH / path).mkdir(parents=True, exist_ok=True)

    # 清除缓存文件夹的函数
    @run_once(path)
    def clear_cache_folder_():
        cache_path = rmtree(str(CACHE_PATH / path))
        (CACHE_PATH / path).mkdir(parents=True, exist_ok=True)

    @wraps(fn)
    def inner(t, *args, __cache_key=None, **kwargs):
        if clear:
            clear_cache_folder_()

        cache_str = __cache_key if exists(__cache_key) else t
        key = hash_fn(cache_str)

        entry_path = CACHE_PATH / path / f'{key}.pt'

        if entry_path.exists():
            log(f'cache hit: fetching {t} from {str(entry_path)}')
            return torch.load(str(entry_path))

        out = fn(t, *args, **kwargs)

        log(f'saving: {t} to {str(entry_path)}')
        torch.save(out, str(entry_path))
        return out
    return inner
```