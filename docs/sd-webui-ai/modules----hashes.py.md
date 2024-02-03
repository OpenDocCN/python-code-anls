# `stable-diffusion-webui\modules\hashes.py`

```py
# 导入 hashlib 模块，用于计算哈希值
# 导入 os.path 模块，用于获取文件的修改时间
import hashlib
import os.path

# 从 modules 模块中导入 shared 和 cache 模块
from modules import shared
import modules.cache

# 从 modules.cache 模块中导入 dump_cache 和 cache 函数
dump_cache = modules.cache.dump_cache
cache = modules.cache.cache

# 计算文件的 SHA256 哈希值
def calculate_sha256(filename):
    # 创建 SHA256 哈希对象
    hash_sha256 = hashlib.sha256()
    # 定义块大小为 1MB
    blksize = 1024 * 1024

    # 打开文件并读取数据进行哈希计算
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    # 返回计算得到的 SHA256 哈希值的十六进制表示
    return hash_sha256.hexdigest()

# 从缓存中获取文件的 SHA256 哈希值
def sha256_from_cache(filename, title, use_addnet_hash=False):
    # 根据 use_addnet_hash 参数选择不同的缓存
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    # 获取文件的修改时间
    ondisk_mtime = os.path.getmtime(filename)

    # 如果 title 不在哈希表中，则返回 None
    if title not in hashes:
        return None

    # 从缓存中获取文件的 SHA256 哈希值和修改时间
    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)

    # 如果文件的修改时间晚于缓存中的修改时间，或者缓存中的哈希值为 None，则返回 None
    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None

    # 返回缓存中的 SHA256 哈希值
    return cached_sha256

# 计算文件的 SHA256 哈希值
def sha256(filename, title, use_addnet_hash=False):
    # 根据 use_addnet_hash 参数选择不同的缓存
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    # 从缓存中获取文件的 SHA256 哈希值
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    # 如果缓存中存在哈希值，则直接返回
    if sha256_value is not None:
        return sha256_value

    # 如果命令行参数中禁用了哈希计算，则返回 None
    if shared.cmd_opts.no_hashing:
        return None

    # 打印计算 SHA256 哈希值的过程
    print(f"Calculating sha256 for {filename}: ", end='')
    # 如果使用 addnet 哈希算法，则调用 addnet_hash_safetensors 函数计算哈希值
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    # 更新哈希表中的哈希值和修改时间
    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    # 将更新后的哈希表写入缓存
    dump_cache()

    # 返回计算得到的 SHA256 哈希值
    return sha256_value

# 计算 safetensors 的 addnet 哈希值
def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    # 创建 SHA256 哈希对象
    hash_sha256 = hashlib.sha256()
    # 定义块大小为 1MB
    blksize = 1024 * 1024

    # 将文件指针移动到开头并读取头部信息
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    # 计算偏移量
    offset = n + 8
    b.seek(offset)
    # 使用 lambda 函数创建一个迭代器，每次读取指定大小的数据块
    for chunk in iter(lambda: b.read(blksize), b""):
        # 更新 SHA-256 哈希对象，计算数据块的哈希值
        hash_sha256.update(chunk)

    # 返回 SHA-256 哈希对象的十六进制表示
    return hash_sha256.hexdigest()
```