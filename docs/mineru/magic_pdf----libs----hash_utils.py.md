# `.\MinerU\magic_pdf\libs\hash_utils.py`

```
# 导入 hashlib 模块以使用哈希函数
import hashlib


# 定义一个函数，计算输入字节的 MD5 值
def compute_md5(file_bytes):
    # 创建一个 MD5 哈希对象
    hasher = hashlib.md5()
    # 更新哈希对象以包含文件字节
    hasher.update(file_bytes)
    # 返回 MD5 哈希值的十六进制表示，转为大写
    return hasher.hexdigest().upper()


# 定义一个函数，计算输入字符串的 SHA-256 值
def compute_sha256(input_string):
    # 创建一个 SHA-256 哈希对象
    hasher = hashlib.sha256()
    # 将输入字符串编码为 UTF-8 字节对象
    input_bytes = input_string.encode('utf-8')
    # 更新哈希对象以包含输入字节
    hasher.update(input_bytes)
    # 返回 SHA-256 哈希值的十六进制表示
    return hasher.hexdigest()
```