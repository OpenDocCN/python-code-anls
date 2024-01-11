# `ZeroNet\src\util\Electrum.py`

```
# 导入 hashlib 模块，用于生成哈希值
# 导入 struct 模块，用于处理字节流
import hashlib
import struct


# 定义函数 bchr，将整数转换为字节流
def bchr(i):
    return struct.pack("B", i)

# 定义函数 encode，将给定值按照指定进制编码，并补齐到指定长度
def encode(val, base, minlen=0):
    base, minlen = int(base), int(minlen)
    # 生成包含 0 到 255 的字节流
    code_string = b"".join([bchr(x) for x in range(256)])
    result = b""
    # 将值按照指定进制编码
    while val > 0:
        index = val % base
        result = code_string[index:index + 1] + result
        val //= base
    # 补齐到指定长度
    return code_string[0:1] * max(minlen - len(result), 0) + result

# 定义函数 insane_int，将整数转换为字节流
def insane_int(x):
    x = int(x)
    if x < 253:
        return bchr(x)
    elif x < 65536:
        return bchr(253) + encode(x, 256, 2)[::-1]
    elif x < 4294967296:
        return bchr(254) + encode(x, 256, 4)[::-1]
    else:
        return bchr(255) + encode(x, 256, 8)[::-1]

# 定义函数 magic，对消息进行特殊处理，添加前缀并转换为字节流
def magic(message):
    return b"\x18Bitcoin Signed Message:\n" + insane_int(len(message)) + message

# 定义函数 format，对消息进行哈希处理，返回哈希值
def format(message):
    return hashlib.sha256(magic(message)).digest()

# 定义函数 dbl_format，对消息进行两次哈希处理，返回哈希值
def dbl_format(message):
    return hashlib.sha256(format(message)).digest()
```