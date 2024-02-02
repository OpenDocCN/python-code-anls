# `ZeroNet\src\lib\pyaes\util.py`

```py
# 定义一个函数，用于将二进制数据转换为可缓冲的对象
def to_bufferable(binary):
    return binary

# 定义一个函数，用于将字符转换为对应的字节值
def _get_byte(c):
    return ord(c)

# 检查是否存在 xrange 函数，如果不存在则定义 to_bufferable 和 _get_byte 函数
try:
    xrange
except:
    # 如果输入是字节对象，则直接返回，否则将输入的字符转换为字节对象
    def to_bufferable(binary):
        if isinstance(binary, bytes):
            return binary
        return bytes(ord(b) for b in binary)

    # 将字符转换为对应的字节值
    def _get_byte(c):
        return c

# 定义一个函数，用于在数据末尾添加 PKCS7 填充
def append_PKCS7_padding(data):
    # 计算需要填充的字节数
    pad = 16 - (len(data) % 16)
    # 在数据末尾添加填充字节
    return data + to_bufferable(chr(pad) * pad)

# 定义一个函数，用于去除数据末尾的 PKCS7 填充
def strip_PKCS7_padding(data):
    # 如果数据长度不是 16 的倍数，则抛出数值错误
    if len(data) % 16 != 0:
        raise ValueError("invalid length")
    # 获取填充的字节值
    pad = _get_byte(data[-1])
    # 如果填充大于16个字节，则抛出数值错误异常
    if pad > 16:
        raise ValueError("invalid padding byte")
    
    # 返回去除填充的数据
    return data[:-pad]
```