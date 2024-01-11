# `ZeroNet\src\util\Msgpack.py`

```
# 导入所需的模块
import os
import struct
import io

import msgpack
import msgpack.fallback


# 根据数据大小生成消息包头部
def msgpackHeader(size):
    if size <= 2 ** 8 - 1:
        return b"\xc4" + struct.pack("B", size)
    elif size <= 2 ** 16 - 1:
        return b"\xc5" + struct.pack(">H", size)
    elif size <= 2 ** 32 - 1:
        return b"\xc6" + struct.pack(">I", size)
    else:
        raise Exception("huge binary string")


# 将数据流写入指定的 writer
def stream(data, writer):
    packer = msgpack.Packer(use_bin_type=True)
    writer(packer.pack_map_header(len(data)))
    for key, val in data.items():
        writer(packer.pack(key))
        if isinstance(val, io.IOBase):  # File obj
            max_size = os.fstat(val.fileno()).st_size - val.tell()
            size = min(max_size, val.read_bytes)
            bytes_left = size
            writer(msgpackHeader(size))
            buff = 1024 * 64
            while 1:
                writer(val.read(min(bytes_left, buff)))
                bytes_left = bytes_left - buff
                if bytes_left <= 0:
                    break
        else:  # Simple
            writer(packer.pack(val))
    return size


# 定义 FilePart 类
class FilePart(object):
    __slots__ = ("file", "read_bytes", "__class__")

    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.__enter__ == self.file.__enter__

    def __getattr__(self, attr):
        return getattr(self.file, attr)

    def __enter__(self, *args, **kwargs):
        return self.file.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        return self.file.__exit__(*args, **kwargs)


# 不要尝试将这些字段的值解码为 utf8
bin_value_keys = ("hashfield_raw", "peers", "peers_ipv6", "peers_onion", "body", "sites", "bin")


# 对象解码钩子函数
def objectDecoderHook(obj):
    global bin_value_keys
    back = {}
    # 遍历对象中的键值对
    for key, val in obj:
        # 如果键的类型是字节流，则解码为 utf8 格式的字符串
        if type(key) is bytes:
            key = key.decode("utf8")
        # 如果键在指定的二进制值键集合中，或者值的类型不是字节流，或者键的长度大于等于 64，则将键值对添加到返回结果中
        if key in bin_value_keys or type(val) is not bytes or len(key) >= 64:
            back[key] = val
        # 否则，将值解码为 utf8 格式的字符串，然后添加到返回结果中
        else:
            back[key] = val.decode("utf8")
    # 返回处理后的结果
    return back
# 定义一个函数，用于获取消息解包器
def getUnpacker(fallback=False, decode=True):
    # 如果需要回退，则使用纯Python的解包器
    if fallback:
        unpacker = msgpack.fallback.Unpacker
    else:
        unpacker = msgpack.Unpacker

    # 设置额外的参数
    extra_kwargs = {"max_buffer_size": 5 * 1024 * 1024}
    # 如果消息包版本大于等于1，则设置严格的映射键为False
    if msgpack.version[0] >= 1:
        extra_kwargs["strict_map_key"] = False

    # 如果需要解码，则尝试将二进制解码为字符串
    if decode:
        unpacker = unpacker(raw=True, object_pairs_hook=objectDecoderHook, **extra_kwargs)
    else:
        unpacker = unpacker(raw=False, **extra_kwargs)

    # 返回解包器
    return unpacker


# 定义一个打包函数，用于将数据打包成消息
def pack(data, use_bin_type=True):
    return msgpack.packb(data, use_bin_type=use_bin_type)


# 定义一个解包函数，用于将消息解包成数据
def unpack(data, decode=True):
    # 获取解包器
    unpacker = getUnpacker(decode=decode)
    # 将数据输入到解包器中
    unpacker.feed(data)
    # 返回解包后的数据
    return next(unpacker)
```