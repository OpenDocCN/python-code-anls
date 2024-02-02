# `ZeroNet\plugins\Bigfile\BigfilePiecefield.py`

```py
# 导入 array 模块

import array


# 封装数据为一个紧凑的二进制表示
def packPiecefield(data):
    # 如果数据不是 bytes 或者 bytearray 类型，则抛出异常
    if not isinstance(data, bytes) and not isinstance(data, bytearray):
        raise Exception("Invalid data type: %s" % type(data))

    res = []
    # 如果数据为空，则返回一个空的 array
    if not data:
        return array.array("H", b"")

    # 如果数据的第一个字节是 b"\x00"
    if data[0] == b"\x00":
        res.append(0)
        find = b"\x01"
    else:
        find = b"\x00"
    last_pos = 0
    pos = 0
    # 循环查找数据中的特定字节序列
    while 1:
        pos = data.find(find, pos)
        if find == b"\x00":
            find = b"\x01"
        else:
            find = b"\x00"
        if pos == -1:
            res.append(len(data) - last_pos)
            break
        res.append(pos - last_pos)
        last_pos = pos
    # 返回封装后的 array
    return array.array("H", res)


# 解析紧凑的二进制表示为数据
def unpackPiecefield(data):
    # 如果数据为空，则返回空字节串
    if not data:
        return b""

    res = []
    char = b"\x01"
    # 根据紧凑表示还原数据
    for times in data:
        if times > 10000:
            return b""
        res.append(char * times)
        if char == b"\x01":
            char = b"\x00"
        else:
            char = b"\x01"
    # 返回还原后的数据
    return b"".join(res)


# 在指定位置插入一个比特位
def spliceBit(data, idx, bit):
    # 如果比特位不是 b"\x00" 或者 b"\x01"，则抛出异常
    if bit != b"\x00" and bit != b"\x01":
        raise Exception("Invalid bit: %s" % bit)

    # 如果数据长度小于指定位置，则在末尾填充 b"\x00"
    if len(data) < idx:
        data = data.ljust(idx + 1, b"\x00")
    # 在指定位置插入比特位
    return data[:idx] + bit + data[idx+ 1:]


# 定义 Piecefield 类
class Piecefield(object):
    # 将数据转换为字符串表示
    def tostring(self):
        return "".join(["1" if b else "0" for b in self.tobytes()])


# 定义 BigfilePiecefield 类，继承自 Piecefield 类
class BigfilePiecefield(Piecefield):
    # 限定 BigfilePiecefield 类的属性为 data
    __slots__ = ["data"]

    # 初始化方法，将 data 属性初始化为空字节串
    def __init__(self):
        self.data = b""

    # 将字节串转换为数据
    def frombytes(self, s):
        # 如果 s 不是 bytes 或者 bytearray 类型，则抛出异常
        if not isinstance(s, bytes) and not isinstance(s, bytearray):
            raise Exception("Invalid type: %s" % type(s))
        self.data = s

    # 将数据转换为字节串
    def tobytes(self):
        return self.data

    # 封装数据
    def pack(self):
        return packPiecefield(self.data).tobytes()

    # 解析数据
    def unpack(self, s):
        self.data = unpackPiecefield(array.array("H", s))
    # 定义特殊方法，用于获取对象中指定键对应的值
    def __getitem__(self, key):
        # 尝试从数据中获取指定键对应的值
        try:
            return self.data[key]
        # 如果指定键不存在，捕获 IndexError 异常
        except IndexError:
            # 返回 False
            return False

    # 定义特殊方法，用于设置对象中指定键对应的值
    def __setitem__(self, key, value):
        # 调用 spliceBit 函数，将指定键对应的值替换为新值
        self.data = spliceBit(self.data, key, value)
# 定义一个名为 BigfilePiecefieldPacked 的类，继承自 Piecefield 类
class BigfilePiecefieldPacked(Piecefield):
    # 定义类的特殊属性 __slots__，限定实例只能拥有 "data" 属性
    __slots__ = ["data"]

    # 定义类的初始化方法
    def __init__(self):
        # 初始化实例的 "data" 属性为空字节串
        self.data = b""

    # 定义从字节串转换为实例数据的方法
    def frombytes(self, data):
        # 如果输入数据不是字节串或字节数组，则抛出异常
        if not isinstance(data, bytes) and not isinstance(data, bytearray):
            raise Exception("Invalid type: %s" % type(data))
        # 将输入数据打包成 Piecefield 对象，再转换为字节串，并赋值给实例的 "data" 属性
        self.data = packPiecefield(data).tobytes()

    # 定义将实例数据转换为字节串的方法
    def tobytes(self):
        # 将实例的 "data" 属性解包成无符号短整型数组，并返回
        return unpackPiecefield(array.array("H", self.data))

    # 定义将实例数据打包成字节串的方法
    def pack(self):
        # 将实例的 "data" 属性打包成字节串，并返回
        return array.array("H", self.data).tobytes()

    # 定义将输入数据赋值给实例的 "data" 属性的方法
    def unpack(self, data):
        # 将输入数据赋值给实例的 "data" 属性
        self.data = data

    # 定义获取实例数据中指定位置的值的方法
    def __getitem__(self, key):
        try:
            # 尝试从实例数据中获取指定位置的值，并返回
            return self.tobytes()[key]
        except IndexError:
            # 如果发生索引错误，则返回 False
            return False

    # 定义设置实例数据中指定位置的值的方法
    def __setitem__(self, key, value):
        # 将实例数据中指定位置的值替换为输入的值，并更新实例的 "data" 属性
        data = spliceBit(self.tobytes(), key, value)
        self.frombytes(data)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 导入 os 模块
    import os
    # 导入 psutil 模块
    import psutil
    # 导入 time 模块
    import time
    # 定义测试数据
    testdata = b"\x01" * 100 + b"\x00" * 900 + b"\x01" * 4000 + b"\x00" * 4999 + b"\x01"
    # 获取当前进程的内存信息
    meminfo = psutil.Process(os.getpid()).memory_info
    # 遍历存储类型列表，依次进行测试
    for storage in [BigfilePiecefieldPacked, BigfilePiecefield]:
        # 打印当前测试的存储类型
        print("-- Testing storage: %s --" % storage)
        # 获取当前内存使用情况
        m = meminfo()[0]
        # 记录当前时间
        s = time.time()
        # 创建空字典用于存储测试数据
        piecefields = {}
        # 循环10000次，创建存储对象并存储测试数据
        for i in range(10000):
            piecefield = storage()
            piecefield.frombytes(testdata[:i] + b"\x00" + testdata[i + 1:])
            piecefields[i] = piecefield

        # 打印创建10000个对象所增加的内存和耗时
        print("Create x10000: +%sKB in %.3fs (len: %s)" % ((meminfo()[0] - m) / 1024, time.time() - s, len(piecefields[0].data)))

        # 获取当前内存使用情况
        m = meminfo()[0]
        # 记录当前时间
        s = time.time()
        # 循环查询10000次，获取指定位置的值
        for piecefield in list(piecefields.values()):
            val = piecefield[1000]

        # 打印查询10000次所增加的内存和耗时
        print("Query one x10000: +%sKB in %.3fs" % ((meminfo()[0] - m) / 1024, time.time() - s))

        # 获取当前内存使用情况
        m = meminfo()[0]
        # 记录当前时间
        s = time.time()
        # 循环修改10000次，设置指定位置的值
        for piecefield in list(piecefields.values()):
            piecefield[1000] = b"\x01"

        # 打印修改10000次所增加的内存和耗时
        print("Change one x10000: +%sKB in %.3fs" % ((meminfo()[0] - m) / 1024, time.time() - s))

        # 获取当前内存使用情况
        m = meminfo()[0]
        # 记录当前时间
        s = time.time()
        # 循环打包10000次
        for piecefield in list(piecefields.values()):
            packed = piecefield.pack()

        # 打印打包10000次所增加的内存和耗时，以及打包后的长度
        print("Pack x10000: +%sKB in %.3fs (len: %s)" % ((meminfo()[0] - m) / 1024, time.time() - s, len(packed)))

        # 获取当前内存使用情况
        m = meminfo()[0]
        # 记录当前时间
        s = time.time()
        # 循环解包10000次
        for piecefield in list(piecefields.values()):
            piecefield.unpack(packed)

        # 打印解包10000次所增加的内存和耗时，以及解包后的长度
        print("Unpack x10000: +%sKB in %.3fs (len: %s)" % ((meminfo()[0] - m) / 1024, time.time() - s, len(piecefields[0].data)))

        # 清空测试数据字典
        piecefields = {}
```