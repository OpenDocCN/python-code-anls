# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_streams.py`

```
""" Testing

"""

# 导入必要的库
import os  # 提供与操作系统交互的功能
import zlib  # 提供压缩和解压缩功能

from io import BytesIO  # 提供内存中流的处理功能

# 导入必要的测试库和函数
from tempfile import mkstemp  # 创建临时文件
from contextlib import contextmanager  # 上下文管理器，用于创建资源的临时上下文

import numpy as np  # 提供数组处理功能

from numpy.testing import assert_, assert_equal  # 测试断言函数
from pytest import raises as assert_raises  # 测试异常断言函数

from scipy.io.matlab._streams import (make_stream,  # 导入流处理相关的函数和类
    GenericStream, ZlibInputStream,
    _read_into, _read_string, BLOCK_SIZE)


@contextmanager
def setup_test_file():
    # 准备测试文件的数据
    val = b'a\x00string'  # 测试用的字节数据
    fd, fname = mkstemp()  # 创建临时文件

    with os.fdopen(fd, 'wb') as fs:
        fs.write(val)  # 将数据写入临时文件
    with open(fname, 'rb') as fs:
        gs = BytesIO(val)  # 创建字节流对象
        cs = BytesIO(val)  # 创建字节流对象
        yield fs, gs, cs  # 返回文件句柄和两个字节流对象
    os.unlink(fname)  # 删除临时文件


def test_make_stream():
    # 测试 make_stream 函数
    with setup_test_file() as (fs, gs, cs):
        # 测试流对象的初始化
        assert_(isinstance(make_stream(gs), GenericStream))


def test_tell_seek():
    # 测试 tell 和 seek 方法
    with setup_test_file() as (fs, gs, cs):
        for s in (fs, gs, cs):
            st = make_stream(s)  # 创建流对象
            res = st.seek(0)  # 将指针移动到文件开头
            assert_equal(res, 0)  # 验证 seek 方法返回值
            assert_equal(st.tell(), 0)  # 验证 tell 方法返回值
            res = st.seek(5)  # 将指针移动到第 5 个字节处
            assert_equal(res, 0)
            assert_equal(st.tell(), 5)
            res = st.seek(2, 1)  # 从当前位置向后移动 2 个字节
            assert_equal(res, 0)
            assert_equal(st.tell(), 7)
            res = st.seek(-2, 2)  # 从文件末尾向前移动 2 个字节
            assert_equal(res, 0)
            assert_equal(st.tell(), 6)


def test_read():
    # 测试 read 方法和相关辅助函数
    with setup_test_file() as (fs, gs, cs):
        for s in (fs, gs, cs):
            st = make_stream(s)  # 创建流对象
            st.seek(0)  # 将指针移动到文件开头
            res = st.read(-1)  # 读取所有数据
            assert_equal(res, b'a\x00string')  # 验证读取的数据内容
            st.seek(0)  # 重新将指针移动到文件开头
            res = st.read(4)  # 读取前 4 个字节
            assert_equal(res, b'a\x00st')  # 验证读取的数据内容
            # 使用 _read_into 函数读取数据
            st.seek(0)
            res = _read_into(st, 4)
            assert_equal(res, b'a\x00st')
            res = _read_into(st, 4)
            assert_equal(res, b'ring')
            assert_raises(OSError, _read_into, st, 2)  # 验证异常情况
            # 使用 _read_string 函数读取数据
            st.seek(0)
            res = _read_string(st, 4)
            assert_equal(res, b'a\x00st')
            res = _read_string(st, 4)
            assert_equal(res, b'ring')
            assert_raises(OSError, _read_string, st, 2)  # 验证异常情况


class TestZlibInputStream:
    def _get_data(self, size):
        # 生成随机数据，并使用 zlib 压缩
        data = np.random.randint(0, 256, size).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data)
        stream = BytesIO(compressed_data)  # 创建压缩数据的字节流对象
        return stream, len(compressed_data), data
    # 定义一个测试方法 test_read，用于测试读取功能
    def test_read(self):
        # 定义文件大小的列表
        SIZES = [0, 1, 10, BLOCK_SIZE//2, BLOCK_SIZE-1,
                 BLOCK_SIZE, BLOCK_SIZE+1, 2*BLOCK_SIZE-1]

        # 定义读取大小的列表
        READ_SIZES = [BLOCK_SIZE//2, BLOCK_SIZE-1,
                      BLOCK_SIZE, BLOCK_SIZE+1]

        # 定义内部函数 check，用于检查读取功能
        def check(size, read_size):
            # 获取压缩流、压缩数据长度和原始数据
            compressed_stream, compressed_data_len, data = self._get_data(size)
            # 创建 ZlibInputStream 对象
            stream = ZlibInputStream(compressed_stream, compressed_data_len)
            # 初始化空数据变量
            data2 = b''
            so_far = 0
            # 循环读取流中的数据块，直到全部读取完毕
            while True:
                # 读取数据块，最大读取长度为 read_size 或者剩余未读取的数据长度
                block = stream.read(min(read_size,
                                        size - so_far))
                if not block:
                    break
                # 更新已读取数据长度和数据内容
                so_far += len(block)
                data2 += block
            # 断言读取得到的数据与原始数据一致
            assert_equal(data, data2)

        # 遍历不同的文件大小和读取大小进行测试
        for size in SIZES:
            for read_size in READ_SIZES:
                check(size, read_size)

    # 定义一个测试方法 test_read_max_length，用于测试最大长度读取功能
    def test_read_max_length(self):
        # 设置数据大小为 1234
        size = 1234
        # 创建随机数据并压缩
        data = np.random.randint(0, 256, size).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data)
        # 构造带有额外数据的压缩流
        compressed_stream = BytesIO(compressed_data + b"abbacaca")
        # 创建 ZlibInputStream 对象
        stream = ZlibInputStream(compressed_stream, len(compressed_data))

        # 读取与数据长度相等的数据
        stream.read(len(data))
        # 断言压缩流的当前位置与压缩数据长度一致
        assert_equal(compressed_stream.tell(), len(compressed_data))

        # 断言在达到流末尾后继续读取会抛出 OSError 异常
        assert_raises(OSError, stream.read, 1)

    # 定义一个测试方法 test_read_bad_checksum，用于测试校验和错误情况下的读取
    def test_read_bad_checksum(self):
        # 创建随机数据并压缩
        data = np.random.randint(0, 256, 10).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data)

        # 打破校验和，修改最后一个字节
        compressed_data = (compressed_data[:-1]
                           + bytes([(compressed_data[-1] + 1) & 255]))

        # 创建带有错误校验和的压缩流
        compressed_stream = BytesIO(compressed_data)
        # 创建 ZlibInputStream 对象
        stream = ZlibInputStream(compressed_stream, len(compressed_data))

        # 断言在校验和错误时读取数据会抛出 zlib.error 异常
        assert_raises(zlib.error, stream.read, len(data))

    # 定义一个测试方法 test_seek，用于测试流的定位功能
    def test_seek(self):
        # 获取压缩流、压缩数据长度和原始数据
        compressed_stream, compressed_data_len, data = self._get_data(1024)

        # 创建 ZlibInputStream 对象
        stream = ZlibInputStream(compressed_stream, compressed_data_len)

        # 将流定位到位置 123
        stream.seek(123)
        p = 123
        # 断言当前流位置与预期位置一致
        assert_equal(stream.tell(), p)
        # 读取长度为 11 的数据并断言与原始数据一致
        d1 = stream.read(11)
        assert_equal(d1, data[p:p+11])

        # 将流向前偏移 321 个字节
        stream.seek(321, 1)
        p = 123+11+321
        # 断言当前流位置与预期位置一致
        assert_equal(stream.tell(), p)
        # 读取长度为 21 的数据并断言与原始数据一致
        d2 = stream.read(21)
        assert_equal(d2, data[p:p+21])

        # 将流定位到位置 641
        stream.seek(641, 0)
        p = 641
        # 断言当前流位置与预期位置一致
        assert_equal(stream.tell(), p)
        # 读取长度为 11 的数据并断言与原始数据一致
        d3 = stream.read(11)
        assert_equal(d3, data[p:p+11])

        # 断言在不支持的定位模式时调用 seek 方法会抛出 OSError 异常
        assert_raises(OSError, stream.seek, 10, 2)
        # 断言在负偏移时调用 seek 方法会抛出 OSError 异常
        assert_raises(OSError, stream.seek, -1, 1)
        # 断言在超出数据范围的偏移时调用 seek 方法会抛出 ValueError 异常
        assert_raises(ValueError, stream.seek, 1, 123)

        # 将流向前偏移 10000 个字节
        stream.seek(10000, 1)
        # 断言在超出数据范围后读取数据会抛出 OSError 异常
        assert_raises(OSError, stream.read, 12)
    def test_seek_bad_checksum(self):
        data = np.random.randint(0, 256, 10).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data)

        # break checksum
        # 打破校验和
        compressed_data = (compressed_data[:-1]
                           + bytes([(compressed_data[-1] + 1) & 255]))

        compressed_stream = BytesIO(compressed_data)
        stream = ZlibInputStream(compressed_stream, len(compressed_data))

        # 断言抛出异常，验证在指定位置寻找数据时出现异常
        assert_raises(zlib.error, stream.seek, len(data))

    def test_all_data_read(self):
        compressed_stream, compressed_data_len, data = self._get_data(1024)
        stream = ZlibInputStream(compressed_stream, compressed_data_len)
        
        # 断言当前流未完全读取所有数据
        assert_(not stream.all_data_read())
        stream.seek(512)
        
        # 断言当前流未完全读取所有数据
        assert_(not stream.all_data_read())
        stream.seek(1024)
        
        # 断言当前流已完全读取所有数据
        assert_(stream.all_data_read())

    def test_all_data_read_overlap(self):
        COMPRESSION_LEVEL = 6

        data = np.arange(33707000).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data, COMPRESSION_LEVEL)
        compressed_data_len = len(compressed_data)

        # check that part of the checksum overlaps
        # 检查校验和的部分重叠
        assert_(compressed_data_len == BLOCK_SIZE + 2)

        compressed_stream = BytesIO(compressed_data)
        stream = ZlibInputStream(compressed_stream, compressed_data_len)
        
        # 断言当前流未完全读取所有数据
        assert_(not stream.all_data_read())
        stream.seek(len(data))
        
        # 断言当前流已完全读取所有数据
        assert_(stream.all_data_read())

    def test_all_data_read_bad_checksum(self):
        COMPRESSION_LEVEL = 6

        data = np.arange(33707000).astype(np.uint8).tobytes()
        compressed_data = zlib.compress(data, COMPRESSION_LEVEL)
        compressed_data_len = len(compressed_data)

        # check that part of the checksum overlaps
        # 检查校验和的部分重叠
        assert_(compressed_data_len == BLOCK_SIZE + 2)

        # break checksum
        # 打破校验和
        compressed_data = (compressed_data[:-1]
                           + bytes([(compressed_data[-1] + 1) & 255]))

        compressed_stream = BytesIO(compressed_data)
        stream = ZlibInputStream(compressed_stream, compressed_data_len)
        
        # 断言当前流未完全读取所有数据
        assert_(not stream.all_data_read())
        stream.seek(len(data))
        
        # 断言抛出异常，验证在指定位置判断是否所有数据已读时出现异常
        assert_raises(zlib.error, stream.all_data_read)
```