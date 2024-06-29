# `.\numpy\numpy\lib\tests\test_packbits.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库
from numpy.testing import assert_array_equal, assert_equal, assert_raises  # 导入NumPy测试相关的函数和类
import pytest  # 导入pytest库
from itertools import chain  # 导入itertools库中的chain函数

# 定义测试函数：测试np.packbits函数
def test_packbits():
    # 定义测试数据a，这里使用了多维列表表示布尔值数组
    a = [[[1, 0, 1], [0, 1, 0]],
         [[1, 1, 0], [0, 0, 1]]]
    
    # 对于不同的数据类型dt进行测试
    for dt in '?bBhHiIlLqQ':
        # 将a转换为NumPy数组，指定数据类型为dt
        arr = np.array(a, dtype=dt)
        # 使用np.packbits函数进行位压缩，压缩轴为最后一个轴
        b = np.packbits(arr, axis=-1)
        # 断言压缩结果的数据类型为np.uint8
        assert_equal(b.dtype, np.uint8)
        # 断言压缩后的结果数组与预期的数组相等
        assert_array_equal(b, np.array([[[160], [64]], [[192], [32]]]))

    # 测试处理异常情况：输入数据类型为float时抛出TypeError异常
    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))


# 定义测试函数：测试处理空数组的np.packbits函数
def test_packbits_empty():
    # 定义不同的空数组形状
    shapes = [
        (0,), (10, 20, 0), (10, 0, 20), (0, 10, 20), (20, 0, 0), (0, 20, 0),
        (0, 0, 20), (0, 0, 0),
    ]
    # 对于不同的数据类型dt和形状shape进行测试
    for dt in '?bBhHiIlLqQ':
        for shape in shapes:
            # 创建指定形状的空数组a，指定数据类型为dt
            a = np.empty(shape, dtype=dt)
            # 使用np.packbits函数对空数组a进行位压缩
            b = np.packbits(a)
            # 断言压缩结果的数据类型为np.uint8
            assert_equal(b.dtype, np.uint8)
            # 断言压缩后的结果数组形状为(0,)
            assert_equal(b.shape, (0,))


# 定义测试函数：测试带有轴参数的np.packbits函数处理空数组
def test_packbits_empty_with_axis():
    # 定义原始形状和不同轴的压缩后形状列表
    shapes = [
        ((0,), [(0,)]),
        ((10, 20, 0), [(2, 20, 0), (10, 3, 0), (10, 20, 0)]),
        ((10, 0, 20), [(2, 0, 20), (10, 0, 20), (10, 0, 3)]),
        ((0, 10, 20), [(0, 10, 20), (0, 2, 20), (0, 10, 3)]),
        ((20, 0, 0), [(3, 0, 0), (20, 0, 0), (20, 0, 0)]),
        ((0, 20, 0), [(0, 20, 0), (0, 3, 0), (0, 20, 0)]),
        ((0, 0, 20), [(0, 0, 20), (0, 0, 20), (0, 0, 3)]),
        ((0, 0, 0), [(0, 0, 0), (0, 0, 0), (0, 0, 0)]),
    ]
    # 对于不同的数据类型dt、输入形状in_shape和输出形状out_shape进行测试
    for dt in '?bBhHiIlLqQ':
        for in_shape, out_shapes in shapes:
            for ax, out_shape in enumerate(out_shapes):
                # 创建指定形状的空数组a，指定数据类型为dt
                a = np.empty(in_shape, dtype=dt)
                # 使用np.packbits函数对空数组a进行位压缩，指定压缩轴为ax
                b = np.packbits(a, axis=ax)
                # 断言压缩结果的数据类型为np.uint8
                assert_equal(b.dtype, np.uint8)
                # 断言压缩后的结果数组形状与预期的out_shape相等
                assert_equal(b.shape, out_shape)


@pytest.mark.parametrize('bitorder', ('little', 'big'))
# 定义测试函数：测试大数据量情况下的np.packbits函数
def test_packbits_large(bitorder):
    # test data large enough for 16 byte vectorization
    # 创建一个包含二进制整数数组的 NumPy 数组
    a = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                  0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
                  1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                  1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
                  1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                  1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
                  1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
                  0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1,
                  1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                  1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                  1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                  0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                  1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
                  1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
                  1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    # 将数组元素重复三次
    a = a.repeat(3)

    # 使用不同的数据类型测试以下代码的结果是否相同
    for dtype in 'bBhHiIlLqQ':
        # 将数组 a 转换为指定的数据类型
        arr = np.array(a, dtype=dtype)
        # 生成与 arr 相同大小的随机整数数组，范围在指定数据类型的最小值和最大值之间
        rnd = np.random.randint(low=np.iinfo(dtype).min,
                                high=np.iinfo(dtype).max, size=arr.size,
                                dtype=dtype)
        # 将 rnd 数组中的零值替换为1
        rnd[rnd == 0] = 1
        # 将 arr 数组与 rnd 数组中对应元素相乘，结果保持 dtype 数据类型
        arr *= rnd.astype(dtype)
        # 将 arr 数组打包成二进制位数组
        b = np.packbits(arr, axis=-1)
        # 验证解包后的结果是否与原始数组 a（去除末尾四个元素）相等
        assert_array_equal(np.unpackbits(b)[:-4], a)

    # 测试将浮点数数组作为输入时是否会引发 TypeError 异常
    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))
def test_packbits_very_large():
    # 对 np.packbits 函数进行大数组测试，用于解决 gh-8637 中的问题
    # 大数组可能更容易触发潜在的 bug
    for s in range(950, 1050):
        # 遍历不同的数据类型
        for dt in '?bBhHiIlLqQ':
            # 创建一个形状为 (200, s) 的全为 True 的数组
            x = np.ones((200, s), dtype=bool)
            # 对数组 x 进行按位压缩，沿着 axis=1 的方向
            np.packbits(x, axis=1)


def test_unpackbits():
    # 从文档字符串中复制的示例
    # 创建一个包含 [[2], [7], [23]] 的 numpy 数组，数据类型为 uint8
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    # 对数组 a 进行按位解压缩，沿着 axis=1 的方向
    b = np.unpackbits(a, axis=1)
    # 断言解压缩后的数组 b 的数据类型为 uint8
    assert_equal(b.dtype, np.uint8)
    # 断言解压缩后的数组 b 与给定的数组相等
    assert_array_equal(b, np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 0, 1, 1, 1]]))


def test_pack_unpack_order():
    # 创建一个包含 [[2], [7], [23]] 的 numpy 数组，数据类型为 uint8
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    # 对数组 a 进行按位解压缩，沿着 axis=1 的方向
    b = np.unpackbits(a, axis=1)
    # 断言解压缩后的数组 b 的数据类型为 uint8
    assert_equal(b.dtype, np.uint8)
    # 使用 'big' 字节顺序对数组 a 进行解压缩，得到数组 b_big
    b_big = np.unpackbits(a, axis=1, bitorder='big')
    # 使用 'little' 字节顺序对数组 a 进行解压缩，得到数组 b_little
    b_little = np.unpackbits(a, axis=1, bitorder='little')
    # 断言数组 b 与数组 b_big 相等
    assert_array_equal(b, b_big)
    # 断言数组 a 与以 'little' 字节顺序对数组 b_little 进行按位压缩后的结果相等
    assert_array_equal(a, np.packbits(b_little, axis=1, bitorder='little'))
    # 断言数组 b 的逆序列与数组 b_little 相等
    assert_array_equal(b[:,::-1], b_little)
    # 断言数组 a 与以 'big' 字节顺序对数组 b_big 进行按位压缩后的结果相等
    assert_array_equal(a, np.packbits(b_big, axis=1, bitorder='big'))
    # 断言当 'bitorder' 参数为 'r' 时，解压缩函数会引发 ValueError 异常
    assert_raises(ValueError, np.unpackbits, a, bitorder='r')
    # 断言当 'bitorder' 参数为整数 10 时，解压缩函数会引发 TypeError 异常
    assert_raises(TypeError, np.unpackbits, a, bitorder=10)


def test_unpackbits_empty():
    # 创建一个空的 numpy 数组，数据类型为 uint8
    a = np.empty((0,), dtype=np.uint8)
    # 对数组 a 进行按位解压缩
    b = np.unpackbits(a)
    # 断言解压缩后的数组 b 的数据类型为 uint8
    assert_equal(b.dtype, np.uint8)
    # 断言解压缩后的数组 b 为空数组
    assert_array_equal(b, np.empty((0,)))


def test_unpackbits_empty_with_axis():
    # 不同轴上的打包形状列表和解包形状列表
    shapes = [
        ([(0,)], (0,)),
        ([(2, 24, 0), (16, 3, 0), (16, 24, 0)], (16, 24, 0)),
        ([(2, 0, 24), (16, 0, 24), (16, 0, 3)], (16, 0, 24)),
        ([(0, 16, 24), (0, 2, 24), (0, 16, 3)], (0, 16, 24)),
        ([(3, 0, 0), (24, 0, 0), (24, 0, 0)], (24, 0, 0)),
        ([(0, 24, 0), (0, 3, 0), (0, 24, 0)], (0, 24, 0)),
        ([(0, 0, 24), (0, 0, 24), (0, 0, 3)], (0, 0, 24)),
        ([(0, 0, 0), (0, 0, 0), (0, 0, 0)], (0, 0, 0)),
    ]
    # 遍历不同的形状对，并测试对应的解包操作
    for in_shapes, out_shape in shapes:
        for ax, in_shape in enumerate(in_shapes):
            # 创建一个空的 numpy 数组，形状为 in_shape，数据类型为 uint8
            a = np.empty(in_shape, dtype=np.uint8)
            # 对数组 a 进行按位解压缩，沿着指定的轴 ax
            b = np.unpackbits(a, axis=ax)
            # 断言解压缩后的数组 b 的数据类型为 uint8
            assert_equal(b.dtype, np.uint8)
            # 断言解压缩后的数组 b 的形状与预期的 out_shape 相等
            assert_equal(b.shape, out_shape)


def test_unpackbits_large():
    # 对所有可能的数字进行测试，通过与已经测试过的 packbits 进行比较
    d = np.arange(277, dtype=np.uint8)
    # 断言解压缩后再压缩的结果与原数组 d 相等
    assert_array_equal(np.packbits(np.unpackbits(d)), d)
    # 断言解压缩后再压缩的结果与 d[::2] 相等
    assert_array_equal(np.packbits(np.unpackbits(d[::2])), d[::2])
    # 将数组 d 在行方向上重复三次
    d = np.tile(d, (3, 1))
    # 断言解压缩后再压缩的结果与原始数组 d 相等，沿着 axis=1 的方向
    assert_array_equal(np.packbits(np.unpackbits(d, axis=1), axis=1), d)
    # 将数组 d 进行转置，并创建其副本
    d = d.T.copy()
    # 断言解压缩后再压缩的结果与原始数组 d 相等，沿着 axis=0 的方向
    assert_array_equal(np.packbits(np.unpackbits(d, axis=0), axis=0), d)


class TestCount():
    # 创建一个 7x7 的二维数组，元素为 0 或 1，数据类型为 uint8
    x = np.array([
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
    ], dtype=np.uint8)
    # 创建一个长度为 57 的一维数组，元素为 0，数据类型为 uint8
    padded1 = np.zeros(57, dtype=np.uint8)
    # 将二维数组 x 按行展开并填充到 padded1 的前 49 个位置
    padded1[:49] = x.ravel()
    # 创建一个长度为 57 的一维数组，元素为 0，数据类型为 uint8
    padded1b = np.zeros(57, dtype=np.uint8)
    # 将二维数组 x 沿水平翻转后再按行展开并填充到 padded1b 的前 49 个位置
    padded1b[:49] = x[::-1].copy().ravel()
    # 创建一个 9x9 的二维数组，元素为 0，数据类型为 uint8
    padded2 = np.zeros((9, 9), dtype=np.uint8)
    # 将二维数组 x 按行、列展开并填充到 padded2 的前 7x7 个位置
    padded2[:7, :7] = x

    # 使用 pytest 的参数化装饰器，指定测试函数的参数化条件
    @pytest.mark.parametrize('bitorder', ('little', 'big'))
    @pytest.mark.parametrize('count', chain(range(58), range(-1, -57, -1)))
    # 定义测试函数 test_roundtrip，参数包括 bitorder 和 count
    def test_roundtrip(self, bitorder, count):
        # 如果 count 小于 0，则设定 cutoff 为 count - 1；否则设定为 count
        if count < 0:
            # 添加额外的零填充
            cutoff = count - 1
        else:
            cutoff = count
        # 对二维数组 x 进行位压缩（packbits），根据指定的 bitorder
        packed = np.packbits(self.x, bitorder=bitorder)
        # 对压缩后的数据进行位解压缩（unpackbits），根据指定的 count 和 bitorder
        unpacked = np.unpackbits(packed, count=count, bitorder=bitorder)
        # 断言解压缩后数据类型为 uint8
        assert_equal(unpacked.dtype, np.uint8)
        # 断言解压缩后的数组与 padded1 的前 cutoff 个元素相等
        assert_array_equal(unpacked, self.padded1[:cutoff])

    # 使用 pytest 的参数化装饰器，指定测试函数的参数化条件
    @pytest.mark.parametrize('kwargs', [
                    {}, {'count': None},
                    ])
    # 定义测试函数 test_count，参数为 kwargs
    def test_count(self, kwargs):
        # 对二维数组 x 进行位压缩
        packed = np.packbits(self.x)
        # 对压缩后的数据进行位解压缩，根据 kwargs 中的参数
        unpacked = np.unpackbits(packed, **kwargs)
        # 断言解压缩后数据类型为 uint8
        assert_equal(unpacked.dtype, np.uint8)
        # 断言解压缩后的数组与 padded1 的除最后一个元素外的所有元素相等
        assert_array_equal(unpacked, self.padded1[:-1])

    # 使用 pytest 的参数化装饰器，指定测试函数的参数化条件
    @pytest.mark.parametrize('bitorder', ('little', 'big'))
    # delta==-1 when count<0 because one extra zero of padding
    @pytest.mark.parametrize('count', chain(range(8), range(-1, -9, -1)))
    # 定义测试函数 test_roundtrip_axis，参数包括 bitorder 和 count
    def test_roundtrip_axis(self, bitorder, count):
        # 如果 count 小于 0，则设定 cutoff 为 count - 1；否则设定为 count
        if count < 0:
            # 添加额外的零填充
            cutoff = count - 1
        else:
            cutoff = count
        # 对二维数组 x 按指定轴进行位压缩，根据指定的 bitorder
        packed0 = np.packbits(self.x, axis=0, bitorder=bitorder)
        # 对压缩后的数据进行位解压缩，根据指定的轴、count 和 bitorder
        unpacked0 = np.unpackbits(packed0, axis=0, count=count,
                                  bitorder=bitorder)
        # 断言解压缩后数据类型为 uint8
        assert_equal(unpacked0.dtype, np.uint8)
        # 断言解压缩后的数组与 padded2 的前 cutoff 行和 x 的列数相等
        assert_array_equal(unpacked0, self.padded2[:cutoff, :self.x.shape[1]])

        # 对二维数组 x 按指定轴进行位压缩，根据指定的 bitorder
        packed1 = np.packbits(self.x, axis=1, bitorder=bitorder)
        # 对压缩后的数据进行位解压缩，根据指定的轴、count 和 bitorder
        unpacked1 = np.unpackbits(packed1, axis=1, count=count,
                                  bitorder=bitorder)
        # 断言解压缩后数据类型为 uint8
        assert_equal(unpacked1.dtype, np.uint8)
        # 断言解压缩后的数组与 padded2 的前 x 的行数和 cutoff 列相等
        assert_array_equal(unpacked1, self.padded2[:self.x.shape[0], :cutoff])

    # 使用 pytest 的参数化装饰器，指定测试函数的参数化条件
    @pytest.mark.parametrize('kwargs', [
                    {}, {'count': None},
                    {'bitorder' : 'little'},
                    {'bitorder': 'little', 'count': None},
                    {'bitorder' : 'big'},
                    {'bitorder': 'big', 'count': None},
                    ])
    # 测试函数，用于验证 np.packbits 和 np.unpackbits 的行为
    def test_axis_count(self, kwargs):
        # 在 axis=0 上对数组 self.x 进行位打包
        packed0 = np.packbits(self.x, axis=0)
        # 在 axis=0 上对 packed0 进行位解包，并根据 kwargs 参数进行额外配置
        unpacked0 = np.unpackbits(packed0, axis=0, **kwargs)
        # 断言解包后的数据类型为 np.uint8
        assert_equal(unpacked0.dtype, np.uint8)
        # 根据 bitorder 参数的配置，进行不同的数组比较和断言
        if kwargs.get('bitorder', 'big') == 'big':
            assert_array_equal(unpacked0, self.padded2[:-1, :self.x.shape[1]])
        else:
            assert_array_equal(unpacked0[::-1, :], self.padded2[:-1, :self.x.shape[1]])

        # 在 axis=1 上对数组 self.x 进行位打包
        packed1 = np.packbits(self.x, axis=1)
        # 在 axis=1 上对 packed1 进行位解包，并根据 kwargs 参数进行额外配置
        unpacked1 = np.unpackbits(packed1, axis=1, **kwargs)
        # 断言解包后的数据类型为 np.uint8
        assert_equal(unpacked1.dtype, np.uint8)
        # 根据 bitorder 参数的配置，进行不同的数组比较和断言
        if kwargs.get('bitorder', 'big') == 'big':
            assert_array_equal(unpacked1, self.padded2[:self.x.shape[0], :-1])
        else:
            assert_array_equal(unpacked1[:, ::-1], self.padded2[:self.x.shape[0], :-1])

    # 测试异常情况下的 np.unpackbits 函数
    def test_bad_count(self):
        # 在 axis=0 上对数组 self.x 进行位打包
        packed0 = np.packbits(self.x, axis=0)
        # 断言在指定 count=-9 的情况下，调用 np.unpackbits 会抛出 ValueError 异常
        assert_raises(ValueError, np.unpackbits, packed0, axis=0, count=-9)

        # 在 axis=1 上对数组 self.x 进行位打包
        packed1 = np.packbits(self.x, axis=1)
        # 断言在指定 count=-9 的情况下，调用 np.unpackbits 会抛出 ValueError 异常
        assert_raises(ValueError, np.unpackbits, packed1, axis=1, count=-9)

        # 对整个数组 self.x 进行位打包
        packed = np.packbits(self.x)
        # 断言在指定 count=-57 的情况下，调用 np.unpackbits 会抛出 ValueError 异常
        assert_raises(ValueError, np.unpackbits, packed, count=-57)
```