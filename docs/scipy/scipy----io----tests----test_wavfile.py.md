# `D:\src\scipysrc\scipy\scipy\io\tests\test_wavfile.py`

```
# 导入必要的库和模块
import os
import sys
from io import BytesIO  # 导入字节流处理模块

import numpy as np  # 导入NumPy库
from numpy.testing import (assert_equal, assert_, assert_array_equal,
                           break_cycles, suppress_warnings, IS_PYPY)
import pytest  # 导入pytest测试框架
from pytest import raises, warns

from scipy.io import wavfile  # 导入SciPy中处理音频文件的模块


def datafile(fn):
    # 构造数据文件的完整路径
    return os.path.join(os.path.dirname(__file__), 'data', fn)


def test_read_1():
    # 测试读取32位PCM格式音频文件（使用可扩展格式）
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)  # 断言采样率为44100
        assert_(np.issubdtype(data.dtype, np.int32))  # 断言数据类型为32位整型
        assert_equal(data.shape, (4410,))  # 断言数据形状为(4410,)，即4410个采样点

        del data  # 删除数据对象，释放内存


def test_read_2():
    # 测试读取8位无符号PCM格式音频文件
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-2ch-1byteu.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 8000)  # 断言采样率为8000
        assert_(np.issubdtype(data.dtype, np.uint8))  # 断言数据类型为8位无符号整型
        assert_equal(data.shape, (800, 2))  # 断言数据形状为(800, 2)，即800个采样点，2个声道

        del data  # 删除数据对象，释放内存


def test_read_3():
    # 测试读取小端格式的浮点数音频文件
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-le.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)  # 断言采样率为44100
        assert_(np.issubdtype(data.dtype, np.float32))  # 断言数据类型为32位浮点数
        assert_equal(data.shape, (441, 2))  # 断言数据形状为(441, 2)，即441个采样点，2个声道

        del data  # 删除数据对象，释放内存


def test_read_4():
    # 测试含有不支持的'PEAK'块的音频文件
    for mmap in [False, True]:
        with suppress_warnings() as sup:
            sup.filter(wavfile.WavFileWarning,
                       "Chunk .non-data. not understood, skipping it")
            filename = 'test-48000Hz-2ch-64bit-float-le-wavex.wav'
            rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 48000)  # 断言采样率为48000
        assert_(np.issubdtype(data.dtype, np.float64))  # 断言数据类型为64位浮点数
        assert_equal(data.shape, (480, 2))  # 断言数据形状为(480, 2)，即480个采样点，2个声道

        del data  # 删除数据对象，释放内存


def test_read_5():
    # 测试大端格式的浮点数音频文件
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-be.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)  # 断言采样率为44100
        assert_(np.issubdtype(data.dtype, np.float32))  # 断言数据类型为32位浮点数
        assert_(data.dtype.byteorder == '>' or (sys.byteorder == 'big' and
                                                data.dtype.byteorder == '='))  # 断言数据类型为大端字节序
        assert_equal(data.shape, (441, 2))  # 断言数据形状为(441, 2)，即441个采样点，2个声道

        del data  # 删除数据对象，释放内存


def test_5_bit_odd_size_no_pad():
    # 测试5位、1字节容器、5个声道、9个采样点、45字节数据块的音频文件
    # 由LTspice生成，它错误地省略了填充字节，但应能正常读取
    # 针对两种内存映射方式（分别为非内存映射和内存映射），执行以下操作：
    for mmap in [False, True]:
        # 指定音频文件名
        filename = 'test-8000Hz-le-5ch-9S-5bit.wav'
        # 使用给定的数据文件函数处理文件名，返回音频数据的文件路径
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        # 断言音频采样率为 8000 Hz
        assert_equal(rate, 8000)
        # 断言数据类型为无符号 8 位整数
        assert_(np.issubdtype(data.dtype, np.uint8))
        # 断言数据形状为 (9, 5)，即9行5列
        assert_equal(data.shape, (9, 5))

        # 断言数据的最低3位（LSBits）应为0
        assert_equal(data & 0b00000111, 0)

        # 断言数据为无符号数
        assert_equal(data.max(), 0b11111000)  # 数据的最大可能值
        assert_equal(data[0, 0], 128)  # 对于 <= 8 位的数据，中间点为 128
        assert_equal(data.min(), 0)  # 数据的最小可能值

        # 清除数据变量，释放内存
        del data
# 定义一个测试函数，用于测试12位偶数大小的音频文件
def test_12_bit_even_size():
    # 定义测试数据的详细信息，包括位深度、通道数、采样数和数据块大小
    # 生成自LTspice的1V峰峰值正弦波
    for mmap in [False, True]:
        # 设置测试文件名
        filename = 'test-8000Hz-le-4ch-9S-12bit.wav'
        # 调用wavfile模块读取文件，并根据mmap参数选择是否进行内存映射
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        # 断言采样率为8000
        assert_equal(rate, 8000)
        # 断言数据类型为np.int16的子类型
        assert_(np.issubdtype(data.dtype, np.int16))
        # 断言数据的形状为(9, 4)
        assert_equal(data.shape, (9, 4))

        # 断言数据的最低4位（即LSB）应为0
        assert_equal(data & 0b00000000_00001111, 0)

        # 断言数据的最大值为0b01111111_11110000，即最大可能值
        assert_equal(data.max(), 0b01111111_11110000)
        # 断言数据的第一个采样点的第一个通道值为0，表示中间值在大于等于9位时为0
        assert_equal(data[0, 0], 0)
        # 断言数据的最小值为-0b10000000_00000000，即最小可能值
        assert_equal(data.min(), -0b10000000_00000000)

        # 删除数据变量，释放内存
        del data


# 定义一个测试函数，用于测试24位奇数大小并带填充字节的音频文件
def test_24_bit_odd_size_with_pad():
    # 定义测试数据的详细信息，包括位深度、通道数、采样数和数据块大小
    # 不应该引发关于数据块填充字节的任何警告
    filename = 'test-8000Hz-le-3ch-5S-24bit.wav'
    # 调用wavfile模块读取文件，关闭内存映射功能
    rate, data = wavfile.read(datafile(filename), mmap=False)

    # 断言采样率为8000
    assert_equal(rate, 8000)
    # 断言数据类型为np.int32的子类型
    assert_(np.issubdtype(data.dtype, np.int32))
    # 断言数据的形状为(5, 3)
    assert_equal(data.shape, (5, 3))

    # 断言数据的最低字节应为0
    assert_equal(data & 0xff, 0)

    # 手工制作的最大/最小采样点，采用不同的约定：
    #                      2**(N-1)     2**(N-1)-1     LSB
    assert_equal(data, [[-0x8000_0000, -0x7fff_ff00, -0x200],
                        [-0x4000_0000, -0x3fff_ff00, -0x100],
                        [+0x0000_0000, +0x0000_0000, +0x000],
                        [+0x4000_0000, +0x3fff_ff00, +0x100],
                        [+0x7fff_ff00, +0x7fff_ff00, +0x200]])
    #                     ^ 被剪辑


# 定义一个测试函数，用于测试20位大小并包含额外数据的音频文件
def test_20_bit_extra_data():
    # 定义测试数据的详细信息，包括位深度、通道数、采样数和数据块大小
    # 附加数据填充容器超过位深度
    filename = 'test-8000Hz-le-1ch-10S-20bit-extra.wav'
    # 调用wavfile模块读取文件，关闭内存映射功能
    rate, data = wavfile.read(datafile(filename), mmap=False)

    # 断言采样率为1234
    assert_equal(rate, 1234)
    # 断言数据类型为np.int32的子类型
    assert_(np.issubdtype(data.dtype, np.int32))
    # 断言数据的形状为(10,)
    assert_equal(data.shape, (10,))

    # 断言数据的最低字节应为0，因为在4字节dtype中使用了3字节的容器
    assert_equal(data & 0xff, 0)

    # 但它应该加载超过20位的数据
    assert_((data & 0xf00).any())

    # 全幅正负采样点，然后每次减半
    assert_equal(data, [+0x7ffff000,       # +全幅20位
                        -0x7ffff000,       # -全幅20位
                        +0x7ffff000 >> 1,  # +1/2
                        -0x7ffff000 >> 1,  # -1/2
                        +0x7ffff000 >> 2,  # +1/4
                        -0x7ffff000 >> 2,  # -1/4
                        +0x7ffff000 >> 3,  # +1/8
                        -0x7ffff000 >> 3,  # -1/8
                        +0x7ffff000 >> 4,  # +1/16
                        -0x7ffff000 >> 4,  # -1/16
                        ])


# 定义一个测试函数，用于测试36位奇数大小的音频文件
def test_36_bit_odd_size():
    # 指定要读取的音频文件名
    filename = 'test-8000Hz-le-3ch-5S-36bit.wav'
    # 调用函数 `datafile` 处理文件名，然后读取 WAV 文件，返回采样率和数据
    rate, data = wavfile.read(datafile(filename), mmap=False)

    # 断言语句：验证采样率是否为 8000
    assert_equal(rate, 8000)
    # 断言语句：验证数据类型是否为 np.int64 的子类型
    assert_(np.issubdtype(data.dtype, np.int64))
    # 断言语句：验证数据形状是否为 (5, 3)
    assert_equal(data.shape, (5, 3))

    # 断言语句：验证数据的最低 28 位是否为 0
    assert_equal(data & 0xfffffff, 0)

    # 手工制作的最大/最小样本集合，按不同规范：
    #            固定点 2**(N-1)    全幅值 2**(N-1)-1       最低有效位
    correct = [[-0x8000_0000_0000_0000, -0x7fff_ffff_f000_0000, -0x2000_0000],
               [-0x4000_0000_0000_0000, -0x3fff_ffff_f000_0000, -0x1000_0000],
               [+0x0000_0000_0000_0000, +0x0000_0000_0000_0000, +0x0000_0000],
               [+0x4000_0000_0000_0000, +0x3fff_ffff_f000_0000, +0x1000_0000],
               [+0x7fff_ffff_f000_0000, +0x7fff_ffff_f000_0000, +0x2000_0000]]
    #              ^ 被截断

    # 断言语句：验证读取的数据是否与预期的正确数据集合 `correct` 相等
    assert_equal(data, correct)
def test_45_bit_even_size():
    # 定义测试函数，用于测试一个特定音频文件格式的处理
    # 45位，6字节容器，3个通道，5个样本，90字节数据块
    filename = 'test-8000Hz-le-3ch-5S-45bit.wav'
    # 调用wavfile模块的read函数读取指定文件的采样率和数据
    rate, data = wavfile.read(datafile(filename), mmap=False)

    # 断言：采样率应为8000
    assert_equal(rate, 8000)
    # 断言：数据类型应为np.int64的子类型
    assert_(np.issubdtype(data.dtype, np.int64))
    # 断言：数据形状应为(5, 3)
    assert_equal(data.shape, (5, 3))

    # 断言：最低19位应为0
    assert_equal(data & 0x7ffff, 0)

    # 手工制作的最大/最小样本在不同约定下：
    # 固定点2**(N-1)    全幅范围2**(N-1)-1      最低有效位
    correct = [[-0x8000_0000_0000_0000, -0x7fff_ffff_fff8_0000, -0x10_0000],
               [-0x4000_0000_0000_0000, -0x3fff_ffff_fff8_0000, -0x08_0000],
               [+0x0000_0000_0000_0000, +0x0000_0000_0000_0000, +0x00_0000],
               [+0x4000_0000_0000_0000, +0x3fff_ffff_fff8_0000, +0x08_0000],
               [+0x7fff_ffff_fff8_0000, +0x7fff_ffff_fff8_0000, +0x10_0000]]
    #              ^ 被剪切

    # 断言：数据应与预期结果相等
    assert_equal(data, correct)


def test_53_bit_odd_size():
    # 定义测试函数，用于测试另一种特定音频文件格式的处理
    # 53位，7字节容器，3个通道，5个样本，105字节数据块 + 填充
    filename = 'test-8000Hz-le-3ch-5S-53bit.wav'
    # 调用wavfile模块的read函数读取指定文件的采样率和数据
    rate, data = wavfile.read(datafile(filename), mmap=False)

    # 断言：采样率应为8000
    assert_equal(rate, 8000)
    # 断言：数据类型应为np.int64的子类型
    assert_(np.issubdtype(data.dtype, np.int64))
    # 断言：数据形状应为(5, 3)
    assert_equal(data.shape, (5, 3))

    # 断言：最低11位应为0
    assert_equal(data & 0x7ff, 0)

    # 手工制作的最大/最小样本在不同约定下：
    # 固定点2**(N-1)    全幅范围2**(N-1)-1    最低有效位
    correct = [[-0x8000_0000_0000_0000, -0x7fff_ffff_ffff_f800, -0x1000],
               [-0x4000_0000_0000_0000, -0x3fff_ffff_ffff_f800, -0x0800],
               [+0x0000_0000_0000_0000, +0x0000_0000_0000_0000, +0x0000],
               [+0x4000_0000_0000_0000, +0x3fff_ffff_ffff_f800, +0x0800],
               [+0x7fff_ffff_ffff_f800, +0x7fff_ffff_ffff_f800, +0x1000]]
    #              ^ 被剪切

    # 断言：数据应与预期结果相等
    assert_equal(data, correct)


def test_64_bit_even_size():
    # 64位，8字节容器，3个通道，5个样本，120字节数据块
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-3ch-5S-64bit.wav'
        # 调用wavfile模块的read函数读取指定文件的采样率和数据
        rate, data = wavfile.read(datafile(filename), mmap=False)

        # 断言：采样率应为8000
        assert_equal(rate, 8000)
        # 断言：数据类型应为np.int64的子类型
        assert_(np.issubdtype(data.dtype, np.int64))
        # 断言：数据形状应为(5, 3)
        assert_equal(data.shape, (5, 3))

        # 手工制作的最大/最小样本在不同约定下：
        # 固定点2**(N-1)    全幅范围2**(N-1)-1   最低有效位
        correct = [[-0x8000_0000_0000_0000, -0x7fff_ffff_ffff_ffff, -0x2],
                   [-0x4000_0000_0000_0000, -0x3fff_ffff_ffff_ffff, -0x1],
                   [+0x0000_0000_0000_0000, +0x0000_0000_0000_0000, +0x0],
                   [+0x4000_0000_0000_0000, +0x3fff_ffff_ffff_ffff, +0x1],
                   [+0x7fff_ffff_ffff_ffff, +0x7fff_ffff_ffff_ffff, +0x2]]
        #              ^ 被剪切

        # 断言：数据应与预期结果相等
        assert_equal(data, correct)

        del data


def test_unsupported_mmap():
    # 测试无法映射到numpy类型的容器
    pass
    # 对于每个文件名，执行以下操作：
    for filename in {'test-8000Hz-le-3ch-5S-24bit.wav',
                     'test-8000Hz-le-3ch-5S-36bit.wav',
                     'test-8000Hz-le-3ch-5S-45bit.wav',
                     'test-8000Hz-le-3ch-5S-53bit.wav',
                     'test-8000Hz-le-1ch-10S-20bit-extra.wav'}:
        # 使用 datafile 函数获取文件的完整路径，并读取 WAV 文件
        with raises(ValueError, match="mmap.*not compatible"):
            # 调用 wavfile 模块中的 read 函数，读取指定文件名的 WAV 文件
            # 设置 mmap=True 参数，尝试通过内存映射方式读取文件（如果支持）
            rate, data = wavfile.read(datafile(filename), mmap=True)
    
    
    这段代码的作用是循环遍历给定的几个 WAV 文件名，尝试以内存映射方式读取每个文件，并捕获在这个过程中可能引发的 `ValueError` 异常，异常信息要求匹配 "mmap.*not compatible"。
def test_rifx():
    # 比较等效的 RIFX 和 RIFF 文件
    for rifx, riff in {('test-44100Hz-be-1ch-4bytes.wav',
                        'test-44100Hz-le-1ch-4bytes.wav'),
                       ('test-8000Hz-be-3ch-5S-24bit.wav',
                        'test-8000Hz-le-3ch-5S-24bit.wav')}:
        # 读取 RIFX 文件的采样率和数据，关闭内存映射
        rate1, data1 = wavfile.read(datafile(rifx), mmap=False)
        # 读取 RIFF 文件的采样率和数据，关闭内存映射
        rate2, data2 = wavfile.read(datafile(riff), mmap=False)
        # 断言两个文件的采样率相等
        assert_equal(rate1, rate2)
        # 断言两个文件的数据完全相等
        assert_equal(data1, data2)


def test_rf64():
    # 比较等效的 RF64 和 RIFF 文件
    for rf64, riff in {('test-44100Hz-le-1ch-4bytes-rf64.wav',
                        'test-44100Hz-le-1ch-4bytes.wav'),
                       ('test-8000Hz-le-3ch-5S-24bit-rf64.wav',
                        'test-8000Hz-le-3ch-5S-24bit.wav')}:
        # 读取 RF64 文件的采样率和数据，关闭内存映射
        rate1, data1 = wavfile.read(datafile(rf64), mmap=False)
        # 读取 RIFF 文件的采样率和数据，关闭内存映射
        rate2, data2 = wavfile.read(datafile(riff), mmap=False)
        # 断言两个文件的采样率数组相等
        assert_array_equal(rate1, rate2)
        # 断言两个文件的数据数组相等
        assert_array_equal(data1, data2)


@pytest.mark.xslow
def test_write_roundtrip_rf64(tmpdir):
    # 定义数据类型为小端格式的 64 位整数
    dtype = np.dtype("<i8")
    # 生成一个临时文件名
    tmpfile = str(tmpdir.join('temp.wav'))
    # 设置采样率为 44100，生成随机数据，数据类型为 dtype
    data = np.random.randint(0, 127, (2**29,)).astype(dtype)
    # 写入 WAV 文件
    wavfile.write(tmpfile, rate, data)

    # 从临时文件中读取数据，开启内存映射
    rate2, data2 = wavfile.read(tmpfile, mmap=True)

    # 断言读取的采样率与写入的采样率相等
    assert_equal(rate, rate2)
    # 检查数据的字节顺序是否为小端、等于或不确定
    msg = f"{data2.dtype} byteorder not in ('<', '=', '|')"
    assert data2.dtype.byteorder in ('<', '=', '|'), msg
    # 断言写入和读取的数据数组完全相等
    assert_array_equal(data, data2)
    # 进一步测试写入功能 (gh-12176)
    data2[0] = 0


def test_read_unknown_filetype_fail():
    # 不是 RIFF 文件
    for mmap in [False, True]:
        filename = 'example_1.nc'
        # 打开文件流，并预期抛出特定的 ValueError 异常
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="CDF.*'RIFF', 'RIFX', and 'RF64' supported"):
                wavfile.read(fp, mmap=mmap)


def test_read_unknown_riff_form_type():
    # 是 RIFF 文件，但不是 WAVE 格式
    for mmap in [False, True]:
        filename = 'Transparent Busy.ani'
        # 打开文件流，并预期抛出特定的 ValueError 异常
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Not a WAV file.*ACON'):
                wavfile.read(fp, mmap=mmap)


def test_read_unknown_wave_format():
    # 是 RIFF 和 WAVE 格式，但不是支持的格式
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-1ch-1byte-ulaw.wav'
        # 打开文件流，并预期抛出特定的 ValueError 异常
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unknown wave file format.*MULAW.*'
                        'Supported formats'):
                wavfile.read(fp, mmap=mmap)


def test_read_early_eof_with_data():
    # 文件在 'data' 块内部结束，但我们保留不完整的数据
    # 对于每个 mmap 的取值 False 和 True，分别执行以下操作
    for mmap in [False, True]:
        # 指定要打开的 WAV 文件名
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof.wav'
        # 打开文件，并使用二进制模式读取文件内容
        with open(datafile(filename), 'rb') as fp:
            # 断言在读取 WAV 文件时发出特定警告，并匹配 'Reached EOF'
            with warns(wavfile.WavFileWarning, match='Reached EOF'):
                # 使用 wavfile.read() 从文件指针 fp 中读取数据和采样率，使用 mmap 参数指定是否使用内存映射
                rate, data = wavfile.read(fp, mmap=mmap)
                # 断言读取的数据大小大于 0
                assert data.size > 0
                # 断言读取的采样率为 44100
                assert rate == 44100
                # 还要测试写入操作（gh-12176）
                # 修改数据的第一个元素为 0，以测试写入功能
                data[0] = 0
# 测试函数：检验在文件提前结束（early eof）情况下的 WAV 文件读取
def test_read_early_eof():
    # 针对内存映射和普通读取两种方式进行测试
    for mmap in [False, True]:
        # 准备测试用的文件名
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav'
        # 打开文件流进行读取
        with open(datafile(filename), 'rb') as fp:
            # 断言应当抛出值错误，并匹配特定错误信息
            with raises(ValueError, match="Unexpected end of file."):
                # 调用 wavfile 模块的读取函数进行读取操作
                wavfile.read(fp, mmap=mmap)


# 测试函数：检验在不完整的数据块（incomplete chunk）情况下的 WAV 文件读取
def test_read_incomplete_chunk():
    # 针对内存映射和普通读取两种方式进行测试
    for mmap in [False, True]:
        # 准备测试用的文件名
        filename = 'test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav'
        # 打开文件流进行读取
        with open(datafile(filename), 'rb') as fp:
            # 断言应当抛出值错误，并匹配特定错误信息
            with raises(ValueError, match="Incomplete chunk ID.*b'f'"):
                # 调用 wavfile 模块的读取函数进行读取操作
                wavfile.read(fp, mmap=mmap)


# 测试函数：检验在不一致的文件头（header）情况下的 WAV 文件读取
def test_read_inconsistent_header():
    # 针对内存映射和普通读取两种方式进行测试
    for mmap in [False, True]:
        # 准备测试用的文件名
        filename = 'test-8000Hz-le-3ch-5S-24bit-inconsistent.wav'
        # 打开文件流进行读取
        with open(datafile(filename), 'rb') as fp:
            # 断言应当抛出值错误，并匹配特定错误信息
            with raises(ValueError, match="header is invalid"):
                # 调用 wavfile 模块的读取函数进行读取操作
                wavfile.read(fp, mmap=mmap)


# 参数化测试：检验 WAV 文件写入和读取的往返操作
# 测试类型包括各种数据类型、声道数、采样率、内存映射和实际文件等情况
@pytest.mark.parametrize("dt_str", ["<i2", "<i4", "<i8", "<f4", "<f8",
                                    ">i2", ">i4", ">i8", ">f4", ">f8", '|u1'])
@pytest.mark.parametrize("channels", [1, 2, 5])
@pytest.mark.parametrize("rate", [8000, 32000])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("realfile", [False, True])
def test_write_roundtrip(realfile, mmap, rate, channels, dt_str, tmpdir):
    # 转换数据类型为 numpy 的数据类型
    dtype = np.dtype(dt_str)
    # 根据测试是否使用真实文件来选择临时文件存储方式
    if realfile:
        tmpfile = str(tmpdir.join('temp.wav'))
    else:
        tmpfile = BytesIO()
    # 生成随机数据
    data = np.random.rand(100, channels)
    # 如果只有一个声道，将数据重新组织为一维数组
    if channels == 1:
        data = data[:, 0]
    # 如果数据类型是浮点型，则将数据类型转换为对应的 dtype 类型
    if dtype.kind == 'f':
        data = data.astype(dtype)
    else:
        data = (data * 128).astype(dtype)

    # 调用 wavfile 模块的写入函数，将数据写入临时文件或内存中
    wavfile.write(tmpfile, rate, data)

    # 调用 wavfile 模块的读取函数，从临时文件或内存中读取数据
    rate2, data2 = wavfile.read(tmpfile, mmap=mmap)

    # 断言读取的采样率应与写入时的采样率一致
    assert_equal(rate, rate2)
    # 断言数据的字节顺序应为小端、大端或平台相关
    assert_(data2.dtype.byteorder in ('<', '=', '|'), msg=data2.dtype)
    # 断言读取的数据与写入的数据应当一致
    assert_array_equal(data, data2)

    # 对于真实文件，并且使用了内存映射的情况下，测试写入只读数据的异常处理
    if realfile:
        data2[0] = 0
    else:
        with pytest.raises(ValueError, match='read-only'):
            data2[0] = 0

    # 对于特定条件下的平台依赖性情况，进行资源清理操作
    if realfile and mmap and IS_PYPY and sys.platform == 'win32':
        # Windows 平台下 PyPy 无法移除由内存映射保持的死文件，需要进行资源清理
        break_cycles()
        break_cycles()


# 参数化测试：检验不支持的 WAV 文件数据类型
# 测试数据类型为 np.float16
@pytest.mark.parametrize("dtype", [np.float16])
def test_wavfile_dtype_unsupported(tmpdir, dtype):
    # 准备临时文件名
    tmpfile = str(tmpdir.join('temp.wav'))
    # 使用随机数生成器生成随机数据，并将数据类型设置为指定的 dtype
    rng = np.random.default_rng(1234)
    data = rng.random((100, 5)).astype(dtype)
    rate = 8000
    # 使用 pytest 的上下文管理器来测试抛出 ValueError 异常，并验证异常信息中包含特定字符串 "Unsupported"
    with pytest.raises(ValueError, match="Unsupported"):
        # 调用 wavfile 模块的 write 函数，将 rate 和 data 写入临时文件 tmpfile
        wavfile.write(tmpfile, rate, data)
```