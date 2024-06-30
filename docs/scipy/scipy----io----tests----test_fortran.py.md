# `D:\src\scipysrc\scipy\scipy\io\tests\test_fortran.py`

```
''' Tests for fortran sequential files '''

import tempfile  # 导入临时文件操作模块
import shutil    # 导入文件和目录操作模块
from os import path   # 导入路径操作模块
from glob import iglob   # 导入文件通配符匹配模块
import re   # 导入正则表达式模块

from numpy.testing import assert_equal, assert_allclose   # 导入断言函数
import numpy as np   # 导入数值计算库
import pytest   # 导入测试框架

from scipy.io import (FortranFile,   # 导入Fortran文件读写相关模块
                      _test_fortran,   # 导入Fortran测试模块
                      FortranEOFError,   # 导入Fortran文件结尾错误异常
                      FortranFormattingError)   # 导入Fortran格式错误异常


DATA_PATH = path.join(path.dirname(__file__), 'data')   # 获取数据文件路径


def test_fortranfiles_read():
    for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
        m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)   # 使用正则表达式匹配文件名
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)   # 如果匹配失败，则抛出运行时错误

        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))   # 提取文件名中的维度信息

        dtype = m.group(1).replace('s', '<')   # 根据文件名中的类型信息设置数据类型

        f = FortranFile(filename, 'r', '<u4')   # 打开Fortran文件对象
        data = f.read_record(dtype=dtype).reshape(dims, order='F')   # 读取记录并按Fortran顺序重塑数据
        f.close()   # 关闭文件对象

        expected = np.arange(np.prod(dims)).reshape(dims).astype(dtype)   # 生成预期的数据数组
        assert_equal(data, expected)   # 断言实际数据与预期数据相等


def test_fortranfiles_mixed_record():
    filename = path.join(DATA_PATH, "fortran-mixed.dat")   # 获取混合记录文件路径
    with FortranFile(filename, 'r', '<u4') as f:   # 使用上下文管理器打开Fortran文件对象
        record = f.read_record('<i4,<f4,<i8,2<f8')   # 读取混合记录

    assert_equal(record['f0'][0], 1)   # 断言第一个字段的第一个元素为1
    assert_allclose(record['f1'][0], 2.3)   # 断言第二个字段的第一个元素约等于2.3
    assert_equal(record['f2'][0], 4)   # 断言第三个字段的第一个元素为4
    assert_allclose(record['f3'][0], [5.6, 7.8])   # 断言第四个字段的第一个元素约等于[5.6, 7.8]


def test_fortranfiles_write():
    for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
        m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)   # 使用正则表达式匹配文件名
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)   # 如果匹配失败，则抛出运行时错误
        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))   # 提取文件名中的维度信息

        dtype = m.group(1).replace('s', '<')   # 根据文件名中的类型信息设置数据类型
        data = np.arange(np.prod(dims)).reshape(dims).astype(dtype)   # 生成测试数据数组

        tmpdir = tempfile.mkdtemp()   # 创建临时目录
        try:
            testFile = path.join(tmpdir, path.basename(filename))   # 构建临时文件路径
            f = FortranFile(testFile, 'w', '<u4')   # 创建Fortran文件写对象
            f.write_record(data.T)   # 写入数据的转置
            f.close()   # 关闭文件对象

            originalfile = open(filename, 'rb')   # 打开原始数据文件
            newfile = open(testFile, 'rb')   # 打开临时生成的数据文件
            assert_equal(originalfile.read(), newfile.read(),   # 断言两个文件的内容是否相同
                         err_msg=filename)   # 错误消息为文件名
            originalfile.close()   # 关闭原始文件对象
            newfile.close()   # 关闭临时文件对象
        finally:
            shutil.rmtree(tmpdir)   # 清理临时目录


def test_fortranfile_read_mixed_record():
    # The data file fortran-3x3d-2i.dat contains the program that
    # produced it at the end.
    #
    # double precision :: a(3,3)
    # integer :: b(2)
    # ...
    # open(1, file='fortran-3x3d-2i.dat', form='unformatted')
    # write(1) a, b
    # close(1)
    #

    filename = path.join(DATA_PATH, "fortran-3x3d-2i.dat")   # 获取混合记录文件路径
    with FortranFile(filename, 'r', '<u4') as f:   # 使用上下文管理器打开Fortran文件对象
        record = f.read_record('(3,3)<f8', '2<i4')   # 读取混合记录
    # 创建一个包含两个整数元素的 NumPy 数组，数据类型为 int32
    bx = np.array([-1, -2], dtype=np.int32)
    
    # 使用断言检查 record 的第一个元素是否等于 ax 的转置
    assert_equal(record[0], ax.T)
    
    # 使用断言检查 record 的第二个元素是否等于 bx 的转置
    assert_equal(record[1], bx.T)
# 定义一个测试函数，用于测试FortranFile类的写入混合记录功能
def test_fortranfile_write_mixed_record(tmpdir):
    # 创建临时文件路径
    tf = path.join(str(tmpdir), 'test.dat')

    # 定义两个记录，每个记录包含数据类型和相应的数据数组
    r1 = (('f4', 'f4', 'i4'), (np.float32(2), np.float32(3), np.int32(100)))
    r2 = (('4f4', '(3,3)f4', '8i4'),
          (np.random.randint(255, size=[4]).astype(np.float32),
           np.random.randint(255, size=[3, 3]).astype(np.float32),
           np.random.randint(255, size=[8]).astype(np.int32)))
    records = [r1, r2]

    # 对每个记录执行以下操作：
    for dtype, a in records:
        # 使用FortranFile打开临时文件以写入数据
        with FortranFile(tf, 'w') as f:
            # 写入记录中的数据
            f.write_record(*a)

        # 使用FortranFile打开临时文件以读取数据
        with FortranFile(tf, 'r') as f:
            # 从文件中读取相同数据类型的记录
            b = f.read_record(*dtype)

        # 断言写入的数据长度与读取的数据长度相等
        assert_equal(len(a), len(b))

        # 逐个比较写入的数据和读取的数据
        for aa, bb in zip(a, b):
            assert_equal(bb, aa)


# 定义一个测试函数，用于测试FortranFile类的读写往返功能
def test_fortran_roundtrip(tmpdir):
    # 创建临时文件路径
    filename = path.join(str(tmpdir), 'test.dat')

    # 设置随机种子
    np.random.seed(1)

    # 测试不同数据类型的往返
    # 测试双精度数据
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n, k)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T)
    a2 = _test_fortran.read_unformatted_double(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3 = f.read_record('(2,3,5)f8').T
    assert_equal(a2, a)
    assert_equal(a3, a)

    # 测试整数数据
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n, k).astype(np.int32)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T)
    a2 = _test_fortran.read_unformatted_int(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3 = f.read_record('(2,3,5)i4').T
    assert_equal(a2, a)
    assert_equal(a3, a)

    # 测试混合数据类型
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n)
    b = np.random.randn(k).astype(np.intc)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T, b.T)
    a2, b2 = _test_fortran.read_unformatted_mixed(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3, b3 = f.read_record('(3,5)f8', '2i4')
        a3 = a3.T
    assert_equal(a2, a)
    assert_equal(a3, a)
    assert_equal(b2, b)
    assert_equal(b3, b)


# 定义一个测试函数，用于测试FortranFile类处理正常情况下的文件结束符（EOF）
def test_fortran_eof_ok(tmpdir):
    # 创建临时文件路径
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    
    # 写入数据到文件
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    
    # 读取数据并进行断言
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        assert len(f.read_reals()) == 3
        # 使用pytest断言应该抛出FortranEOFError异常
        with pytest.raises(FortranEOFError):
            f.read_reals()


# 定义一个测试函数，用于测试FortranFile类处理文件大小不正确的情况
def test_fortran_eof_broken_size(tmpdir):
    # 创建临时文件路径
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    
    # 写入数据到文件
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    
    # 在文件末尾追加一个错误的字节
    with open(filename, "ab") as f:
        f.write(b"\xff")
    
    # 读取数据并进行断言
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        assert len(f.read_reals()) == 3
        # 使用pytest断言应该抛出FortranFormattingError异常
        with pytest.raises(FortranFormattingError):
            f.read_reals()
    # 设置随机数种子为1，用于确保随机数生成的可重复性
    np.random.seed(1)
    # 使用 FortranFile 打开文件以进行写入操作
    with FortranFile(filename, 'w') as f:
        # 将一组随机生成的5个标准正态分布数写入文件
        f.write_record(np.random.randn(5))
        # 将一组随机生成的3个标准正态分布数写入文件
        f.write_record(np.random.randn(3))
    # 以二进制读写模式再次打开文件
    with open(filename, "w+b") as f:
        # 向文件中写入十六进制表示的字节序列 b"\xff\xff"
        f.write(b"\xff\xff")
    # 使用 FortranFile 打开文件以进行读取操作
    with FortranFile(filename, 'r') as f:
        # 使用 pytest 的异常断言，期望捕获 Fortran 格式错误异常
        with pytest.raises(FortranFormattingError):
            # 读取文件中的实数数据
            f.read_reals()
# 定义测试函数，测试在处理 Fortran 文件时是否能正确处理意外的文件结尾问题
def test_fortran_eof_broken_record(tmpdir):
    # 创建临时文件路径
    filename = path.join(str(tmpdir), "scratch")
    # 设定随机数种子为1，以确保结果可重现
    np.random.seed(1)
    
    # 使用 'w' 模式打开 Fortran 文件对象，写入一个随机数数组并写入记录
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    
    # 以追加二进制模式打开文件，截断文件末尾的20个字节
    with open(filename, "ab") as f:
        f.truncate(path.getsize(filename)-20)
    
    # 再次使用 'r' 模式打开 Fortran 文件对象
    with FortranFile(filename, 'r') as f:
        # 断言读取实数数组的长度为5，验证是否正确处理了意外的文件结尾
        assert len(f.read_reals()) == 5
        # 使用 pytest 断言，预期会抛出 FortranFormattingError 异常
        with pytest.raises(FortranFormattingError):
            f.read_reals()


# 定义测试函数，测试在处理多维数组时是否能正确处理意外的文件结尾问题
def test_fortran_eof_multidimensional(tmpdir):
    # 创建临时文件路径
    filename = path.join(str(tmpdir), "scratch")
    n, m, q = 3, 5, 7
    # 定义数据类型，包含一个名为 "field" 的多维数组，元素为 np.float64 类型，大小为 (n, m)
    dt = np.dtype([("field", np.float64, (n, m))])
    # 创建一个形状为 (q,) 的零数组，使用上述数据类型
    a = np.zeros(q, dtype=dt)
    
    # 使用 'w' 模式打开 Fortran 文件对象，依次写入数组的第一个元素、整个数组两次
    with FortranFile(filename, 'w') as f:
        f.write_record(a[0])
        f.write_record(a)
        f.write_record(a)
    
    # 以追加二进制模式打开文件，截断文件末尾的20个字节
    with open(filename, "ab") as f:
        f.truncate(path.getsize(filename)-20)
    
    # 再次使用 'r' 模式打开 Fortran 文件对象
    with FortranFile(filename, 'r') as f:
        # 断言读取记录的返回长度为1，验证是否正确处理了意外的文件结尾
        assert len(f.read_record(dtype=dt)) == 1
        # 断言再次读取记录的返回长度为q，验证是否正确处理了意外的文件结尾
        assert len(f.read_record(dtype=dt)) == q
        # 使用 pytest 断言，预期会抛出 FortranFormattingError 异常
        with pytest.raises(FortranFormattingError):
            f.read_record(dtype=dt)
```