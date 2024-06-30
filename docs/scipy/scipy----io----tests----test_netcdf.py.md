# `D:\src\scipysrc\scipy\scipy\io\tests\test_netcdf.py`

```
''' Tests for netcdf '''
# 导入所需的模块和库
import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager

# 导入 numpy 库，并从中导入测试相关的函数
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           break_cycles, suppress_warnings, IS_PYPY)
# 导入 pytest 中的 raises 函数，用于断言异常
from pytest import raises as assert_raises

# 导入 scipy 库中的 netcdf_file 模块
from scipy.io import netcdf_file
# 导入 scipy 库中的临时目录管理模块
from scipy._lib._tmpdirs import in_tempdir

# 定义测试数据路径常量
TEST_DATA_PATH = pjoin(dirname(__file__), 'data')

# 设置示例变量的元素数量和类型
N_EG_ELS = 11  # number of elements for example variable
VARTYPE_EG = 'b'  # var type for example variable


@contextmanager
def make_simple(*args, **kwargs):
    # 创建 netCDF 文件对象
    f = netcdf_file(*args, **kwargs)
    # 设置文件的历史属性
    f.history = 'Created for a test'
    # 创建名为 'time' 的维度，长度为 N_EG_ELS
    f.createDimension('time', N_EG_ELS)
    # 在文件中创建名为 'time' 的变量，并赋值为从 0 到 N_EG_ELS-1 的数组
    time = f.createVariable('time', VARTYPE_EG, ('time',))
    time[:] = np.arange(N_EG_ELS)
    # 设置 'time' 变量的单位属性
    time.units = 'days since 2008-01-01'
    # 刷新文件内容到磁盘
    f.flush()
    # 返回文件对象
    yield f
    # 关闭文件对象
    f.close()


def check_simple(ncfileobj):
    '''Example fileobj tests '''
    # 断言文件对象的历史属性与预期相符
    assert_equal(ncfileobj.history, b'Created for a test')
    # 获取 'time' 变量并进行相关断言
    time = ncfileobj.variables['time']
    assert_equal(time.units, b'days since 2008-01-01')
    assert_equal(time.shape, (N_EG_ELS,))
    assert_equal(time[-1], N_EG_ELS-1)

def assert_mask_matches(arr, expected_mask):
    '''
    Asserts that the mask of arr is effectively the same as expected_mask.

    In contrast to numpy.ma.testutils.assert_mask_equal, this function allows
    testing the 'mask' of a standard numpy array (the mask in this case is treated
    as all False).

    Parameters
    ----------
    arr : ndarray or MaskedArray
        Array to test.
    expected_mask : array_like of booleans
        A list giving the expected mask.
    '''

    # 获取 arr 的掩码数组，并与预期的掩码数组进行断言
    mask = np.ma.getmaskarray(arr)
    assert_equal(mask, expected_mask)


def test_read_write_files():
    # test round trip for example file
    # 获取当前工作目录
    cwd = os.getcwd()
    # 尝试在临时目录中创建临时文件夹
    try:
        tmpdir = tempfile.mkdtemp()
        # 切换当前工作目录至临时文件夹
        os.chdir(tmpdir)
        # 使用 'w' 模式创建一个名为 'simple.nc' 的 NetCDF 文件对象，并什么也不做
        with make_simple('simple.nc', 'w') as f:
            pass
        
        # 在 'a' 模式下读取刚刚创建的文件
        with netcdf_file('simple.nc', 'a') as f:
            # 检查简单的内容
            check_simple(f)
            # 添加一个属性 'appendRan' 到文件对象的私有属性中
            f._attributes['appendRan'] = 1

        # 以默认方式（使用 mmap，但不在 PyPy 上使用）读取刚创建的 NetCDF 文件
        with netcdf_file('simple.nc') as f:
            # 确保使用 mmap（除非在 PyPy 上）
            assert_equal(f.use_mmap, not IS_PYPY)
            check_simple(f)
            # 确保刚添加的属性值正确
            assert_equal(f._attributes['appendRan'], 1)

        # 以 'a' 模式读取文件，并检查 mmap 是否关闭
        with netcdf_file('simple.nc', 'a') as f:
            assert_(not f.use_mmap)
            check_simple(f)
            assert_equal(f._attributes['appendRan'], 1)

        # 现在禁用 mmap，再次读取文件
        with netcdf_file('simple.nc', mmap=False) as f:
            assert_(not f.use_mmap)
            check_simple(f)

        # 从文件对象中读取 NetCDF 文件，不使用 mmap
        # 当 n * n_bytes(var_type) 不能被 4 整除时，会在 pupynere 1.0.12 和 scipy rev 5893 中引发错误
        # 因为计算的 vsize 在 4 的单位上四舍五入
        with open('simple.nc', 'rb') as fobj:
            with netcdf_file(fobj) as f:
                # 默认情况下，对文件对象不使用 mmap
                assert_(not f.use_mmap)
                check_simple(f)

        # 使用 mmap 从文件对象中读取文件
        with suppress_warnings() as sup:
            if IS_PYPY:
                # 在 PyPy 中无法关闭使用 mmap=True 打开的 netcdf_file 的警告
                sup.filter(RuntimeWarning,
                           "Cannot close a netcdf_file opened with mmap=True.*")
            with open('simple.nc', 'rb') as fobj:
                with netcdf_file(fobj, mmap=True) as f:
                    assert_(f.use_mmap)
                    check_simple(f)

        # 再次以 'a' 模式打开文件（添加另一个属性）
        with open('simple.nc', 'r+b') as fobj:
            with netcdf_file(fobj, 'a') as f:
                assert_(not f.use_mmap)
                check_simple(f)
                # 创建一个名为 'app_dim' 的维度
                f.createDimension('app_dim', 1)
                # 创建一个名为 'app_var' 的变量，类型为整型 'i'，维度为 ('app_dim',)
                var = f.createVariable('app_var', 'i', ('app_dim',))
                var[:] = 42

        # 确保 'app_var' 变量被正确添加进去
        with netcdf_file('simple.nc') as f:
            check_simple(f)
            assert_equal(f.variables['app_var'][:], 42)

    finally:
        if IS_PYPY:
            # 在 PyPy 中，Windows 无法移除由 mmap 持有的未收集的死文件
            break_cycles()
            break_cycles()
        # 恢复当前工作目录并删除临时文件夹
        os.chdir(cwd)
        shutil.rmtree(tmpdir)
# 定义一个测试函数，用于测试读写 BytesIO 的操作
def test_read_write_sio():
    # 创建一个空的 BytesIO 对象
    eg_sio1 = BytesIO()
    # 使用 make_simple 函数向 eg_sio1 写入数据，模式为 'w'
    with make_simple(eg_sio1, 'w'):
        # 获取 eg_sio1 中的数据并赋值给 str_val
        str_val = eg_sio1.getvalue()

    # 创建一个新的 BytesIO 对象，用之前写入的 str_val 数据初始化
    eg_sio2 = BytesIO(str_val)
    # 使用 netcdf_file 打开 eg_sio2，模式为 'r'，并将返回的文件对象赋给 f2
    with netcdf_file(eg_sio2) as f2:
        # 检查 f2 的内容是否符合预期
        check_simple(f2)

    # 测试：如果尝试为 sio 使用 mmap，则应该抛出 ValueError
    eg_sio3 = BytesIO(str_val)
    # 断言：调用 netcdf_file 函数时，传入 eg_sio3 和模式 'r'，同时启用 mmap，预期抛出 ValueError
    assert_raises(ValueError, netcdf_file, eg_sio3, 'r', True)

    # 测试 64 位偏移的写入和读取
    eg_sio_64 = BytesIO()
    # 使用 make_simple 函数向 eg_sio_64 写入数据，模式为 'w'，版本为 2
    with make_simple(eg_sio_64, 'w', version=2) as f_64:
        # 获取 eg_sio_64 中的数据并赋值给 str_val
        str_val = eg_sio_64.getvalue()

    # 创建一个新的 BytesIO 对象，用之前写入的 str_val 数据初始化
    eg_sio_64 = BytesIO(str_val)
    # 使用 netcdf_file 打开 eg_sio_64，返回文件对象赋给 f_64
    with netcdf_file(eg_sio_64) as f_64:
        # 检查 f_64 的内容是否符合预期
        check_simple(f_64)
        # 断言：f_64 的版本号应为 2
        assert_equal(f_64.version_byte, 2)

    # 再次测试：当显式指定版本为 2 时
    eg_sio_64 = BytesIO(str_val)
    # 使用 netcdf_file 打开 eg_sio_64，传入版本号为 2，返回文件对象赋给 f_64
    with netcdf_file(eg_sio_64, version=2) as f_64:
        # 检查 f_64 的内容是否符合预期
        check_simple(f_64)
        # 断言：f_64 的版本号应为 2
        assert_equal(f_64.version_byte, 2)


# 定义一个测试函数，用于测试 BytesIO 的使用情况
def test_bytes():
    # 创建一个空的 BytesIO 对象
    raw_file = BytesIO()
    # 使用 netcdf_file 打开 raw_file，模式为 'w'，返回文件对象赋给 f
    f = netcdf_file(raw_file, mode='w')
    # 设置属性 'a' 为 'b'
    f.a = 'b'
    # 创建一个维度 'dim'，大小为 1
    f.createDimension('dim', 1)
    # 创建一个名为 'var' 的变量，类型为 np.int16，维度为 ('dim',)
    var = f.createVariable('var', np.int16, ('dim',))
    # 设置变量 var 的第一个元素为 -9999
    var[0] = -9999
    # 设置变量 var 的属性 'c' 为 'd'
    var.c = 'd'
    # 将数据同步到文件
    f.sync()

    # 获取 raw_file 中的所有数据并赋值给 actual
    actual = raw_file.getvalue()

    # 预期的数据格式，以字节表示
    expected = (b'CDF\x01'
                b'\x00\x00\x00\x00'
                b'\x00\x00\x00\x0a'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x03'
                b'dim\x00'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x0c'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x01'
                b'a\x00\x00\x00'
                b'\x00\x00\x00\x02'
                b'\x00\x00\x00\x01'
                b'b\x00\x00\x00'
                b'\x00\x00\x00\x0b'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x03'
                b'var\x00'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x00'
                b'\x00\x00\x00\x0c'
                b'\x00\x00\x00\x01'
                b'\x00\x00\x00\x01'
                b'c\x00\x00\x00'
                b'\x00\x00\x00\x02'
                b'\x00\x00\x00\x01'
                b'd\x00\x00\x00'
                b'\x00\x00\x00\x03'
                b'\x00\x00\x00\x04'
                b'\x00\x00\x00\x78'
                b'\xd8\xf1\x80\x01')

    # 断言：actual 应该等于 expected
    assert_equal(actual, expected)


# 定义一个测试函数，用于测试编码的填充值
def test_encoded_fill_value():
    # 使用 netcdf_file 打开一个空的 BytesIO 对象，模式为 'w'，返回文件对象赋给 f
    with netcdf_file(BytesIO(), mode='w') as f:
        # 创建一个维度 'x'，大小为 1
        f.createDimension('x', 1)
        # 创建一个名为 'var' 的变量，类型为 'S1'（单字节字符串），维度为 ('x',)
        var = f.createVariable('var', 'S1', ('x',))
        # 断言：var 的编码填充值应该是 b'\x00'
        assert_equal(var._get_encoded_fill_value(), b'\x00')
        # 设置 var 的 _FillValue 属性为 b'\x01'
        var._FillValue = b'\x01'
        # 断言：var 的编码填充值应该是 b'\x01'
        assert_equal(var._get_encoded_fill_value(), b'\x01')
        # 设置 var 的 _FillValue 属性为 b'\x00\x00'（无效值，长度错误）
        var._FillValue = b'\x00\x00'
        # 断言：var 的编码填充值应该是 b'\x00'
        assert_equal(var._get_encoded_fill_value(), b'\x00')


# 定义一个测试函数，用于读取示例数据文件
def test_read_example_data():
    # 读取任何示例数据文件
    # 使用 glob 模块获取指定目录下所有以 '.nc' 结尾的文件名列表
    for fname in glob(pjoin(TEST_DATA_PATH, '*.nc')):
        # 使用 netcdf_file 打开指定文件名的 NetCDF 文件以只读模式
        with netcdf_file(fname, 'r'):
            # 空语句，仅用于在 with 块内执行完毕后自动关闭文件
            pass
        # 使用 netcdf_file 打开指定文件名的 NetCDF 文件以只读模式，并关闭内存映射
        with netcdf_file(fname, 'r', mmap=False):
            # 空语句，仅用于在 with 块内执行完毕后自动关闭文件
            pass
# 测试项集中防止只读状态下的段错误问题
def test_itemset_no_segfault_on_readonly():
    # 对于 #1202 号问题的回归测试。
    # 以只读模式打开测试文件

    # 拼接测试数据路径和文件名
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    # 使用 suppress_warnings 上下文管理器，过滤特定的 RuntimeWarning
    with suppress_warnings() as sup:
        # 定义特定警告消息，用于过滤
        message = ("Cannot close a netcdf_file opened with mmap=True, when "
                   "netcdf_variables or arrays referring to its data still exist")
        sup.filter(RuntimeWarning, message)
        # 使用 netcdf_file 打开文件，以只读模式，使用 mmap=True
        with netcdf_file(filename, 'r', mmap=True) as f:
            # 获取 'time' 变量
            time_var = f.variables['time']

    # time_var.assignValue(42) 应该引发 RuntimeError，而不是段错误！
    assert_raises(RuntimeError, time_var.assignValue, 42)


# 测试在附加问题 #8625 中的处理
def test_appending_issue_gh_8625():
    # 创建一个字节流对象
    stream = BytesIO()

    # 使用 make_simple 上下文管理器，以写入模式打开流
    with make_simple(stream, mode='w') as f:
        # 创建维度 'x'
        f.createDimension('x', 2)
        # 创建变量 'x'，类型为 float，维度为 ('x',)
        f.createVariable('x', float, ('x',))
        # 设置变量 'x' 的值为 1
        f.variables['x'][...] = 1
        # 刷新流
        f.flush()
        # 获取流的内容
        contents = stream.getvalue()

    # 使用新的字节流对象，重新打开 netcdf 文件，以附加模式 ('a') 打开
    stream = BytesIO(contents)
    with netcdf_file(stream, mode='a') as f:
        # 设置变量 'x' 的值为 2
        f.variables['x'][...] = 2


# 测试写入无效数据类型的处理
def test_write_invalid_dtype():
    # 定义数据类型列表
    dtypes = ['int64', 'uint64']
    # 如果是 64 位机器，添加额外的数据类型 'int' 和 'uint'
    if np.dtype('int').itemsize == 8:   # 64 位机器
        dtypes.append('int')
    if np.dtype('uint').itemsize == 8:   # 64 位机器
        dtypes.append('uint')

    # 使用 BytesIO 创建一个空的 netcdf 文件，以写入模式 ('w') 打开
    with netcdf_file(BytesIO(), 'w') as f:
        # 创建维度 'time'，其大小为 N_EG_ELS
        f.createDimension('time', N_EG_ELS)
        # 遍历数据类型列表
        for dt in dtypes:
            # 断言应该引发 ValueError，尝试创建变量 'time' 使用无效的数据类型
            assert_raises(ValueError, f.createVariable, 'time', dt, ('time',))


# 测试 flush 和 rewind 的效果
def test_flush_rewind():
    # 创建一个字节流对象
    stream = BytesIO()
    # 使用 make_simple 上下文管理器，以写入模式 ('w') 打开流
    with make_simple(stream, mode='w') as f:
        # 创建维度 'x'，在创建变量 'v' 时使用
        f.createDimension('x', 4)
        # 创建变量 'v'，类型为 'i2'，维度为 ['x']
        v = f.createVariable('v', 'i2', ['x'])
        # 将变量 'v' 的所有元素设置为 1
        v[:] = 1
        # 刷新流
        f.flush()
        # 获取刷新后流的长度
        len_single = len(stream.getvalue())
        # 再次刷新流
        f.flush()
        # 获取再次刷新后流的长度
        len_double = len(stream.getvalue())

    # 断言单次刷新和双次刷新后流的长度相同
    assert_(len_single == len_double)


# 测试数据类型的指定
def test_dtype_specifiers():
    # Numpy 1.7.0-dev 中存在 'i2' 无法正常工作的 bug。
    # 从与此注释相同的提交开始，只有指定 np.int16 等类型才有效。
    with make_simple(BytesIO(), mode='w') as f:
        # 创建维度 'x'，大小为 4
        f.createDimension('x', 4)
        # 创建变量 'v1'，类型为 'i2'，维度为 ['x']
        f.createVariable('v1', 'i2', ['x'])
        # 创建变量 'v2'，类型为 np.int16，维度为 ['x']
        f.createVariable('v2', np.int16, ['x'])
        # 创建变量 'v3'，类型为 np.int16，维度为 ['x']
        f.createVariable('v3', np.dtype(np.int16), ['x'])


# 测试票号 #1720
def test_ticket_1720():
    # 创建一个字节流对象
    io = BytesIO()

    # 定义测试项列表
    items = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # 使用 netcdf_file 打开字节流对象，以写入模式 ('w') 打开
    with netcdf_file(io, 'w') as f:
        # 设置属性 'history'
        f.history = 'Created for a test'
        # 创建维度 'float_var'，大小为 10
        f.createDimension('float_var', 10)
        # 创建变量 'float_var'，类型为 'f'，维度为 ('float_var',)
        float_var = f.createVariable('float_var', 'f', ('float_var',))
        # 将变量 'float_var' 的值设置为 items 列表的值
        float_var[:] = items
        # 设置变量 'float_var' 的单位为 'metres'
        float_var.units = 'metres'
        # 刷新数据到文件
        f.flush()
        # 获取流的内容
        contents = io.getvalue()

    # 使用新的字节流对象，重新打开 netcdf 文件，以只读模式 ('r') 打开
    io = BytesIO(contents)
    with netcdf_file(io, 'r') as f:
        # 断言属性 'history' 的值应为 b'Created for a test'
        assert_equal(f.history, b'Created for a test')
        # 获取变量 'float_var'
        float_var = f.variables['float_var']
        # 断言变量 'float_var' 的单位应为 b'metres'
        assert_equal(float_var.units, b'metres')
        # 断言变量 'float_var' 的形状应为 (10,)
        assert_equal(float_var.shape, (10,))
        # 断言变量 'float_var' 的值应接近于 items 列表的值
        assert_allclose(float_var[:], items)
# 定义一个测试函数，用于测试在 mmap 模式下读取 netCDF 文件时的行为
def test_mmaps_segfault():
    # 设置测试文件名为 TEST_DATA_PATH 下的 example_1.nc
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')

    # 如果不是在 PyPy 环境下
    if not IS_PYPY:
        # 使用警告捕获机制
        with warnings.catch_warnings():
            # 将警告级别设置为异常
            warnings.simplefilter("error")
            # 打开 netCDF 文件，使用 mmap 模式
            with netcdf_file(filename, mmap=True) as f:
                # 读取 'lat' 变量的所有数据
                x = f.variables['lat'][:]
                # 应该不会触发警告
                del x

    # 定义内部函数 doit
    def doit():
        # 打开 netCDF 文件，使用 mmap 模式
        with netcdf_file(filename, mmap=True) as f:
            # 返回 'lat' 变量的所有数据
            return f.variables['lat'][:]

    # 应该不会导致崩溃
    with suppress_warnings() as sup:
        # 设置特定警告消息
        message = ("Cannot close a netcdf_file opened with mmap=True, when "
                   "netcdf_variables or arrays referring to its data still exist")
        # 过滤特定消息的 RuntimeWarning 警告
        sup.filter(RuntimeWarning, message)
        # 执行 doit 函数，并将结果赋给 x
        x = doit()
    # 对 x 进行求和操作
    x.sum()


# 定义一个测试函数，用于测试零维变量的行为
def test_zero_dimensional_var():
    # 创建一个空的字节流对象
    io = BytesIO()
    # 使用 make_simple 函数向字节流中写入数据，并使用 'w' 模式打开
    with make_simple(io, 'w') as f:
        # 创建一个名为 'zerodim' 的零维整数型变量
        v = f.createVariable('zerodim', 'i2', [])
        # 断言检查 .isrec 属性是否为 False，不要简化为 'assert not ...'
        assert v.isrec is False, v.isrec
        # 刷新文件缓冲区
        f.flush()


# 定义一个测试函数，用于测试全局字节类型的属性
def test_byte_gatts():
    # 在临时目录中进行操作
    with in_tempdir():
        # 定义文件名为 'g_byte_atts.nc'，以 'w' 模式创建 netCDF 文件
        filename = 'g_byte_atts.nc'
        f = netcdf_file(filename, 'w')
        # 设置全局属性 '_attributes' 中的 'holy' 属性为字节串 b'grail'
        f._attributes['holy'] = b'grail'
        # 设置全局属性 '_attributes' 中的 'witch' 属性为字符串 'floats'
        f._attributes['witch'] = 'floats'
        # 关闭 netCDF 文件
        f.close()

        # 以 'r' 模式重新打开 netCDF 文件
        f = netcdf_file(filename, 'r')
        # 断言检查 'holy' 属性是否与预期的字节串 b'grail' 相等
        assert_equal(f._attributes['holy'], b'grail')
        # 断言检查 'witch' 属性是否与预期的字节串 b'floats' 相等
        assert_equal(f._attributes['witch'], b'floats')
        # 再次关闭 netCDF 文件
        f.close()


# 定义一个测试函数，用于测试在 'a' 模式下打开 netCDF 文件的行为
def test_open_append():
    # 在临时目录中进行操作
    with in_tempdir():
        # 定义文件名为 'append_dat.nc'，以 'w' 模式创建 netCDF 文件
        filename = 'append_dat.nc'
        f = netcdf_file(filename, 'w')
        # 设置全局属性 '_attributes' 中的 'Kilroy' 属性为字符串 'was here'
        f._attributes['Kilroy'] = 'was here'
        # 关闭 netCDF 文件
        f.close()

        # 以 'a' 模式重新打开 netCDF 文件
        f = netcdf_file(filename, 'a')
        # 断言检查 'Kilroy' 属性是否与预期的字节串 b'was here' 相等
        assert_equal(f._attributes['Kilroy'], b'was here')
        # 设置全局属性 '_attributes' 中的 'naughty' 属性为字节串 b'Zoot'
        f._attributes['naughty'] = b'Zoot'
        # 再次关闭 netCDF 文件
        f.close()

        # 以 'r' 模式重新打开 netCDF 文件
        f = netcdf_file(filename, 'r')
        # 断言检查 'Kilroy' 属性是否与预期的字节串 b'was here' 相等
        assert_equal(f._attributes['Kilroy'], b'was here')
        # 断言检查 'naughty' 属性是否与预期的字节串 b'Zoot' 相等
        assert_equal(f._attributes['naughty'], b'Zoot')
        # 再次关闭 netCDF 文件
        f.close()


# 定义一个测试函数，用于测试添加记录维度的行为
def test_append_recordDimension():
    # 数据大小设定为 100
    dataSize = 100
    with in_tempdir():
        # 在临时目录中创建文件 'withRecordDimension.nc'，并添加时间维度
        with netcdf_file('withRecordDimension.nc', 'w') as f:
            # 创建时间维度，长度可变
            f.createDimension('time', None)
            # 创建名为 'time' 的变量，类型为双精度浮点数，维度为 ('time',)
            f.createVariable('time', 'd', ('time',))
            # 创建维度为 'x'，大小为 dataSize
            f.createDimension('x', dataSize)
            # 创建名为 'x' 的变量，类型为双精度浮点数，维度为 ('x',)
            x = f.createVariable('x', 'd', ('x',))
            # 将数组 [0, 1, ..., dataSize-1] 赋值给变量 x
            x[:] = np.array(range(dataSize))
            # 创建维度为 'y'，大小为 dataSize
            f.createDimension('y', dataSize)
            # 创建名为 'y' 的变量，类型为双精度浮点数，维度为 ('y',)
            y = f.createVariable('y', 'd', ('y',))
            # 将数组 [0, 1, ..., dataSize-1] 赋值给变量 y
            y[:] = np.array(range(dataSize))
            # 创建名为 'testData' 的变量，类型为整型，维度为 ('time', 'x', 'y')
            f.createVariable('testData', 'i', ('time', 'x', 'y'))
            # 刷新文件，确保数据写入
            f.flush()
            # 关闭文件
            f.close()

        for i in range(2):
            # 以附加模式打开文件，并添加数据
            with netcdf_file('withRecordDimension.nc', 'a') as f:
                # 向 'time' 变量的数据数组追加 i
                f.variables['time'].data = np.append(f.variables["time"].data, i)
                # 向 'testData' 变量的最后一个时间步骤写入全为 i 的数据
                f.variables['testData'][i, :, :] = np.full((dataSize, dataSize), i)
                # 刷新文件
                f.flush()

            # 读取文件并检查附加操作是否成功
            with netcdf_file('withRecordDimension.nc') as f:
                # 断言最后一个 'time' 变量的值是否为 i
                assert_equal(f.variables['time'][-1], i)
                # 断言最后一个 'testData' 变量的数据是否与全为 i 的数组相等
                assert_equal(f.variables['testData'][-1, :, :].copy(),
                             np.full((dataSize, dataSize), i))
                # 断言 'time' 变量数据数组的长度是否为 i+1
                assert_equal(f.variables['time'].data.shape[0], i+1)
                # 断言 'testData' 变量数据数组的第一个维度长度是否为 i+1
                assert_equal(f.variables['testData'].data.shape[0], i+1)

        # 读取文件并检查在附加操作期间 'data' 是否未保存为 'testData' 变量的用户定义属性
        with netcdf_file('withRecordDimension.nc') as f:
            with assert_raises(KeyError) as ar:
                # 断言 'testData' 变量的用户定义属性 'data' 是否存在
                f.variables['testData']._attributes['data']
            ex = ar.value
            # 断言异常的消息是否为 'data'
            assert_equal(ex.args[0], 'data')
# ------------------------------------------------------------------------
# Test reading with masked values (_FillValue / missing_value)
# ------------------------------------------------------------------------

# 测试读取带有屏蔽值（_FillValue / missing_value）的情况

def test_read_withValuesNearFillValue():
    # Regression test for ticket #5626
    # 回归测试，用于检查问题票号 #5626

    # 拼接测试数据路径和文件名
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')

    # 使用 netcdf_file 打开文件，并启用 maskandscale 模式
    with netcdf_file(fname, maskandscale=True) as f:
        # 读取名为 'var1_fillval0' 的变量数据
        vardata = f.variables['var1_fillval0'][:]
        
        # 断言检查变量数据的屏蔽匹配情况，预期为 [False, True, False]
        assert_mask_matches(vardata, [False, True, False])

def test_read_withNoFillValue():
    # For a variable with no fill value, reading data with maskandscale=True
    # should return unmasked data
    # 对于没有填充值的变量，使用 maskandscale=True 读取数据应返回未屏蔽的数据
    
    # 拼接测试数据路径和文件名
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')

    # 使用 netcdf_file 打开文件，并启用 maskandscale 模式
    with netcdf_file(fname, maskandscale=True) as f:
        # 读取名为 'var2_noFillval' 的变量数据
        vardata = f.variables['var2_noFillval'][:]
        
        # 断言检查变量数据的屏蔽匹配情况，预期为 [False, False, False]
        assert_mask_matches(vardata, [False, False, False])
        
        # 断言检查变量数据的值，预期为 [1, 2, 3]
        assert_equal(vardata, [1, 2, 3])

def test_read_withFillValueAndMissingValue():
    # For a variable with both _FillValue and missing_value, the _FillValue
    # should be used
    # 对于同时具有 _FillValue 和 missing_value 的变量，应使用 _FillValue
    
    # 定义一个无关紧要的值
    IRRELEVANT_VALUE = 9999
    
    # 拼接测试数据路径和文件名
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')

    # 使用 netcdf_file 打开文件，并启用 maskandscale 模式
    with netcdf_file(fname, maskandscale=True) as f:
        # 读取名为 'var3_fillvalAndMissingValue' 的变量数据
        vardata = f.variables['var3_fillvalAndMissingValue'][:]
        
        # 断言检查变量数据的屏蔽匹配情况，预期为 [True, False, False]
        assert_mask_matches(vardata, [True, False, False])
        
        # 断言检查变量数据的值，预期为 [IRRELEVANT_VALUE, 2, 3]
        assert_equal(vardata, [IRRELEVANT_VALUE, 2, 3])

def test_read_withMissingValue():
    # For a variable with missing_value but not _FillValue, the missing_value
    # should be used
    # 对于具有 missing_value 但没有 _FillValue 的变量，应使用 missing_value
    
    # 拼接测试数据路径和文件名
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    # 使用 netcdf_file 函数打开指定的 NetCDF 文件 `fname`，并使用 maskandscale=True 参数
    # 这将返回一个文件对象 `f`，可以使用 `with` 语句来确保在使用完后自动关闭文件
    with netcdf_file(fname, maskandscale=True) as f:
        # 从文件对象 `f` 中获取名为 'var4_missingValue' 的变量数据并赋值给 `vardata`
        vardata = f.variables['var4_missingValue'][:]
        # 使用断言确保 `vardata` 的值与指定的预期值匹配，即 [False, True, False]
        assert_mask_matches(vardata, [False, True, False])
# 测试读取具有 NaN 填充值的变量数据
def test_read_withFillValNaN():
    # 构造文件路径
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    # 打开 NetCDF 文件并设置 maskandscale=True 来处理缺失值和比例尺
    with netcdf_file(fname, maskandscale=True) as f:
        # 从文件中读取名为 'var5_fillvalNaN' 的变量数据
        vardata = f.variables['var5_fillvalNaN'][:]
        # 使用自定义函数检查数据的掩码是否匹配预期结果
        assert_mask_matches(vardata, [False, True, False])

# 测试读取具有字符数据的变量
def test_read_withChar():
    # 构造文件路径
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    # 打开 NetCDF 文件并设置 maskandscale=True 来处理缺失值和比例尺
    with netcdf_file(fname, maskandscale=True) as f:
        # 从文件中读取名为 'var6_char' 的变量数据
        vardata = f.variables['var6_char'][:]
        # 使用自定义函数检查数据的掩码是否匹配预期结果
        assert_mask_matches(vardata, [False, True, False])

# 测试读取具有二维变量数据的变量
def test_read_with2dVar():
    # 构造文件路径
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    # 打开 NetCDF 文件并设置 maskandscale=True 来处理缺失值和比例尺
    with netcdf_file(fname, maskandscale=True) as f:
        # 从文件中读取名为 'var7_2d' 的变量数据
        vardata = f.variables['var7_2d'][:]
        # 使用自定义函数检查数据的掩码是否匹配预期结果
        assert_mask_matches(vardata, [[True, False], [False, False], [False, True]])

# 测试读取具有禁用 maskandscale 的变量数据
def test_read_withMaskAndScaleFalse():
    # 如果变量具有 _FillValue（或 missing_value）属性，但使用 maskandscale=False 读取，
    # 结果应为未屏蔽状态
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    # 打开文件，设置 maskandscale=False 和 mmap=False 以避免关闭 mmap 文件时的问题
    with netcdf_file(fname, maskandscale=False, mmap=False) as f:
        # 从文件中读取名为 'var3_fillvalAndMissingValue' 的变量数据
        vardata = f.variables['var3_fillvalAndMissingValue'][:]
        # 使用自定义函数检查数据的掩码是否匹配预期结果
        assert_mask_matches(vardata, [False, False, False])
        # 使用 assert_equal 检查数据是否与预期的数值列表完全一致
        assert_equal(vardata, [1, 2, 3])
```