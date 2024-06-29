# `.\numpy\numpy\_core\tests\test_cython.py`

```
# 导入 datetime 模块中的 datetime 类，用于处理日期和时间
from datetime import datetime
# 导入 os 模块，提供了访问操作系统功能的接口
import os
# 导入 shutil 模块，提供了高级文件操作功能
import shutil
# 导入 subprocess 模块，用于执行系统命令
import subprocess
# 导入 sys 模块，提供了访问 Python 解释器及其环境的变量和函数
import sys
# 导入 time 模块，提供了处理时间的功能
import time
# 导入 pytest 模块，用于编写和运行测试
import pytest

# 导入 numpy 库，用于数值计算
import numpy as np
# 从 numpy.testing 模块中导入 assert_array_equal 函数，用于比较两个数组是否相等
from numpy.testing import assert_array_equal, IS_WASM, IS_EDITABLE

# 这里是从 random.tests.test_extending 中复制的导入语句
# 尝试导入 cython 模块，用于编写 C 扩展
try:
    import cython
    # 从 Cython.Compiler.Version 中导入 version 属性，获取 Cython 的版本信息
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    # 导入 numpy._utils 模块中的 _pep440 模块
    from numpy._utils import _pep440

    # 定义需要的 Cython 最低版本
    required_version = "3.0.6"
    # 检查当前 Cython 的版本是否满足最低要求
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # 如果版本过低或者 Cython 未安装，则将 cython 设置为 None，跳过测试
        cython = None

# 使用 pytest.mark.skipif 标记，如果 cython 为 None，则跳过测试，给出原因
pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")

# 如果 IS_EDITABLE 为真，则跳过测试，给出原因
if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )

# 使用 pytest.fixture 装饰器定义一个作用域为 'module' 的测试夹具
@pytest.fixture(scope='module')
def install_temp(tmpdir_factory):
    # 基于 random.tests.test_extending 中的 test_cython 函数实现部分功能
    # 如果 IS_WASM 为真，则跳过测试，给出原因
    if IS_WASM:
        pytest.skip("No subprocess")

    # 定义源代码目录，应该是 examples/cython 的路径
    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'cython')
    # 创建一个临时构建目录，用于编译过程中的文件生成
    build_dir = tmpdir_factory.mktemp("cython_test") / "build"
    # 确保构建目录存在，如果不存在则创建
    os.makedirs(build_dir, exist_ok=True)
    # 创建一个文件用于指定正确的 Python 解释器路径，即使在不同环境中也能正确使用 'meson'
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'")

    try:
        # 检查是否能够调用 'meson' 命令
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        # 如果找不到 'meson' 命令，则跳过测试，给出原因
        pytest.skip("No usable 'meson' found")

    # 根据操作系统类型选择不同的 'meson setup' 命令进行调用
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup",
                               "--native-file", native_file, str(srcdir)],
                              cwd=build_dir
                              )

    try:
        # 调用 'meson compile' 命令进行编译
        subprocess.check_call(["meson", "compile", "-vv"], cwd=build_dir)
    except subprocess.CalledProcessError:
        # 如果编译过程中出现错误，则打印错误信息并抛出异常
        print("----------------")
        print("meson build failed when doing")
        print(f"'meson setup --native-file {native_file} {srcdir}'")
        print(f"'meson compile -vv'")
        print(f"in {build_dir}")
        print("----------------")
        raise

    # 将构建目录添加到系统路径中，使得测试代码可以访问编译后的模块
    sys.path.append(str(build_dir))

# 定义一个测试函数，测试是否为 timedelta64 对象
def test_is_timedelta64_object(install_temp):
    # 导入自定义的 checks 模块
    import checks

    # 使用 assert 语句检查是否是 timedelta64 对象
    assert checks.is_td64(np.timedelta64(1234))
    assert checks.is_td64(np.timedelta64(1234, "ns"))
    assert checks.is_td64(np.timedelta64("NaT", "ns"))

    # 使用 assert 语句检查是否不是 timedelta64 对象
    assert not checks.is_td64(1)
    assert not checks.is_td64(None)
    # 断言检查字符串 "foo" 不是 timedelta64 类型
    assert not checks.is_td64("foo")
    # 断言检查当前时间戳的 numpy datetime64 对象不是 timedelta64 类型
    assert not checks.is_td64(np.datetime64("now", "s"))
def test_is_datetime64_object(install_temp):
    import checks  # 导入名为checks的模块

    assert checks.is_dt64(np.datetime64(1234, "ns"))  # 断言检查np.datetime64对象是否为datetime64类型
    assert checks.is_dt64(np.datetime64("NaT", "ns"))  # 断言检查np.datetime64对象是否为datetime64类型

    assert not checks.is_dt64(1)  # 断言检查整数1不是datetime64类型
    assert not checks.is_dt64(None)  # 断言检查None不是datetime64类型
    assert not checks.is_dt64("foo")  # 断言检查字符串"foo"不是datetime64类型
    assert not checks.is_dt64(np.timedelta64(1234))  # 断言检查np.timedelta64对象不是datetime64类型


def test_get_datetime64_value(install_temp):
    import checks  # 导入名为checks的模块

    dt64 = np.datetime64("2016-01-01", "ns")  # 创建一个datetime64对象

    result = checks.get_dt64_value(dt64)  # 调用函数获取datetime64对象的值
    expected = dt64.view("i8")  # 获取datetime64对象的i8视图

    assert result == expected  # 断言检查结果是否等于期望值


def test_get_timedelta64_value(install_temp):
    import checks  # 导入名为checks的模块

    td64 = np.timedelta64(12345, "h")  # 创建一个timedelta64对象

    result = checks.get_td64_value(td64)  # 调用函数获取timedelta64对象的值
    expected = td64.view("i8")  # 获取timedelta64对象的i8视图

    assert result == expected  # 断言检查结果是否等于期望值


def test_get_datetime64_unit(install_temp):
    import checks  # 导入名为checks的模块

    dt64 = np.datetime64("2016-01-01", "ns")  # 创建一个datetime64对象
    result = checks.get_dt64_unit(dt64)  # 调用函数获取datetime64对象的单位
    expected = 10  # 期望的单位数值
    assert result == expected  # 断言检查结果是否等于期望值

    td64 = np.timedelta64(12345, "h")  # 创建一个timedelta64对象
    result = checks.get_dt64_unit(td64)  # 调用函数获取timedelta64对象的单位
    expected = 5  # 期望的单位数值
    assert result == expected  # 断言检查结果是否等于期望值


def test_abstract_scalars(install_temp):
    import checks  # 导入名为checks的模块

    assert checks.is_integer(1)  # 断言检查整数1是否为整数类型
    assert checks.is_integer(np.int8(1))  # 断言检查np.int8(1)是否为整数类型
    assert checks.is_integer(np.uint64(1))  # 断言检查np.uint64(1)是否为整数类型


def test_default_int(install_temp):
    import checks  # 导入名为checks的模块

    assert checks.get_default_integer() is np.dtype(int)  # 断言检查默认整数类型是否为np.dtype(int)


def test_convert_datetime64_to_datetimestruct(install_temp):
    # GH#21199
    import checks  # 导入名为checks的模块

    res = checks.convert_datetime64_to_datetimestruct()  # 调用函数将datetime64转换为日期时间结构

    exp = {  # 期望的日期时间结构字典
        "year": 2022,
        "month": 3,
        "day": 15,
        "hour": 20,
        "min": 1,
        "sec": 55,
        "us": 260292,
        "ps": 0,
        "as": 0,
    }

    assert res == exp  # 断言检查结果是否等于期望值


class TestDatetimeStrings:
    def test_make_iso_8601_datetime(self, install_temp):
        # GH#21199
        import checks  # 导入名为checks的模块
        dt = datetime(2016, 6, 2, 10, 45, 19)  # 创建一个datetime对象
        result = checks.make_iso_8601_datetime(dt)  # 调用函数生成ISO 8601格式的日期时间字符串
        assert result == b"2016-06-02T10:45:19"  # 断言检查结果是否等于期望值

    def test_get_datetime_iso_8601_strlen(self, install_temp):
        # GH#21199
        import checks  # 导入名为checks的模块
        res = checks.get_datetime_iso_8601_strlen()  # 调用函数获取ISO 8601日期时间字符串的长度
        assert res == 48  # 断言检查结果是否等于期望值


@pytest.mark.parametrize(  # 使用pytest.mark.parametrize装饰器进行参数化测试
    "arrays",  # 参数名
    [  # 参数列表
        [np.random.rand(2)],  # 第一组参数，包含一个形状为(2,)的随机数组
        [np.random.rand(2), np.random.rand(3, 1)],  # 第二组参数，包含两个随机数组
        [np.random.rand(2), np.random.rand(2, 3, 2), np.random.rand(1, 3, 2)],  # 第三组参数，包含三个随机数组
        [np.random.rand(2, 1)] * 4 + [np.random.rand(1, 1, 1)],  # 第四组参数，包含四个形状为(2,1)的随机数组和一个形状为(1,1,1)的随机数组
    ]
)
def test_multiiter_fields(install_temp, arrays):
    import checks  # 导入名为checks的模块
    bcast = np.broadcast(*arrays)  # 使用np.broadcast创建广播对象

    assert bcast.ndim == checks.get_multiiter_number_of_dims(bcast)  # 断言检查广播对象的维数是否与函数返回值相等
    assert bcast.size == checks.get_multiiter_size(bcast)  # 断言检查广播对象的大小是否与函数返回值相等
    assert bcast.numiter == checks.get_multiiter_num_of_iterators(bcast)  # 断言检查广播对象的迭代器数量是否与函数返回值相等
    assert bcast.shape == checks.get_multiiter_shape(bcast)  # 断言检查广播对象的形状是否与函数返回值相等
    assert bcast.index == checks.get_multiiter_current_index(bcast)  # 断言检查广播对象的当前索引是否与函数返回值相等
    # 断言语句，验证条件是否为真
    assert all(
        [
            x.base is y.base
            # 使用 zip 函数同时迭代 bcast.iters 和 checks.get_multiiter_iters(bcast) 的元素 x, y
            for x, y in zip(bcast.iters, checks.get_multiiter_iters(bcast))
        ]
    )
def test_dtype_flags(install_temp):
    # 导入名为 checks 的模块，用于后续的函数调用
    import checks
    # 创建一个包含一些有趣标志的数据类型对象
    dtype = np.dtype("i,O")  # dtype with somewhat interesting flags
    # 断言这个数据类型对象的标志与通过函数获取的标志相同
    assert dtype.flags == checks.get_dtype_flags(dtype)


def test_conv_intp(install_temp):
    # 导入名为 checks 的模块，用于后续的函数调用
    import checks

    class myint:
        def __int__(self):
            return 3

    # 这些转换通过 `__int__` 方法进行，而不是 `__index__` 方法：
    # 断言对浮点数进行转换后得到的结果为整数 3
    assert checks.conv_intp(3.) == 3
    # 断言对自定义类 myint 的实例进行转换后得到的结果为整数 3
    assert checks.conv_intp(myint()) == 3


def test_npyiter_api(install_temp):
    # 导入名为 checks 的模块，用于后续的函数调用
    import checks
    # 创建一个形状为 (3, 2) 的随机数组
    arr = np.random.rand(3, 2)

    # 创建一个数组迭代器对象 it
    it = np.nditer(arr)
    # 断言通过函数获取的迭代器大小与 it 对象的 itersize 相同，并且等于 arr 数组元素个数
    assert checks.get_npyiter_size(it) == it.itersize == np.prod(arr.shape)
    # 断言通过函数获取的迭代器维度与 it 对象的 ndim 相同，并且等于 1
    assert checks.get_npyiter_ndim(it) == it.ndim == 1
    # 断言通过函数获取的迭代器是否具有索引与 it 对象的 has_index 属性相同，均为 False
    assert checks.npyiter_has_index(it) == it.has_index == False

    # 创建一个带有 "c_index" 标志的数组迭代器对象 it
    it = np.nditer(arr, flags=["c_index"])
    # 断言通过函数获取的迭代器是否具有索引与 it 对象的 has_index 属性相同，均为 True
    assert checks.npyiter_has_index(it) == it.has_index == True
    # 断言通过函数获取的迭代器是否具有延迟缓冲分配与 it 对象的 has_delayed_bufalloc 属性相同，均为 False
    assert (
        checks.npyiter_has_delayed_bufalloc(it)
        == it.has_delayed_bufalloc
        == False
    )

    # 创建一个带有 "buffered", "delay_bufalloc" 标志的数组迭代器对象 it
    it = np.nditer(arr, flags=["buffered", "delay_bufalloc"])
    # 断言通过函数获取的迭代器是否具有延迟缓冲分配与 it 对象的 has_delayed_bufalloc 属性相同，均为 True
    assert (
        checks.npyiter_has_delayed_bufalloc(it)
        == it.has_delayed_bufalloc
        == True
    )

    # 创建一个带有 "multi_index" 标志的数组迭代器对象 it
    it = np.nditer(arr, flags=["multi_index"])
    # 断言通过函数获取的迭代器大小与 it 对象的 itersize 相同，并且等于 arr 数组元素个数
    assert checks.get_npyiter_size(it) == it.itersize == np.prod(arr.shape)
    # 断言通过函数获取的迭代器是否具有多重索引与 it 对象的 has_multi_index 属性相同，均为 True
    assert checks.npyiter_has_multi_index(it) == it.has_multi_index == True
    # 断言通过函数获取的迭代器维度与 it 对象的 ndim 相同，并且等于 2
    assert checks.get_npyiter_ndim(it) == it.ndim == 2

    # 创建一个形状为 (2, 1, 2) 的随机数组 arr2
    arr2 = np.random.rand(2, 1, 2)
    # 创建一个同时迭代 arr 和 arr2 的数组迭代器对象 it
    it = np.nditer([arr, arr2])
    # 断言通过函数获取的迭代器操作数数目与 it 对象的 nop 相同，均为 2
    assert checks.get_npyiter_nop(it) == it.nop == 2
    # 断言通过函数获取的迭代器大小与 it 对象的 itersize 相同，并且等于 12
    assert checks.get_npyiter_size(it) == it.itersize == 12
    # 断言通过函数获取的迭代器维度与 it 对象的 ndim 相同，并且等于 3
    assert checks.get_npyiter_ndim(it) == it.ndim == 3
    # 断言通过函数获取的迭代器操作数列表中的每个元素与 it 对象的 operands 属性中的每个元素相同
    assert all(
        x is y for x, y in zip(checks.get_npyiter_operands(it), it.operands)
    )
    # 断言通过函数获取的迭代器视图列表中的每个元素与 it 对象的 itviews 属性中的每个元素在数值上相近
    assert all(
        [
            np.allclose(x, y)
            for x, y in zip(checks.get_npyiter_itviews(it), it.itviews)
        ]
    )


def test_fillwithbytes(install_temp):
    # 导入名为 checks 的模块，用于后续的函数调用
    import checks
    # 调用函数生成填充有字节的数组 arr
    arr = checks.compile_fillwithbyte()
    # 断言 arr 数组等于形状为 (1, 2) 且元素全为 1 的数组
    assert_array_equal(arr, np.ones((1, 2)))


def test_complex(install_temp):
    # 从 checks 模块中导入 inc2_cfloat_struct 函数
    from checks import inc2_cfloat_struct
    
    # 创建一个复数数组 arr，包含整数和复数
    arr = np.array([0, 10+10j], dtype="F")
    # 调用函数处理这个数组 arr
    inc2_cfloat_struct(arr)
    # 断言处理后 arr 的第二个元素等于 (12 + 12j)
    assert arr[1] == (12 + 12j)
```