# `.\numpy\numpy\_core\tests\test_casting_unittests.py`

```
"""
The tests exercise the casting machinery in a more low-level manner.
The reason is mostly to test a new implementation of the casting machinery.

Unlike most tests in NumPy, these are closer to unit-tests rather
than integration tests.
"""

# 引入pytest库，用于运行测试
import pytest
# 引入textwrap库，用于文本包装
import textwrap
# 引入enum枚举类，用于定义常量
import enum
# 引入random库，用于生成随机数
import random
# 引入ctypes库，用于与C语言兼容的数据类型转换

# 引入NumPy库，并从中导入相关模块和函数
import numpy as np
# 从NumPy的stride_tricks模块中导入as_strided函数
from numpy.lib.stride_tricks import as_strided
# 从NumPy的testing子模块中导入assert_array_equal函数，用于数组比较
from numpy.testing import assert_array_equal
# 从NumPy的_core._multiarray_umath模块中导入_get_castingimpl函数
from numpy._core._multiarray_umath import _get_castingimpl as get_castingimpl


# 定义简单数据类型列表，排除object、parametric和long double（在struct中不支持）
simple_dtypes = "?bhilqBHILQefdFD"
# 如果long和long long的字节大小不同，则移除它们
if np.dtype("l").itemsize != np.dtype("q").itemsize:
    simple_dtypes = simple_dtypes.replace("l", "").replace("L", "")
# 将字符类型转换为对应的NumPy数据类型对象列表
simple_dtypes = [type(np.dtype(c)) for c in simple_dtypes]


# 定义生成简单数据类型实例的生成器函数
def simple_dtype_instances():
    # 遍历简单数据类型列表中的每个数据类型类
    for dtype_class in simple_dtypes:
        # 创建该数据类型的实例
        dt = dtype_class()
        # 使用pytest.param封装数据类型实例，作为测试参数，并使用其字符串表示作为id
        yield pytest.param(dt, id=str(dt))
        # 如果数据类型的字节顺序不是本地字节顺序，则创建一个新的与本地字节顺序相同的实例
        if dt.byteorder != "|":
            dt = dt.newbyteorder()
            # 再次封装新的数据类型实例作为测试参数，并使用其字符串表示作为id
            yield pytest.param(dt, id=str(dt))


# 定义根据数据类型返回预期字符串长度的函数
def get_expected_stringlength(dtype):
    """Returns the string length when casting the basic dtypes to strings.
    """
    # 如果数据类型是布尔型，返回固定的长度5
    if dtype == np.bool:
        return 5
    # 如果数据类型是无符号整型或有符号整型
    if dtype.kind in "iu":
        # 根据不同的字节大小返回不同的预期长度
        if dtype.itemsize == 1:
            length = 3
        elif dtype.itemsize == 2:
            length = 5
        elif dtype.itemsize == 4:
            length = 10
        elif dtype.itemsize == 8:
            length = 20
        else:
            raise AssertionError(f"did not find expected length for {dtype}")

        # 对于有符号整型，长度需要加上一个字符用于符号位
        if dtype.kind == "i":
            length += 1

        return length

    # 对于长双精度浮点数，根据其字符类型返回固定的长度
    if dtype.char == "g":
        return 48
    elif dtype.char == "G":
        return 48 * 2
    elif dtype.kind == "f":
        return 32  # 半精度浮点数也是32
    elif dtype.kind == "c":
        return 32 * 2  # 复数类型是32的两倍长度

    raise AssertionError(f"did not find expected length for {dtype}")


# 定义枚举类Casting，用于表示类型转换的不同方式
class Casting(enum.IntEnum):
    no = 0
    equiv = 1
    safe = 2
    same_kind = 3
    unsafe = 4


# 定义获取类型转换表的函数
def _get_cancast_table():
    # 创建一个多行字符串，其中包含表格，用于定义数据类型间的类型转换关系
    table = textwrap.dedent("""
        X ? b h i l q B H I L Q e f d g F D G S U V O M m
        ? # = = = = = = = = = = = = = = = = = = = = = . =
        b . # = = = = . . . . . = = = = = = = = = = = . =
        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        B . ~ = = = = # = = = =```
    # 创建一个多行字符串，其中包含表格，用于定义数据类型间的类型转换关系
    table = textwrap.dedent("""
        X ? b h i l q B H I L Q e f d g F D G S U V O M m
        ? # = = = = = = = = = = = = = = = = = = = = = . =
        b . # = = = = . . . . . = = = = = = = = = = = . =
        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        B . ~ = = = = # = = = =```
        # 创建一个多行字符串，其中包含表格，用于定义数据类型间的类型转换关系
        table = textwrap.dedent("""
            X ? b h i l q B H I L Q e f d g F D G S U V O M m
            ? # = = = = = = = = = = = = = = = = = = = = = . =
            b . # = = = = . . . . . = = = = = = = = = = = . =
            h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
            i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
            l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
            q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
            B . ~ = = = = # = = = =
    
    
    
        # 根据字符定义转换策略的字典，将字符映射到对应的转换类型
        convert_cast = {".": Casting.unsafe, "~": Casting.same_kind,
                        "=": Casting.safe, "#": Casting.equiv,
                        " ": -1}
    
    
    
        # 创建一个空字典用于存储数据类型间的转换策略
        cancast = {}
        # 遍历数据类型列表和表格的行，将转换策略填充到字典中
        for from_dt, row in zip(dtypes, table[1:]):
            cancast[from_dt] = {}
            for to_dt, c in zip(dtypes, row[2::2]):
                cancast[from_dt][to_dt] = convert_cast[c]
    
    
    
        # 返回填充好的数据类型间转换策略字典
        return cancast
# 调用函数 _get_cancast_table() 并将结果赋给 CAST_TABLE
CAST_TABLE = _get_cancast_table()

# 定义测试类 TestChanges，用于测试行为变化
class TestChanges:
    """
    These test cases exercise some behaviour changes
    """

    # 测试函数 test_float_to_string，参数化测试浮点数转换为字符串
    @pytest.mark.parametrize("string", ["S", "U"])
    @pytest.mark.parametrize("floating", ["e", "f", "d", "g"])
    def test_float_to_string(self, floating, string):
        # 断言能够从浮点数 floating 转换为字符串 string
        assert np.can_cast(floating, string)
        # 断言能够从浮点数 floating 转换为带有后缀 "100" 的字符串 string
        # 100 is long enough to hold any formatted floating
        assert np.can_cast(floating, f"{string}100")

    # 测试函数 test_to_void，验证转换为 void 类型的安全性
    def test_to_void(self):
        # 断言能够从双精度浮点数 "d" 转换为 void 类型 "V"
        assert np.can_cast("d", "V")
        # 断言能够从长度为 20 的字符串 "S20" 转换为 void 类型 "V"

        assert np.can_cast("S20", "V")

        # 当 void 类型的长度过小时，断言转换不安全
        assert not np.can_cast("d", "V1")
        assert not np.can_cast("S20", "V1")
        assert not np.can_cast("U1", "V1")

        # 当源类型和目标类型都是结构化类型时，使用 "same_kind" 转换类型
        assert np.can_cast("d,i", "V", casting="same_kind")

        # 当源类型和目标类型都是无结构化 void 类型时，使用 "no" 表示无需转换
        assert np.can_cast("V3", "V", casting="no")
        assert np.can_cast("V0", "V", casting="no")


# 定义测试类 TestCasting，用于测试类型转换
class TestCasting:
    # 定义类变量 size，用于设置数组大小，最好大于 NPY_LOWLEVEL_BUFFER_BLOCKSIZE * itemsize

    size = 1500  # Best larger than NPY_LOWLEVEL_BUFFER_BLOCKSIZE * itemsize

    # 定义函数 get_data，根据给定的 dtype1 和 dtype2 获取数据
    def get_data(self, dtype1, dtype2):
        # 根据 dtype1 和 dtype2 的大小确定数组的长度
        if dtype2 is None or dtype1.itemsize >= dtype2.itemsize:
            length = self.size // dtype1.itemsize
        else:
            length = self.size // dtype2.itemsize

        # 使用 dtype1 创建空数组 arr1，并进行一些断言以确保数组属性
        arr1 = np.empty(length, dtype=dtype1)
        assert arr1.flags.c_contiguous
        assert arr1.flags.aligned

        # 生成随机值列表 values
        values = [random.randrange(-128, 128) for _ in range(length)]

        # 遍历 values，通过 item 赋值方式向 arr1 中填充值
        for i, value in enumerate(values):
            # 如果 value 小于 0 且 dtype1 是无符号整数类型，则手动转换成对应的正数值
            if value < 0 and dtype1.kind == "u":
                value = value + np.iinfo(dtype1).max + 1
            arr1[i] = value

        # 如果 dtype2 是 None，则返回 arr1 和 values
        if dtype2 is None:
            if dtype1.char == "?":
                values = [bool(v) for v in values]
            return arr1, values

        # 如果 dtype2 是布尔类型，则将 values 转换成布尔值列表
        if dtype2.char == "?":
            values = [bool(v) for v in values]

        # 使用 dtype2 创建空数组 arr2，并进行一些断言以确保数组属性
        arr2 = np.empty(length, dtype=dtype2)
        assert arr2.flags.c_contiguous
        assert arr2.flags.aligned

        # 再次遍历 values，通过 item 赋值方式向 arr2 中填充值
        for i, value in enumerate(values):
            # 如果 value 小于 0 且 dtype2 是无符号整数类型，则手动转换成对应的正数值
            if value < 0 and dtype2.kind == "u":
                value = value + np.iinfo(dtype2).max + 1
            arr2[i] = value

        # 返回 arr1, arr2 和 values
        return arr1, arr2, values
    # 定义一个方法，用于生成经过变异处理的数据数组
    def get_data_variation(self, arr1, arr2, aligned=True, contig=True):
        """
        Returns a copy of arr1 that may be non-contiguous or unaligned, and a
        matching array for arr2 (although not a copy).
        """
        # 如果需要保证连续性，计算 arr1 和 arr2 的字节步长
        if contig:
            stride1 = arr1.dtype.itemsize
            stride2 = arr2.dtype.itemsize
        # 如果需要保证对齐性，计算 arr1 和 arr2 的字节步长
        elif aligned:
            stride1 = 2 * arr1.dtype.itemsize
            stride2 = 2 * arr2.dtype.itemsize
        # 否则，计算 arr1 和 arr2 的字节步长
        else:
            stride1 = arr1.dtype.itemsize + 1
            stride2 = arr2.dtype.itemsize + 1

        # 计算允许的最大字节数，用来分配新数组
        max_size1 = len(arr1) * 3 * arr1.dtype.itemsize + 1
        max_size2 = len(arr2) * 3 * arr2.dtype.itemsize + 1
        # 创建用于存储数据的新数组，类型为无符号 8 位整数
        from_bytes = np.zeros(max_size1, dtype=np.uint8)
        to_bytes = np.zeros(max_size2, dtype=np.uint8)

        # 对上述分配是否足够进行断言检查
        assert stride1 * len(arr1) <= from_bytes.nbytes
        assert stride2 * len(arr2) <= to_bytes.nbytes

        # 如果需要对齐，使用 as_strided 函数生成新的 arr1 和 arr2
        if aligned:
            new1 = as_strided(from_bytes[:-1].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[:-1].view(arr2.dtype),
                              arr2.shape, (stride2,))
        # 否则，稍微偏移后使用 as_strided 函数生成新的 arr1 和 arr2
        else:
            new1 = as_strided(from_bytes[1:].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[1:].view(arr2.dtype),
                              arr2.shape, (stride2,))

        # 将 arr1 的数据复制到新生成的 new1 数组中
        new1[...] = arr1

        # 如果不需要保证连续性，进行进一步的检查确保没有写入不应该写入的字节
        if not contig:
            offset = arr1.dtype.itemsize if aligned else 0
            buf = from_bytes[offset::stride1].tobytes()
            assert buf.count(b"\0") == len(buf)

        # 根据条件检查 new1 和 new2 的连续性
        if contig:
            assert new1.flags.c_contiguous
            assert new2.flags.c_contiguous
        else:
            assert not new1.flags.c_contiguous
            assert not new2.flags.c_contiguous

        # 根据条件检查 new1 和 new2 的对齐性
        if aligned:
            assert new1.flags.aligned
            assert new2.flags.aligned
        else:
            assert not new1.flags.aligned or new1.dtype.alignment == 1
            assert not new2.flags.aligned or new2.dtype.alignment == 1

        # 返回生成的变异后的数组 new1 和 new2
        return new1, new2

    # 使用 pytest 的参数化功能，针对 simple_dtypes 参数化测试用例
    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    # 定义测试函数，用于测试从指定类型到简单数据类型的类型转换
    def test_simple_cancast(self, from_Dt):
        # 遍历简单数据类型列表，将from_Dt转换为to_Dt并进行测试
        for to_Dt in simple_dtypes:
            # 获取类型转换的实现函数
            cast = get_castingimpl(from_Dt, to_Dt)

            # 遍历from_Dt及其字节序新顺序的实例
            for from_dt in [from_Dt(), from_Dt().newbyteorder()]:
                # 解析描述符，获取默认值和转换结果
                default = cast._resolve_descriptors((from_dt, None))[1][1]
                assert default == to_Dt()
                del default

                # 遍历to_Dt及其字节序新顺序的实例
                for to_dt in [to_Dt(), to_Dt().newbyteorder()]:
                    # 解析描述符，获取类型转换、转换结果、视图偏移
                    casting, (from_res, to_res), view_off = (
                            cast._resolve_descriptors((from_dt, to_dt)))
                    assert(type(from_res) == from_Dt)
                    assert(type(to_res) == to_Dt)

                    # 如果视图偏移不为空，表示视图可接受，需无需进行强制转换
                    # 并且字节顺序必须匹配
                    if view_off is not None:
                        assert casting == Casting.no
                        assert Casting.equiv == CAST_TABLE[from_Dt][to_Dt]
                        assert from_res.isnative == to_res.isnative
                    else:
                        # 如果from_Dt与to_Dt相同，则需要确认to_res不同于from_dt
                        if from_Dt == to_Dt:
                            assert from_res.isnative != to_res.isnative
                        # 否则，根据CAST_TABLE检查强制转换类型
                        assert casting == CAST_TABLE[from_Dt][to_Dt]

                    # 如果from_Dt与to_Dt相同，则from_dt应等于from_res，to_dt应等于to_res
                    if from_Dt is to_Dt:
                        assert(from_dt is from_res)
                        assert(to_dt is to_res)

    # 使用pytest标记过滤掉特定的警告信息
    @pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
    # 使用pytest.mark.parametrize装饰器，参数化测试函数的输入参数from_dt
    @pytest.mark.parametrize("from_dt", simple_dtype_instances())
    # 定义名为 test_simple_direct_casts 的测试方法，用于测试直接类型转换
    def test_simple_direct_casts(self, from_dt):
        """
        This test checks numeric direct casts for dtypes supported also by the
        struct module (plus complex).  It tries to be test a wide range of
        inputs, but skips over possibly undefined behaviour (e.g. int rollover).
        Longdouble and CLongdouble are tested, but only using double precision.

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        
        # 遍历 simple_dtype_instances() 返回的数据类型实例
        for to_dt in simple_dtype_instances():
            # 取类型实例的第一个值
            to_dt = to_dt.values[0]
            
            # 获取从 from_dt 到 to_dt 的类型转换方法
            cast = get_castingimpl(type(from_dt), type(to_dt))
            
            # 解析类型转换器的描述符，获取转换方法、原始类型结果、目标类型结果及视图偏移量
            casting, (from_res, to_res), view_off = cast._resolve_descriptors((from_dt, to_dt))
            
            # 如果 from_res 或 to_res 不等于 from_dt 或 to_dt，则不进行测试，因为已在多个步骤中测试过
            if from_res is not from_dt or to_res is not to_dt:
                # Do not test this case, it is handled in multiple steps,
                # each of which should is tested individually.
                return
            
            # 判断是否安全转换
            safe = casting <= Casting.safe
            
            # 删除变量以释放内存
            del from_res, to_res, casting
            
            # 获取从 from_dt 到 to_dt 的数据
            arr1, arr2, values = self.get_data(from_dt, to_dt)
            
            # 使用 cast 对象进行简单分块调用
            cast._simple_strided_call((arr1, arr2))
            
            # 使用 Python 列表检查结果是否与预期值 values 相同
            assert arr2.tolist() == values
            
            # 使用分块循环检查是否达到相同的结果
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))
            
            # 使用 assert_array_equal 检查 arr2_o 是否等于 arr2
            assert_array_equal(arr2_o, arr2)
            # 检查 arr2_o 的字节表示是否等于 arr2 的字节表示
            assert arr2_o.tobytes() == arr2.tobytes()
            
            # 如果支持不对齐访问并且对齐会影响结果，则进一步检查
            if ((from_dt.alignment == 1 and to_dt.alignment == 1) or
                    not cast._supports_unaligned):
                return
            
            # 检查不同的数据变化情况下是否依然保持一致性
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, True)
            cast._simple_strided_call((arr1_o, arr2_o))
            
            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()
            
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, False)
            cast._simple_strided_call((arr1_o, arr2_o))
            
            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()
            
            # 释放内存，删除变量
            del arr1_o, arr2_o, cast
    
    # 使用 pytest 的参数化标记，指定测试数据来源为 simple_dtypes
    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    # 定义一个测试方法，用于测试从日期时间类型到不同时间类型的转换
    def test_numeric_to_times(self, from_Dt):
        # 当前只实现连续循环，因此只需要测试这些情况。
        # 实例化一个从 from_Dt 获取的日期时间对象
        from_dt = from_Dt()

        # 定义不同时间数据类型的列表
        time_dtypes = [np.dtype("M8"), np.dtype("M8[ms]"), np.dtype("M8[4D]"),
                       np.dtype("m8"), np.dtype("m8[ms]"), np.dtype("m8[4D]")]
        
        # 遍历不同的时间数据类型
        for time_dt in time_dtypes:
            # 获取从 from_dt 到 time_dt 的类型转换实现
            cast = get_castingimpl(type(from_dt), type(time_dt))

            # 解析类型转换的描述符，确定转换、起始和目标结果、视图偏移
            casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, time_dt))

            # 断言起始结果为 from_dt
            assert from_res is from_dt
            # 断言目标结果为 time_dt
            assert to_res is time_dt
            # 清理中间变量
            del from_res, to_res

            # 断言转换在转换表中有效
            assert casting & CAST_TABLE[from_Dt][type(time_dt)]
            # 视图偏移应为 None
            assert view_off is None

            # 定义一个 int64 数据类型
            int64_dt = np.dtype(np.int64)
            # 获取数据数组 arr1, arr2 以及其值
            arr1, arr2, values = self.get_data(from_dt, int64_dt)
            # 将 arr2 视图转换为 time_dt 类型
            arr2 = arr2.view(time_dt)
            # 将 arr2 中的所有值设置为 NaT
            arr2[...] = np.datetime64("NaT")

            # 如果时间数据类型为 np.dtype("M8")
            if time_dt == np.dtype("M8"):
                # 确保至少有一个值不是 NaT
                arr1[-1] = 0  # ensure at least one value is not NaT

                # 进行简单的类型转换调用，预期会引发 ValueError 异常
                cast._simple_strided_call((arr1, arr2))
                with pytest.raises(ValueError):
                    str(arr2[-1])  # e.g. conversion to string fails
                # 结束当前测试
                return

            # 进行简单的类型转换调用
            cast._simple_strided_call((arr1, arr2))

            # 断言 arr2 的值列表应与预期的 values 相等
            assert [int(v) for v in arr2.tolist()] == values

            # 检查在步进循环中得到相同的结果
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            # 断言 arr2_o 应与 arr2 相等
            assert_array_equal(arr2_o, arr2)
            # 断言 arr2_o 的字节表示应与 arr2 的字节表示相同
            assert arr2_o.tobytes() == arr2.tobytes()
    # 使用pytest的参数化装饰器，为测试用例传入不同的参数组合
    @pytest.mark.parametrize(
            ["from_dt", "to_dt", "expected_casting", "expected_view_off",
             "nom", "denom"],
            # 设置参数组合列表
            [("M8[ns]", None, Casting.no, 0, 1, 1),  # 设置参数组合
             (str(np.dtype("M8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),  # 设置参数组合
             ("M8", "M8[ms]", Casting.safe, 0, 1, 1),  # 设置参数组合
             # should be invalid cast:
             ("M8[ms]", "M8", Casting.unsafe, None, 1, 1),  # 设置参数组合
             ("M8[5ms]", "M8[5ms]", Casting.no, 0, 1, 1),  # 设置参数组合
             ("M8[ns]", "M8[ms]", Casting.same_kind, None, 1, 10**6),  # 设置参数组合
             ("M8[ms]", "M8[ns]", Casting.safe, None, 10**6, 1),  # 设置参数组合
             ("M8[ms]", "M8[7ms]", Casting.same_kind, None, 1, 7),  # 设置参数组合
             ("M8[4D]", "M8[1M]", Casting.same_kind, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, -1, 1314, -1315, 564442610]),  # 设置参数组合
             ("m8[ns]", None, Casting.no, 0, 1, 1),  # 设置参数组合
             (str(np.dtype("m8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),  # 设置参数组合
             ("m8", "m8[ms]", Casting.safe, 0, 1, 1),  # 设置参数组合
             # should be invalid cast:
             ("m8[ms]", "m8", Casting.unsafe, None, 1, 1),  # 设置参数组合
             ("m8[5ms]", "m8[5ms]", Casting.no, 0, 1, 1),  # 设置参数组合
             ("m8[ns]", "m8[ms]", Casting.same_kind, None, 1, 10**6),  # 设置参数组合
             ("m8[ms]", "m8[ns]", Casting.safe, None, 10**6, 1),  # 设置参数组合
             ("m8[ms]", "m8[7ms]", Casting.same_kind, None, 1, 7),  # 设置参数组合
             ("m8[4D]", "m8[1M]", Casting.unsafe, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, 0, 1314, -1315, 564442610])])  # 设置参数组合
    # 定义测试函数，用于测试时间类型转换功能
    def test_time_to_time(self, from_dt, to_dt,
                          expected_casting, expected_view_off,
                          nom, denom):
        # 将输入参数 from_dt 转换为 numpy 的数据类型对象
        from_dt = np.dtype(from_dt)
        # 如果 to_dt 不为 None，则将其转换为 numpy 的数据类型对象
        if to_dt is not None:
            to_dt = np.dtype(to_dt)

        # 测试几个数值以进行类型转换（使用 NumPy 1.19 生成结果）
        values = np.array([-2**63, 1, 2**63-1, 10000, -10000, 2**32])
        # 将数值数组转换为指定字节序的 int64 类型
        values = values.astype(np.dtype("int64").newbyteorder(from_dt.byteorder))
        # 断言数值数组的字节序与 from_dt 的字节序相同
        assert values.dtype.byteorder == from_dt.byteorder
        # 断言第一个值为 NaT（Not a Time）
        assert np.isnat(values.view(from_dt)[0])

        # 获取 from_dt 类型的类对象
        DType = type(from_dt)
        # 获取类型转换函数的实现
        cast = get_castingimpl(DType, DType)
        # 解析类型转换的描述符
        casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, to_dt))
        # 断言解析后的 from_res 与 from_dt 相同
        assert from_res is from_dt
        # 断言解析后的 to_res 与 to_dt 相同或为 None
        assert to_res is to_dt or to_dt is None
        # 断言解析后的类型转换方式与预期相同
        assert casting == expected_casting
        # 断言解析后的视图偏移量与预期相同
        assert view_off == expected_view_off

        # 如果 nom 不为 None
        if nom is not None:
            # 计算预期输出，将结果视图转换为 to_res 类型
            expected_out = (values * nom // denom).view(to_res)
            # 将第一个元素设为 "NaT"
            expected_out[0] = "NaT"
        else:
            # 创建一个与 values 相同形状的空数组，并赋值为 denom
            expected_out = np.empty_like(values)
            expected_out[...] = denom
            # 将数组视图转换为 to_dt 类型
            expected_out = expected_out.view(to_dt)

        # 将 values 数组视图转换为 from_dt 类型
        orig_arr = values.view(from_dt)
        # 创建一个与 expected_out 相同形状的空数组
        orig_out = np.empty_like(expected_out)

        # 如果 casting 为 Casting.unsafe 并且 to_dt 为 "m8" 或 "M8"
        if casting == Casting.unsafe and (to_dt == "m8" or to_dt == "M8"):
            # 如果从非通用单位到通用单位的类型转换应报告为无效转换
            with pytest.raises(ValueError):
                cast._simple_strided_call((orig_arr, orig_out))
            return

        # 遍历 aligned 和 contig 的组合进行测试
        for aligned in [True, True]:
            for contig in [True, True]:
                # 调用 get_data_variation 获取变化后的数据数组和输出数组
                arr, out = self.get_data_variation(
                        orig_arr, orig_out, aligned, contig)
                # 将输出数组清零
                out[...] = 0
                # 调用类型转换函数进行简单的分步调用
                cast._simple_strided_call((arr, out))
                # 断言输出数组视图与预期输出视图相等
                assert_array_equal(out.view("int64"), expected_out.view("int64"))

    # 根据输入的 dtype 和修改长度，返回一个新的 dtype 对象
    def string_with_modified_length(self, dtype, change_length):
        # 如果 dtype 的字符为 "S"，则 fact 为 1，否则为 4
        fact = 1 if dtype.char == "S" else 4
        # 计算新的长度
        length = dtype.itemsize // fact + change_length
        # 构造并返回一个新的 dtype 对象
        return np.dtype(f"{dtype.byteorder}{dtype.char}{length}")

    # 使用参数化测试标记，对 simple_dtypes 和 string_char 进行参数化测试
    @pytest.mark.parametrize("other_DT", simple_dtypes)
    @pytest.mark.parametrize("string_char", ["S", "U"])
    # 测试字符串是否可转换的方法，使用给定的参数进行测试
    def test_string_cancast(self, other_DT, string_char):
        # 如果字符串类型是"S"，则设定因子为1，否则设定为4
        fact = 1 if string_char == "S" else 4

        # 获取字符串字符对应的 NumPy 数据类型
        string_DT = type(np.dtype(string_char))
        
        # 调用函数获取类型转换的实现
        cast = get_castingimpl(other_DT, string_DT)

        # 创建其他数据类型的实例
        other_dt = other_DT()
        
        # 获取预期的字符串长度
        expected_length = get_expected_stringlength(other_dt)
        
        # 构造 NumPy 数据类型，以给定字符和预期长度
        string_dt = np.dtype(f"{string_char}{expected_length}")

        # 解析描述符，获取安全性、返回的数据类型以及视图偏移量
        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert safety == Casting.safe  # 我们认为字符串转换是“安全的”
        assert view_off is None
        assert isinstance(res_dt, string_DT)

        # 对于不同长度的字符串，检查类型转换的安全性
        for change_length in [-1, 0, 1]:
            if change_length >= 0:
                expected_safety = Casting.safe
            else:
                expected_safety = Casting.same_kind

            # 获取修改后的字符串数据类型
            to_dt = self.string_with_modified_length(string_dt, change_length)
            
            # 解析描述符，获取安全性、返回的数据类型以及视图偏移量
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))
            assert res_dt is to_dt
            assert safety == expected_safety
            assert view_off is None

        # 反向转换总是被认为是不安全的：
        cast = get_castingimpl(string_DT, other_DT)

        # 解析描述符，获取安全性以及视图偏移量
        safety, _, view_off = cast._resolve_descriptors((string_dt, other_dt))
        assert safety == Casting.unsafe
        assert view_off is None

        # 再次解析描述符，获取安全性、返回的数据类型以及视图偏移量
        cast = get_castingimpl(string_DT, other_DT)
        safety, (_, res_dt), view_off = cast._resolve_descriptors(
            (string_dt, None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert other_dt is res_dt  # 返回简单数据类型的单例对象
    # 定义一个测试方法，用于验证字符串和其他数据类型之间的转换是否满足往返属性
    def test_simple_string_casts_roundtrip(self, other_dt, string_char):
        """
        Tests casts from and to string by checking the roundtripping property.

        The test also covers some string to string casts (but not all).

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        # 获取 other_dt 对应的 dtype 类型
        string_DT = type(np.dtype(string_char))

        # 获取类型转换的实现函数
        cast = get_castingimpl(type(other_dt), string_DT)
        cast_back = get_castingimpl(string_DT, type(other_dt))

        # 解析描述符，获取返回的 other_dt 和 string_dt 类型
        _, (res_other_dt, string_dt), _ = cast._resolve_descriptors(
                (other_dt, None))

        # 如果 res_other_dt 不等于 other_dt，表示不支持非本机字节顺序，跳过测试
        if res_other_dt is not other_dt:
            assert other_dt.byteorder != res_other_dt.byteorder
            return

        # 获取原始数组和值
        orig_arr, values = self.get_data(other_dt, None)

        # 创建一个与 orig_arr 等长的零填充数组，用于字符串类型 str_arr
        str_arr = np.zeros(len(orig_arr), dtype=string_dt)

        # 创建修改长度后的字符串类型数组，长度为 -1
        string_dt_short = self.string_with_modified_length(string_dt, -1)
        str_arr_short = np.zeros(len(orig_arr), dtype=string_dt_short)

        # 创建修改长度后的字符串类型数组，长度为 1
        string_dt_long = self.string_with_modified_length(string_dt, 1)
        str_arr_long = np.zeros(len(orig_arr), dtype=string_dt_long)

        # 断言不支持非对齐的简单赋值操作
        assert not cast._supports_unaligned
        assert not cast_back._supports_unaligned

        # 循环检查连续和非连续数组的数据变化
        for contig in [True, False]:
            other_arr, str_arr = self.get_data_variation(
                orig_arr, str_arr, True, contig)
            _, str_arr_short = self.get_data_variation(
                orig_arr, str_arr_short.copy(), True, contig)
            _, str_arr_long = self.get_data_variation(
                orig_arr, str_arr_long, True, contig)

            # 对简单的步进调用进行类型转换
            cast._simple_strided_call((other_arr, str_arr))

            # 对短数组进行步进调用，并断言转换后的结果
            cast._simple_strided_call((other_arr, str_arr_short))
            assert_array_equal(str_arr.astype(string_dt_short), str_arr_short)

            # 对长数组进行步进调用，并断言转换后的结果
            cast._simple_strided_call((other_arr, str_arr_long))
            assert_array_equal(str_arr, str_arr_long)

            # 如果 other_dt 的类型是布尔类型，则跳过循环
            if other_dt.kind == "b":
                continue

            # 将 other_arr 数组的所有元素设为 0
            other_arr[...] = 0

            # 对字符串数组 str_arr 和 other_arr 进行反向转换
            cast_back._simple_strided_call((str_arr, other_arr))

            # 断言 orig_arr 和 other_arr 相等
            assert_array_equal(orig_arr, other_arr)

            # 将 other_arr 数组的所有元素设为 0
            other_arr[...] = 0

            # 对长字符串数组 str_arr_long 和 other_arr 进行反向转换
            cast_back._simple_strided_call((str_arr_long, other_arr))

            # 断言 orig_arr 和 other_arr 相等
            assert_array_equal(orig_arr, other_arr)
    # 测试字符串到字符串类型的强制转换是否可行
    def test_string_to_string_cancast(self, other_dt, string_char):
        # 将 other_dt 转换为 NumPy 的数据类型对象
        other_dt = np.dtype(other_dt)

        # 根据 string_char 确定 factor 值，用于计算预期长度
        fact = 1 if string_char == "S" else 4
        # 根据 other_dt 的字符类型确定 div 值，用于计算预期长度
        div = 1 if other_dt.char == "S" else 4

        # 确定 string_DT 为与 string_char 相对应的 NumPy 数据类型对象
        string_DT = type(np.dtype(string_char))
        # 获取从 other_dt 到 string_DT 的转换实现对象
        cast = get_castingimpl(type(other_dt), string_DT)

        # 计算预期的字符串长度
        expected_length = other_dt.itemsize // div
        # 创建一个新的字符串数据类型对象，长度为预期长度
        string_dt = np.dtype(f"{string_char}{expected_length}")

        # 调用转换实现对象的 _resolve_descriptors 方法，解析描述符以获取安全性、结果数据类型和视图偏移量
        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        # 断言结果数据类型的字节大小符合预期长度乘以 factor
        assert res_dt.itemsize == expected_length * fact
        # 断言结果数据类型是预期的字符串数据类型对象
        assert isinstance(res_dt, string_DT)

        # 根据 other_dt 和 string_char 的值确定预期的安全性和视图偏移量
        expected_view_off = None
        if other_dt.char == string_char:
            if other_dt.isnative:
                expected_safety = Casting.no
                expected_view_off = 0
            else:
                expected_safety = Casting.equiv
        elif string_char == "U":
            expected_safety = Casting.safe
        else:
            expected_safety = Casting.unsafe

        # 断言实际的视图偏移量与预期的视图偏移量一致
        assert view_off == expected_view_off
        # 断言实际的安全性与预期的安全性一致
        assert expected_safety == safety

        # 遍历修改长度为[-1, 0, 1]的情况
        for change_length in [-1, 0, 1]:
            # 调用 self.string_with_modified_length 方法，修改 string_dt 的长度为 to_dt
            to_dt = self.string_with_modified_length(string_dt, change_length)
            # 再次调用转换实现对象的 _resolve_descriptors 方法，解析描述符以获取安全性、结果数据类型和视图偏移量
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))

            # 断言结果数据类型为预期的 to_dt
            assert res_dt is to_dt
            # 根据不同的 change_length 断言视图偏移量是否符合预期
            if change_length <= 0:
                assert view_off == expected_view_off
            else:
                assert view_off is None
            # 根据预期的安全性断言实际的安全性
            if expected_safety == Casting.unsafe:
                assert safety == expected_safety
            elif change_length < 0:
                assert safety == Casting.same_kind
            elif change_length == 0:
                assert safety == expected_safety
            elif change_length > 0:
                assert safety == Casting.safe

    @pytest.mark.parametrize("order1", [">", "<"])
    @pytest.mark.parametrize("order2", [">", "<"])
    def test_unicode_byteswapped_cast(self, order1, order2):
        # 非常具体的测试，用于测试 Unicode 的字节交换，包括非对齐数组数据。
        # 创建两种不同字节顺序的数据类型对象 dtype1 和 dtype2
        dtype1 = np.dtype(f"{order1}U30")
        dtype2 = np.dtype(f"{order2}U30")
        # 创建两个未对齐的数组 data1 和 data2，将其视图设置为 dtype1 和 dtype2
        data1 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype1)
        data2 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype2)
        # 如果 dtype1 的对齐方式不为 1，则进行断言检查
        if dtype1.alignment != 1:
            assert not data1.flags.aligned
            assert not data2.flags.aligned

        # 创建一个 Unicode 元素
        element = "this is a ünicode string‽"
        data1[()] = element
        # 测试 data1 和 data1.copy()（应该是对齐的）两种情况
        for data in [data1, data1.copy()]:
            # 将 data1 的值复制给 data2，并断言它们的值相等
            data2[...] = data1
            assert data2[()] == element
            assert data2.copy()[()] == element
    # 测试空类型到字符串的特殊情况转换
    def test_void_to_string_special_case(self):
        # 测试空类型到字符串的特殊情况转换，这种情况可能可以转换为错误（与下面的 `test_object_to_parametric_internal_error` 进行比较）。
        assert np.array([], dtype="V5").astype("S").dtype.itemsize == 5
        assert np.array([], dtype="V5").astype("U").dtype.itemsize == 4 * 5

    # 测试从对象到参数化类型的内部错误处理
    def test_object_to_parametric_internal_error(self):
        # 拒绝从对象到参数化类型的转换，需要先确定正确的实例。
        object_dtype = type(np.dtype(object))
        other_dtype = type(np.dtype(str))
        cast = get_castingimpl(object_dtype, other_dtype)
        with pytest.raises(TypeError,
                    match="casting from object to the parametric DType"):
            cast._resolve_descriptors((np.dtype("O"), None))

    # 使用简单的数据类型实例参数化测试对象和简单解析
    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_object_and_simple_resolution(self, dtype):
        # 简单的测试，用于测试当没有指定实例时的转换
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(object_dtype, type(dtype))

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), dtype))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt is dtype

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt == dtype.newbyteorder("=")

    # 使用简单的数据类型实例参数化测试简单到对象的解析
    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_simple_to_object_resolution(self, dtype):
        # 简单的测试，用于测试当没有指定实例时从简单类型到对象的转换
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(type(dtype), object_dtype)

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (dtype, None))
        assert safety == Casting.safe
        assert view_off is None
        assert res_dt is np.dtype("O")

    # 使用"no"或"unsafe"参数化测试空类型和带有子数组的结构化数组
    @pytest.mark.parametrize("casting", ["no", "unsafe"])
    def test_void_and_structured_with_subarray(self, casting):
        # 对应于gh-19325的测试案例
        dtype = np.dtype([("foo", "<f4", (3, 2))])
        expected = casting == "unsafe"
        assert np.can_cast("V4", dtype, casting=casting) == expected
        assert np.can_cast(dtype, "V4", casting=casting) == expected
    @pytest.mark.parametrize(["to_dt", "expected_off"],
            [  # Same as `from_dt` but with both fields shifted:
             (np.dtype({"names": ["a", "b"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Additional change of the names
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Incompatible field offset change
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 6]}), None)])
    def test_structured_field_offsets(self, to_dt, expected_off):
        # This checks the cast-safety and view offset for swapped and "shifted"
        # fields which are viewable
        
        # Define the original structured data type with field names "a" and "b",
        # integer and float formats, and specified offsets.
        from_dt = np.dtype({"names": ["a", "b"],
                            "formats": ["i4", "f4"],
                            "offsets": [2, 6]})
        
        # Obtain the casting implementation for converting from `from_dt` to `to_dt`.
        cast = get_castingimpl(type(from_dt), type(to_dt))
        
        # Resolve the casting descriptors and retrieve safety, ignored flags, and view offset.
        safety, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        
        # Assert the safety of the cast operation based on the equality of field names.
        if from_dt.names == to_dt.names:
            assert safety == Casting.equiv
        else:
            assert safety == Casting.safe
        
        # Assert the expected view offset after shifting the original data pointer by -2 bytes.
        # This ensures alignment by effectively adding 2 bytes of spacing before `from_dt`.
        assert view_off == expected_off
    @pytest.mark.parametrize(("from_dt", "to_dt", "expected_off"), [
        # 使用 pytest 的参数化标记，定义测试参数和预期结果
        # Subarray cases:
        ("i", "(1,1)i", 0),  # 子数组情况，预期偏移量为0
        ("(1,1)i", "i", 0),  # 子数组情况，预期偏移量为0
        ("(2,1)i", "(2,1)i", 0),  # 子数组情况，预期偏移量为0
        # field cases (field to field is tested explicitly also):
        # 考虑字段到字段的情况（字段间也会被显式测试）：
        # 由于负偏移可能导致结构化 dtype 间接访问无效内存，因此不视为可查看
        ("i", dict(names=["a"], formats=["i"], offsets=[2]), None),
        (dict(names=["a"], formats=["i"], offsets=[2]), "i", 2),  # 字段到字段情况，预期偏移量为2
        # 当前不视为可查看，因为存在多个字段，即使它们重叠（也许我们不应该允许这种情况？）
        ("i", dict(names=["a", "b"], formats=["i", "i"], offsets=[2, 2]), None),
        # 不同数量的字段无法工作，应该直接失败，因此从不报告为“可查看”：
        ("i,i", "i,i,i", None),
        # Unstructured void cases:
        ("i4", "V3", 0),  # void较小或相等，预期偏移量为0
        ("i4", "V4", 0),  # void较小或相等，预期偏移量为0
        ("i4", "V10", None),  # void较大（无法查看）
        ("O", "V4", None),  # 当前拒绝对象用于视图
        ("O", "V8", None),  # 当前拒绝对象用于视图
        ("V4", "V3", 0),  # void较小或相等，预期偏移量为0
        ("V4", "V4", 0),  # void较小或相等，预期偏移量为0
        ("V3", "V4", None),  # void较大（无法查看）
        # 注意，当前的void到其他类型的转换通过字节字符串进行，并不是基于“视图”的转换方式，与反向方向不同：
        ("V4", "i4", None),  # 完全无效/不可能的转换
        ("i,i", "i,i,i", None),  # 完全无效/不可能的转换
    ])
    def test_structured_view_offsets_paramteric(
            self, from_dt, to_dt, expected_off):
        # TODO: 虽然这个测试相当彻底，但现在它并没有真正测试一些可能具有非零偏移量的路径（它们实际上不存在）。
        # 使用 pytest 的参数化测试方法，测试结构化视图的偏移量参数化情况
        from_dt = np.dtype(from_dt)
        to_dt = np.dtype(to_dt)
        # 获取类型转换的实现
        cast = get_castingimpl(type(from_dt), type(to_dt))
        # 解析描述符以获取类型转换的偏移量
        _, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        # 断言视图偏移量是否符合预期
        assert view_off == expected_off

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    # 测试对象类型转换中 NULL 和 None 的等效性
    def test_object_casts_NULL_None_equivalence(self, dtype):
        # None 转换为其他类型可能成功或失败，但 NULL 化的数组必须与填充了 None 的数组行为相同。
        arr_normal = np.array([None] * 5)  # 创建一个包含 None 的数组
        arr_NULLs = np.empty_like(arr_normal)  # 创建一个形状与 arr_normal 相同的空数组
        ctypes.memset(arr_NULLs.ctypes.data, 0, arr_NULLs.nbytes)  # 使用 ctypes 将 arr_NULLs 的内存数据置为 0
        # 检查是否满足条件，如果失败（也许应该），测试就失去了其目的：
        assert arr_NULLs.tobytes() == b"\x00" * arr_NULLs.nbytes

        try:
            expected = arr_normal.astype(dtype)  # 尝试将 arr_normal 转换为指定的 dtype
        except TypeError:
            with pytest.raises(TypeError):
                arr_NULLs.astype(dtype),  # 如果出现 TypeError，预期会抛出异常
        else:
            assert_array_equal(expected, arr_NULLs.astype(dtype))  # 断言转换后的 arr_NULLs 与预期的数组相等

    @pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + np.typecodes["AllFloat"])
    # 测试非标准布尔类型到其他类型的转换
    def test_nonstandard_bool_to_other(self, dtype):
        # 简单测试将 bool_ 类型转换为数值类型，不应暴露 NumPy 布尔值有时可以取除 0 和 1 之外的值的细节。参见也 gh-19514。
        nonstandard_bools = np.array([0, 3, -7], dtype=np.int8).view(bool)  # 创建一个非标准布尔数组
        res = nonstandard_bools.astype(dtype)  # 将非标准布尔数组转换为指定的 dtype
        expected = [0, 1, 1]
        assert_array_equal(res, expected)  # 断言转换后的结果与预期结果相等
```