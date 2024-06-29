# `.\numpy\numpy\_core\tests\test_stringdtype.py`

```
# 导入需要的模块
import concurrent.futures  # 提供并发执行的工具
import itertools  # 提供高效的迭代器工具
import os  # 提供与操作系统交互的功能
import pickle  # 提供对象序列化和反序列化的功能
import string  # 提供处理字符串的常用函数
import sys  # 提供与 Python 解释器交互的功能
import tempfile  # 提供临时文件和目录的创建功能

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 Pytest 测试框架

from numpy.dtypes import StringDType  # 导入 NumPy 的字符串数据类型
from numpy._core.tests._natype import pd_NA  # 导入 Pandas 的 NA 数据类型
from numpy.testing import assert_array_equal, IS_WASM  # 导入 NumPy 的测试工具


@pytest.fixture
def string_list():
    return ["abc", "def", "ghi" * 10, "A¢☃€ 😊" * 100, "Abc" * 1000, "DEF"]
    # 返回一个包含不同长度字符串的列表


@pytest.fixture
def random_string_list():
    chars = list(string.ascii_letters + string.digits)
    chars = np.array(chars, dtype="U1")
    ret = np.random.choice(chars, size=100 * 10, replace=True)
    return ret.view("U100")
    # 返回一个包含1000个长度为100的随机字符串的数组


@pytest.fixture(params=[True, False])
def coerce(request):
    return request.param
    # 返回一个布尔值参数，用于测试类型转换功能


@pytest.fixture(
    params=["unset", None, pd_NA, np.nan, float("nan"), "__nan__"],
    ids=["unset", "None", "pandas.NA", "np.nan", "float('nan')", "string nan"],
)
def na_object(request):
    return request.param
    # 返回一个 NA 对象参数，用于测试缺失值的不同表示形式


def get_dtype(na_object, coerce=True):
    # 根据给定的 NA 对象和类型转换标志创建 StringDType 对象
    # 对于 pd_NA，显式检查，因为与 pd_NA 不等于的结果仍是 pd_NA
    if na_object is pd_NA or na_object != "unset":
        return StringDType(na_object=na_object, coerce=coerce)
    else:
        return StringDType(coerce=coerce)


@pytest.fixture()
def dtype(na_object, coerce):
    return get_dtype(na_object, coerce)
    # 返回一个 StringDType 对象，用于测试不同的 NA 对象和类型转换标志


# 为了进行类型转换测试，创建第二份 dtype 复制，执行 dtypes 的笛卡尔积
@pytest.fixture(params=[True, False])
def coerce2(request):
    return request.param
    # 返回一个布尔值参数，用于测试类型转换功能


@pytest.fixture(
    params=["unset", None, pd_NA, np.nan, float("nan"), "__nan__"],
    ids=["unset", "None", "pandas.NA", "np.nan", "float('nan')", "string nan"],
)
def na_object2(request):
    return request.param
    # 返回一个 NA 对象参数，用于测试缺失值的不同表示形式


@pytest.fixture()
def dtype2(na_object2, coerce2):
    # 对于 pd_NA，显式检查，因为与 pd_NA 不等于的结果仍是 pd_NA
    if na_object2 is pd_NA or na_object2 != "unset":
        return StringDType(na_object=na_object2, coerce=coerce2)
    else:
        return StringDType(coerce=coerce2)
    # 返回一个 StringDType 对象，用于测试不同的 NA 对象和类型转换标志


def test_dtype_creation():
    hashes = set()
    dt = StringDType()
    assert not hasattr(dt, "na_object") and dt.coerce is True
    hashes.add(hash(dt))

    dt = StringDType(na_object=None)
    assert dt.na_object is None and dt.coerce is True
    hashes.add(hash(dt))

    dt = StringDType(coerce=False)
    assert not hasattr(dt, "na_object") and dt.coerce is False
    hashes.add(hash(dt))

    dt = StringDType(na_object=None, coerce=False)
    assert dt.na_object is None and dt.coerce is False
    hashes.add(hash(dt))

    assert len(hashes) == 4

    dt = np.dtype("T")
    assert dt == StringDType()
    assert dt.kind == "T"
    assert dt.char == "T"

    hashes.add(hash(dt))
    assert len(hashes) == 4
    # 对 StringDType 对象的创建、属性和哈希值进行测试


def test_dtype_equality(dtype):
    assert dtype == dtype
    for ch in "SU":
        assert dtype != np.dtype(ch)
        assert dtype != np.dtype(f"{ch}8")
    # 测试 StringDType 对象的相等性比较


def test_dtype_repr(dtype):
    if not hasattr(dtype, "na_object") and dtype.coerce:
        assert repr(dtype) == "StringDType()"
    # 测试 StringDType 对象的字符串表示形式
    elif dtype.coerce:
        # 如果 dtype.coerce 为真，则进行以下断言
        assert repr(dtype) == f"StringDType(na_object={repr(dtype.na_object)})"
    elif not hasattr(dtype, "na_object"):
        # 如果 dtype 没有属性 "na_object"，则进行以下断言
        assert repr(dtype) == "StringDType(coerce=False)"
    else:
        # 其他情况下进行以下断言
        assert (
            repr(dtype)
            == f"StringDType(na_object={repr(dtype.na_object)}, coerce=False)"
        )
# 定义一个测试函数，用于测试指定数据类型的 na_object 属性是否存在
def test_create_with_na(dtype):
    # 如果数据类型没有 na_object 属性，则跳过测试
    if not hasattr(dtype, "na_object"):
        pytest.skip("does not have an na object")
    
    # 获取 na_object 属性的值
    na_val = dtype.na_object
    
    # 创建一个包含字符串和 na_val 的列表
    string_list = ["hello", na_val, "world"]
    
    # 使用指定的数据类型创建一个 NumPy 数组
    arr = np.array(string_list, dtype=dtype)
    
    # 断言数组转换成字符串后的格式是否正确
    assert str(arr) == "[" + " ".join([repr(s) for s in string_list]) + "]"
    
    # 断言数组的第二个元素是否为 na_object
    assert arr[1] is dtype.na_object


# 使用 pytest 的参数化装饰器，对 test_set_replace_na 函数进行多组参数化测试
@pytest.mark.parametrize("i", list(range(5)))
def test_set_replace_na(i):
    # 测试不同长度的字符串能否被设置为 NaN 并进行替换
    s_empty = ""  # 空字符串
    s_short = "0123456789"  # 短字符串
    s_medium = "abcdefghijklmnopqrstuvwxyz"  # 中等长度字符串
    s_long = "-=+" * 100  # 长字符串
    strings = [s_medium, s_empty, s_short, s_medium, s_long]
    
    # 使用带有自定义 NaN 对象的 StringDType 类型创建数组
    a = np.array(strings, StringDType(na_object=np.nan))
    
    # 遍历一系列字符串，并进行设置为 NaN 和替换的断言测试
    for s in [a[i], s_medium+s_short, s_short, s_empty, s_long]:
        a[i] = np.nan
        assert np.isnan(a[i])
        a[i] = s
        assert a[i] == s
        assert_array_equal(a, strings[:i] + [s] + strings[i+1:])


# 测试包含空字符的字符串数组能否正确存储和检索
def test_null_roundtripping():
    data = ["hello\0world", "ABC\0DEF\0\0"]
    arr = np.array(data, dtype="T")
    assert data[0] == arr[0]
    assert data[1] == arr[1]


# 测试如果字符串数组过大会触发 MemoryError 异常
def test_string_too_large_error():
    arr = np.array(["a", "b", "c"], dtype=StringDType())
    with pytest.raises(MemoryError):
        arr * (2**63 - 2)


# 使用参数化装饰器测试不同编码的字符串数组创建及其数据类型
@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "ghi"],  # ASCII 字符串
        ["🤣", "📵", "😰"],  # UTF-8 表情符号
        ["🚜", "🙃", "😾"],  # 更多的 UTF-8 表情符号
        ["😹", "🚠", "🚌"],  # 另一组 UTF-8 表情符号
    ],
)
def test_array_creation_utf8(dtype, data):
    # 使用指定的数据类型创建字符串数组，并进行断言检查
    arr = np.array(data, dtype=dtype)
    assert str(arr) == "[" + " ".join(["'" + str(d) + "'" for d in data]) + "]"
    assert arr.dtype == dtype


# 使用参数化装饰器测试不同类型数据的字符串转换
@pytest.mark.parametrize(
    ("data"),
    [
        [1, 2, 3],  # 整数数组
        [b"abc", b"def", b"ghi"],  # 字节串数组
        [object, object, object],  # 对象数组
    ],
)
def test_scalars_string_conversion(data, dtype):
    # 如果数据类型支持强制转换，则断言转换后的数组与预期一致
    if dtype.coerce:
        assert_array_equal(
            np.array(data, dtype=dtype),
            np.array([str(d) for d in data], dtype=dtype),
        )
    else:
        # 否则断言应该引发 ValueError 异常
        with pytest.raises(ValueError):
            np.array(data, dtype=dtype)


# 使用参数化装饰器测试不同字符串数组的自我类型转换
@pytest.mark.parametrize(
    ("strings"),
    [
        ["this", "is", "an", "array"],  # 普通字符串数组
        ["€", "", "😊"],  # 包含 UTF-8 表情符号的数组
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],  # 更复杂的 UTF-8 字符串数组
    ],
)
def test_self_casts(dtype, dtype2, strings):
    # 如果第一个数据类型具有 na_object 属性，则在数组末尾添加 na_object
    if hasattr(dtype, "na_object"):
        strings = strings + [dtype.na_object]
    elif hasattr(dtype2, "na_object"):
        strings = strings + [""]
    
    # 使用第一个数据类型创建字符串数组
    arr = np.array(strings, dtype=dtype)
    
    # 将数组转换为第二个数据类型，并进行相应的断言检查
    newarr = arr.astype(dtype2)
    
    if hasattr(dtype, "na_object") and not hasattr(dtype2, "na_object"):
        assert newarr[-1] == str(dtype.na_object)
        with pytest.raises(TypeError):
            arr.astype(dtype2, casting="safe")
    elif hasattr(dtype, "na_object") and hasattr(dtype2, "na_object"):
        assert newarr[-1] is dtype2.na_object
        arr.astype(dtype2, casting="safe")
    elif hasattr(dtype2, "na_object"):
        # 如果dtype2具有属性"na_object"，则执行以下操作
        assert newarr[-1] == ""
        # 断言新数组的最后一个元素为空字符串
        arr.astype(dtype2, casting="safe")
        # 将数组arr转换为dtype2类型，使用安全转换模式
    else:
        # 如果dtype2没有属性"na_object"，则执行以下操作
        arr.astype(dtype2, casting="safe")
        # 将数组arr转换为dtype2类型，使用安全转换模式

    if hasattr(dtype, "na_object") and hasattr(dtype2, "na_object"):
        # 如果dtype和dtype2都具有属性"na_object"，则执行以下操作
        na1 = dtype.na_object
        # 获取dtype的na_object属性值，赋给na1
        na2 = dtype2.na_object
        # 获取dtype2的na_object属性值，赋给na2
        if ((na1 is not na2 and
             # 如果na1不等于na2，并且满足以下条件：
             # 首先检查pd_NA，因为bool(pd_NA)会导致错误
             ((na1 is pd_NA or na2 is pd_NA) or
              # 第二个条件是NaN检查，采用这种方式避免math.isnan和np.isnan的错误
              (na1 != na2 and not (na1 != na1 and na2 != na2))))):
            # 如果上述条件满足，则执行以下操作
            with pytest.raises(TypeError):
                # 使用pytest断言抛出TypeError异常
                arr[:-1] == newarr[:-1]
            return
        # 如果条件不满足，则不返回

    assert_array_equal(arr[:-1], newarr[:-1])
    # 使用assert_array_equal函数断言数组arr和newarr的前n-1个元素相等
@pytest.mark.parametrize(
    ("strings"),  # 使用 pytest.mark.parametrize 装饰器，参数为一个元组，包含一个名为 "strings" 的参数
    [
        ["this", "is", "an", "array"],  # 参数 "strings" 的第一个测试数据是包含字符串的列表
        ["€", "", "😊"],  # 参数 "strings" 的第二个测试数据是包含特殊字符的列表
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],  # 参数 "strings" 的第三个测试数据是包含特殊字符和空格的列表
    ],
)
class TestStringLikeCasts:
    def test_unicode_casts(self, dtype, strings):
        arr = np.array(strings, dtype=np.str_).astype(dtype)  # 将字符串列表转换为指定数据类型的 NumPy 数组
        expected = np.array(strings, dtype=dtype)  # 创建期望结果的 NumPy 数组
        assert_array_equal(arr, expected)  # 断言两个数组相等

        arr_as_U8 = expected.astype("U8")  # 将期望结果数组转换为 UTF-8 编码的字符串数组
        assert_array_equal(arr_as_U8, np.array(strings, dtype="U8"))  # 断言两个数组相等
        assert_array_equal(arr_as_U8.astype(dtype), arr)  # 断言转换后的数组与原始数组相等
        arr_as_U3 = expected.astype("U3")  # 将期望结果数组转换为长度不超过3的字符串数组
        assert_array_equal(arr_as_U3, np.array(strings, dtype="U3"))  # 断言两个数组相等
        assert_array_equal(
            arr_as_U3.astype(dtype),
            np.array([s[:3] for s in strings], dtype=dtype),  # 断言转换后的数组与预期截取长度后的数组相等
        )

    def test_void_casts(self, dtype, strings):
        sarr = np.array(strings, dtype=dtype)  # 创建指定数据类型的字符串数组
        utf8_bytes = [s.encode("utf-8") for s in strings]  # 将字符串列表编码为 UTF-8 字节列表
        void_dtype = f"V{max([len(s) for s in utf8_bytes])}"  # 计算 UTF-8 字节的最大长度，构建 void 类型字符串
        varr = np.array(utf8_bytes, dtype=void_dtype)  # 创建 void 类型的数组
        assert_array_equal(varr, sarr.astype(void_dtype))  # 断言两个数组相等
        assert_array_equal(varr.astype(dtype), sarr)  # 断言转换后的数组与原始数组相等

    def test_bytes_casts(self, dtype, strings):
        sarr = np.array(strings, dtype=dtype)  # 创建指定数据类型的字符串数组
        try:
            utf8_bytes = [s.encode("ascii") for s in strings]  # 尝试将字符串列表编码为 ASCII 字节列表
            bytes_dtype = f"S{max([len(s) for s in utf8_bytes])}"  # 计算 ASCII 字节的最大长度，构建 bytes 类型字符串
            barr = np.array(utf8_bytes, dtype=bytes_dtype)  # 创建 bytes 类型的数组
            assert_array_equal(barr, sarr.astype(bytes_dtype))  # 断言两个数组相等
            assert_array_equal(barr.astype(dtype), sarr)  # 断言转换后的数组与原始数组相等
        except UnicodeEncodeError:
            with pytest.raises(UnicodeEncodeError):  # 捕获 Unicode 编码错误
                sarr.astype("S20")  # 尝试将字符串数组转换为长度不超过20的 ASCII 字符串数组


def test_additional_unicode_cast(random_string_list, dtype):
    arr = np.array(random_string_list, dtype=dtype)  # 创建指定数据类型的随机字符串数组
    # 测试是否能正确地短路
    assert_array_equal(arr, arr.astype(arr.dtype))  # 断言两个数组相等
    # 通过比较促进器测试转换
    assert_array_equal(arr, arr.astype(random_string_list.dtype))  # 断言两个数组相等


def test_insert_scalar(dtype, string_list):
    """测试插入标量是否正常工作。"""
    arr = np.array(string_list, dtype=dtype)  # 创建指定数据类型的字符串数组
    scalar_instance = "what"  # 创建一个标量实例
    arr[1] = scalar_instance  # 将标量插入数组的第二个位置
    assert_array_equal(
        arr,
        np.array(string_list[:1] + ["what"] + string_list[2:], dtype=dtype),  # 断言两个数组相等
    )


comparison_operators = [
    np.equal,
    np.not_equal,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
]


@pytest.mark.parametrize("op", comparison_operators)
@pytest.mark.parametrize("o_dtype", [np.str_, object, StringDType()])
def test_comparisons(string_list, dtype, op, o_dtype):
    sarr = np.array(string_list, dtype=dtype)  # 创建指定数据类型的字符串数组
    oarr = np.array(string_list, dtype=o_dtype)  # 创建指定数据类型的字符串数组

    # 测试比较操作符是否工作
    res = op(sarr, sarr)  # 执行比较操作
    ores = op(oarr, oarr)  # 执行比较操作
    # 测试类型提升是否正常工作
    orres = op(sarr, oarr)  # 执行比较操作
    olres = op(oarr, sarr)  # 执行比较操作
    # 检查两个 NumPy 数组是否相等，如果不相等则抛出 AssertionError
    assert_array_equal(res, ores)
    assert_array_equal(res, orres)
    assert_array_equal(res, olres)

    # 测试对不等长度的字符串数组进行操作时是否得到正确的结果
    # 创建一个新的字符串数组 sarr2，其中每个字符串都在原字符串后面加上字符 "2"
    sarr2 = np.array([s + "2" for s in string_list], dtype=dtype)
    # 创建一个新的字符串数组 oarr2，其中每个字符串都在原字符串后面加上字符 "2"
    oarr2 = np.array([s + "2" for s in string_list], dtype=o_dtype)

    # 对字符串数组 sarr 和 sarr2 执行操作 op，得到结果 res
    # 对字符串数组 oarr 和 oarr2 执行操作 op，得到结果 ores
    # 对字符串数组 oarr 和 sarr2 执行操作 op，得到结果 olres
    # 对字符串数组 sarr 和 oarr2 执行操作 op，得到结果 orres
    res = op(sarr, sarr2)
    ores = op(oarr, oarr2)
    olres = op(oarr, sarr2)
    orres = op(sarr, oarr2)

    # 检查以上四组操作的结果是否相等，如果不相等则抛出 AssertionError
    assert_array_equal(res, ores)
    assert_array_equal(res, olres)
    assert_array_equal(res, orres)

    # 对字符串数组 sarr2 和 sarr 执行操作 op，得到结果 res
    # 对字符串数组 oarr2 和 oarr 执行操作 op，得到结果 ores
    # 对字符串数组 oarr2 和 sarr 执行操作 op，得到结果 olres
    # 对字符串数组 sarr2 和 oarr 执行操作 op，得到结果 orres
    res = op(sarr2, sarr)
    ores = op(oarr2, oarr)
    olres = op(oarr2, sarr)
    orres = op(sarr2, oarr)

    # 检查以上四组操作的结果是否相等，如果不相等则抛出 AssertionError
    assert_array_equal(res, ores)
    assert_array_equal(res, olres)
    assert_array_equal(res, orres)
# 定义一个测试函数，用于检查特定数据类型和字符串列表的 NaN（Not a Number）处理
def test_isnan(dtype, string_list):
    # 如果数据类型 dtype 没有 na_object 属性，跳过测试并提示无 NaN 支持
    if not hasattr(dtype, "na_object"):
        pytest.skip("no na support")
    
    # 将字符串列表和 dtype.na_object 合并为一个 numpy 数组 sarr
    sarr = np.array(string_list + [dtype.na_object], dtype=dtype)
    
    # 检查 dtype.na_object 是否为浮点数 NaN，并且使用 np.isnan 检查是否为 NaN
    is_nan = isinstance(dtype.na_object, float) and np.isnan(dtype.na_object)
    
    # 初始化 bool_errors 记录是否发生 TypeError 异常
    bool_errors = 0
    try:
        # 尝试将 dtype.na_object 转换为布尔值，捕获 TypeError 异常
        bool(dtype.na_object)
    except TypeError:
        bool_errors = 1
    
    # 如果是 NaN 或者存在 bool_errors 异常
    if is_nan or bool_errors:
        # 断言 np.isnan(sarr) 的结果与预期结果相等，预期最后一个值为 1，其余为 0
        assert_array_equal(
            np.isnan(sarr),
            np.array([0] * len(string_list) + [1], dtype=np.bool),
        )
    else:
        # 如果没有 NaN 或者异常，断言 sarr 中不存在 NaN
        assert not np.any(np.isnan(sarr))


# 定义一个测试函数，用于测试 numpy 数组的序列化和反序列化（pickle）
def test_pickle(dtype, string_list):
    # 将字符串列表 string_list 转换为 numpy 数组 arr，并指定数据类型为 dtype
    arr = np.array(string_list, dtype=dtype)

    # 使用临时文件存储序列化后的 arr 和 dtype
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        pickle.dump([arr, dtype], f)

    # 从临时文件中读取反序列化的结果 res
    with open(f.name, "rb") as f:
        res = pickle.load(f)

    # 断言反序列化后的结果与原始 arr 相等
    assert_array_equal(res[0], arr)
    # 断言反序列化后的数据类型与原始 dtype 相等
    assert res[1] == dtype

    # 删除临时文件
    os.remove(f.name)


# 使用 pytest.mark.parametrize 对 test_sort 函数进行参数化测试
@pytest.mark.parametrize(
    "strings",
    [
        ["left", "right", "leftovers", "righty", "up", "down"],
        ["left" * 10, "right" * 10, "leftovers" * 10, "righty" * 10, "up" * 10],
        ["🤣🤣", "🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_sort(dtype, strings):
    """Test that sorting matches python's internal sorting."""
    
    # 定义内部函数 test_sort，用于测试排序功能
    def test_sort(strings, arr_sorted):
        # 创建 numpy 数组 arr，并使用 np.random.default_rng().shuffle 随机打乱顺序
        arr = np.array(strings, dtype=dtype)
        np.random.default_rng().shuffle(arr)
        
        # 获取 dtype 的 na_object 属性
        na_object = getattr(arr.dtype, "na_object", "")
        
        # 如果 na_object 为 None 并且字符串列表中包含 None，预期会抛出 ValueError 异常
        if na_object is None and None in strings:
            with pytest.raises(
                ValueError,
                match="Cannot compare null that is not a nan-like value",
            ):
                arr.sort()
        else:
            # 否则，对 arr 进行排序
            arr.sort()
            # 断言排序后的 arr 与预期的 arr_sorted 相等，支持 NaN 相等判断
            assert np.array_equal(arr, arr_sorted, equal_nan=True)
    
    # 复制 strings 列表，避免修改测试的固定列表
    strings = strings.copy()
    
    # 使用 sorted 函数创建预期的排序后的数组 arr_sorted
    arr_sorted = np.array(sorted(strings), dtype=dtype)
    
    # 调用 test_sort 函数进行测试
    test_sort(strings, arr_sorted)

    # 如果 dtype 没有 na_object 属性，直接返回
    if not hasattr(dtype, "na_object"):
        return

    # 确保 NaN 被排序到数组末尾，并且字符串类型的 NaN 被按照字符串规则排序
    strings.insert(0, dtype.na_object)
    strings.insert(2, dtype.na_object)
    
    # 如果 na_object 不是字符串类型，将其添加到 arr_sorted 的末尾
    if not isinstance(dtype.na_object, str):
        arr_sorted = np.array(
            arr_sorted.tolist() + [dtype.na_object, dtype.na_object],
            dtype=dtype,
        )
    else:
        # 否则，按照字符串规则重新排序 strings，并创建 arr_sorted
        arr_sorted = np.array(sorted(strings), dtype=dtype)

    # 再次调用 test_sort 函数进行测试
    test_sort(strings, arr_sorted)


# 使用 pytest.mark.parametrize 对 test_sort 函数进行参数化测试
@pytest.mark.parametrize(
    "strings",
    [
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
        ["A¢☃€ 😊", "", " ", " "],
        ["", "a", "😸", "ááðfáíóåéë"],
    ],
)
def test_nonzero(strings, na_object):
    # 获取 na_object 的数据类型
    dtype = get_dtype(na_object)
    # 将 strings 转换为 numpy 数组，使用指定的数据类型
    arr = np.array(strings, dtype=dtype)
    # 找出非零元素的索引
    is_nonzero = np.array(
        [i for i, item in enumerate(strings) if len(item) != 0])
    # 断言非零元素的索引与计算得到的索引数组相等
    assert_array_equal(arr.nonzero()[0], is_nonzero)

    # 如果 na_object 不是 pd_NA 并且等于 'unset'，则返回
    if na_object is not pd_NA and na_object == 'unset':
        return

    # 创建包含 na_object 的新数组
    strings_with_na = np.array(strings + [na_object], dtype=dtype)
    # 检查是否存在 NaN，返回布尔值
    is_nan = np.isnan(np.array([dtype.na_object], dtype=dtype))[0]

    # 如果存在 NaN
    if is_nan:
        # 断言带有 na_object 的数组的最后一个非零元素的索引为 4
        assert strings_with_na.nonzero()[0][-1] == 4
    else:
        # 否则，断言最后一个非零元素的索引为 3
        assert strings_with_na.nonzero()[0][-1] == 3

    # 检查将数组转换为布尔值后的非零元素索引是否与原数组的非零元素相同
    assert_array_equal(strings_with_na[strings_with_na.nonzero()],
                       strings_with_na[strings_with_na.astype(bool)])


def test_where(string_list, na_object):
    # 获取 na_object 的数据类型
    dtype = get_dtype(na_object)
    # 将 string_list 转换为 numpy 数组，使用指定的数据类型
    a = np.array(string_list, dtype=dtype)
    # 创建 a 的逆序数组
    b = a[::-1]
    # 根据条件选择返回 a 或 b 中的元素
    res = np.where([True, False, True, False, True, False], a, b)
    # 断言结果数组与预期结果相等
    assert_array_equal(res, [a[0], b[1], a[2], b[3], a[4], b[5]])


def test_fancy_indexing(string_list):
    # 将 string_list 转换为 numpy 数组，数据类型为 "T"（字符串类型）
    sarr = np.array(string_list, dtype="T")
    # 断言数组与使用其索引创建的新数组相等
    assert_array_equal(sarr, sarr[np.arange(sarr.shape[0])])


def test_creation_functions():
    # 断言创建的全为字符串空数组与预期结果相等
    assert_array_equal(np.zeros(3, dtype="T"), ["", "", ""])
    # 断言创建的空数组与预期结果相等
    assert_array_equal(np.empty(3, dtype="T"), ["", "", ""])

    # 断言全为字符串的零数组的第一个元素为空字符串
    assert np.zeros(3, dtype="T")[0] == ""
    # 断言空数组的第一个元素为空字符串
    assert np.empty(3, dtype="T")[0] == ""


def test_concatenate(string_list):
    # 将 string_list 转换为 numpy 数组，数据类型为 "T"（字符串类型）
    sarr = np.array(string_list, dtype="T")
    # 拼接数组 sarr 自身，沿第 0 轴
    sarr_cat = np.array(string_list + string_list, dtype="T")

    # 断言拼接后的结果与预期结果相等
    assert_array_equal(np.concatenate([sarr], axis=0), sarr)


def test_create_with_copy_none(string_list):
    # 将 string_list 转换为 numpy 数组，数据类型为 StringDType()
    arr = np.array(string_list, dtype=StringDType())
    # 创建 arr 的逆序数组，数据类型与 arr 相同
    arr_rev = np.array(string_list[::-1], dtype=StringDType())

    # 创建 arr 的副本，确保新数组与 arr_rev 不共享内存分配器或 arena
    arr_copy = np.array(arr, copy=None, dtype=arr_rev.dtype)
    np.testing.assert_array_equal(arr, arr_copy)
    assert arr_copy.base is None

    # 使用 copy=False 时，应抛出 ValueError 异常
    with pytest.raises(ValueError, match="Unable to avoid copy"):
        np.array(arr, copy=False, dtype=arr_rev.dtype)

    # 因为使用了 arr 的 dtype 实例，因此视图是安全的
    arr_view = np.array(arr, copy=None, dtype=arr.dtype)
    np.testing.assert_array_equal(arr, arr)
    np.testing.assert_array_equal(arr_view[::-1], arr_rev)
    assert arr_view is arr


def test_astype_copy_false():
    orig_dt = StringDType()
    # 创建包含字符串的数组，数据类型为 StringDType()
    arr = np.array(["hello", "world"], dtype=StringDType())
    # 断言使用 copy=False 时，不会创建 arr 的副本
    assert not arr.astype(StringDType(coerce=False), copy=False).dtype.coerce

    # 断言使用指定的 dtype 时，不会创建 arr 的副本
    assert arr.astype(orig_dt, copy=False).dtype is orig_dt
    # 一个包含多个子列表的列表，每个子列表包含不同数量的字符串元素
    [
        # 第一个子列表包含 6 个字符串元素
        ["left", "right", "leftovers", "righty", "up", "down"],
        # 第二个子列表包含 4 个字符串元素，包括表情符号和文字
        ["🤣🤣", "🤣", "📵", "😰"],
        # 第三个子列表包含 3 个字符串元素，都是表情符号
        ["🚜", "🙃", "😾"],
        # 第四个子列表包含 3 个字符串元素，都是表情符号
        ["😹", "🚠", "🚌"],
        # 第五个子列表包含 4 个字符串元素，包括特殊符号和表情符号
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
# 导入所需的库或模块
)
def test_argmax(strings):
    """Test that argmax/argmin matches what python calculates."""
    # 将字符串列表转换为 numpy 数组，数据类型为 T (字符串类型)
    arr = np.array(strings, dtype="T")
    # 断言 numpy 计算的最大索引与 Python 内置函数 max 返回的索引相同
    assert np.argmax(arr) == strings.index(max(strings))
    # 断言 numpy 计算的最小索引与 Python 内置函数 min 返回的索引相同
    assert np.argmin(arr) == strings.index(min(strings))


@pytest.mark.parametrize(
    "arrfunc,expected",
    [
        [np.sort, None],  # 测试 np.sort 函数，期望结果为 None
        [np.nonzero, (np.array([], dtype=np.int_),)],  # 测试 np.nonzero 函数，期望返回空数组
        [np.argmax, 0],  # 测试 np.argmax 函数，期望返回索引 0
        [np.argmin, 0],  # 测试 np.argmin 函数，期望返回索引 0
    ],
)
def test_arrfuncs_zeros(arrfunc, expected):
    # 创建一个长度为 10 的零数组，数据类型为 T (字符串类型)
    arr = np.zeros(10, dtype="T")
    # 调用被测函数 arrfunc，得到结果
    result = arrfunc(arr)
    # 如果期望结果为 None，则将其设为 arr 本身
    if expected is None:
        expected = arr
    # 断言两个数组是否相等
    assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize(
    ("strings", "cast_answer", "any_answer", "all_answer"),
    [
        [["hello", "world"], [True, True], True, True],  # 测试包含非空字符串的情况
        [["", ""], [False, False], False, False],  # 测试全为空字符串的情况
        [["hello", ""], [True, False], True, False],  # 测试有一个非空字符串的情况
        [["", "world"], [False, True], True, False],  # 测试有一个非空字符串的情况
    ],
)
def test_cast_to_bool(strings, cast_answer, any_answer, all_answer):
    # 创建一个字符串数组，数据类型为 T (字符串类型)
    sarr = np.array(strings, dtype="T")
    # 断言将数组转换为布尔型数组后是否与期望结果 cast_answer 相等
    assert_array_equal(sarr.astype("bool"), cast_answer)

    # 断言数组中是否有任意非零元素，结果应与 any_answer 相等
    assert np.any(sarr) == any_answer
    # 断言数组中所有元素是否都非零，结果应与 all_answer 相等
    assert np.all(sarr) == all_answer


@pytest.mark.parametrize(
    ("strings", "cast_answer"),
    [
        [[True, True], ["True", "True"]],  # 测试全部为 True 的布尔数组转换为字符串数组
        [[False, False], ["False", "False"]],  # 测试全部为 False 的布尔数组转换为字符串数组
        [[True, False], ["True", "False"]],  # 测试包含 True 和 False 的布尔数组转换为字符串数组
        [[False, True], ["False", "True"]],  # 测试包含 False 和 True 的布尔数组转换为字符串数组
    ],
)
def test_cast_from_bool(strings, cast_answer):
    # 创建一个布尔数组
    barr = np.array(strings, dtype=bool)
    # 断言将布尔数组转换为字符串数组后是否与期望结果 cast_answer 相等
    assert_array_equal(barr.astype("T"), np.array(cast_answer, dtype="T"))


@pytest.mark.parametrize("bitsize", [8, 16, 32, 64])
@pytest.mark.parametrize("signed", [True, False])
def test_sized_integer_casts(bitsize, signed):
    # 根据参数动态生成整数类型的字符串表示
    idtype = f"int{bitsize}"
    # 根据 signed 参数确定输入数组 inp
    if signed:
        # 生成输入数组，包括负数和正数
        inp = [-(2**p - 1) for p in reversed(range(bitsize - 1))]
        inp += [2**p - 1 for p in range(1, bitsize - 1)]
    else:
        # 生成无符号整数的输入数组
        idtype = "u" + idtype
        inp = [2**p - 1 for p in range(bitsize)]
    # 创建 numpy 数组，数据类型为动态生成的整数类型 idtype
    ainp = np.array(inp, dtype=idtype)
    # 断言两个数组是否相等
    assert_array_equal(ainp, ainp.astype("T").astype(idtype))

    # 测试安全转换是否有效
    ainp.astype("T", casting="safe")

    # 测试不安全转换是否会引发 TypeError
    with pytest.raises(TypeError):
        ainp.astype("T").astype(idtype, casting="safe")

    # 测试超出范围的输入是否会引发 OverflowError
    oob = [str(2**bitsize), str(-(2**bitsize))]
    with pytest.raises(OverflowError):
        np.array(oob, dtype="T").astype(idtype)

    # 测试无法解析的字符串输入是否会引发 ValueError
    with pytest.raises(ValueError):
        np.array(["1", np.nan, "3"],
                 dtype=StringDType(na_object=np.nan)).astype(idtype)


@pytest.mark.parametrize("typename", ["byte", "short", "int", "longlong"])
@pytest.mark.parametrize("signed", ["", "u"])
def test_unsized_integer_casts(typename, signed):
    # 根据参数动态生成整数类型的字符串表示
    idtype = f"{signed}{typename}"

    # 创建一个整数数组
    inp = [1, 2, 3, 4]
    # 创建 numpy 数组，数据类型为动态生成的整数类型 idtype
    ainp = np.array(inp, dtype=idtype)
    # 断言两个数组是否相等
    assert_array_equal(ainp, ainp.astype("T").astype(idtype))
    [
        # 使用 pytest.param 创建一个参数化测试的参数，将 "longdouble" 作为参数值
        pytest.param(
            "longdouble",
            # 标记这个测试为预期失败，如果条件不符合会失败，且严格模式下失败
            marks=pytest.mark.xfail(
                # 检查 np.dtypes.LongDoubleDType() 是否不等于 np.dtypes.Float64DType()
                np.dtypes.LongDoubleDType() != np.dtypes.Float64DType(),
                # 如果条件不符合，失败原因为 "numpy lacks an ld2a implementation"
                reason="numpy lacks an ld2a implementation",
                # 使用严格模式，条件不符合时严格失败
                strict=True,
            ),
        ),
        # 下面三个元素都是普通的字符串 "float64", "float32", "float16"
        "float64",
        "float32",
        "float16",
    ],
# 定义一个测试函数，用于测试浮点数类型的转换
def test_float_casts(typename):
    # 定义输入的浮点数列表
    inp = [1.1, 2.8, -3.2, 2.7e4]
    # 将输入列表转换为指定类型的 NumPy 数组
    ainp = np.array(inp, dtype=typename)
    # 断言：将数组先转换为字符串类型，再转回指定类型，应与原数组相等
    assert_array_equal(ainp, ainp.astype("T").astype(typename))

    # 另一组输入数据
    inp = [0.1]
    # 将输入列表转换为指定类型的 NumPy 数组，并转换为字符串类型
    sres = np.array(inp, dtype=typename).astype("T")
    # 再将字符串类型的数组转回指定类型
    res = sres.astype(typename)
    # 断言：应当与原始输入数组相等
    assert_array_equal(np.array(inp, dtype=typename), res)
    # 断言：转换后的字符串数组第一个元素应为 "0.1"
    assert sres[0] == "0.1"

    # 如果指定类型为 "longdouble"，则跳过，不进行下面的测试
    if typename == "longdouble":
        return

    # 获取指定类型的浮点数信息
    fi = np.finfo(typename)

    # 更复杂的输入数据和期望结果
    inp = [1e-324, fi.smallest_subnormal, -1e-324, -fi.smallest_subnormal]
    eres = [0, fi.smallest_subnormal, -0, -fi.smallest_subnormal]
    # 断言：先转为字符串类型，再转回指定类型，结果应与期望一致
    res = np.array(inp, dtype=typename).astype("T").astype(typename)
    assert_array_equal(eres, res)

    # 更复杂的输入数据和期望结果
    inp = [2e308, fi.max, -2e308, fi.min]
    eres = [np.inf, fi.max, -np.inf, fi.min]
    # 断言：先转为字符串类型，再转回指定类型，结果应与期望一致
    res = np.array(inp, dtype=typename).astype("T").astype(typename)
    assert_array_equal(eres, res)


# 使用 pytest 的参数化功能，对复数浮点数类型进行测试
@pytest.mark.parametrize(
    "typename",
    [
        "csingle",
        "cdouble",
        # 复数长双精度类型的测试，标记为预期失败，因为 numpy 缺乏 ld2a 实现
        pytest.param(
            "clongdouble",
            marks=pytest.mark.xfail(
                np.dtypes.CLongDoubleDType() != np.dtypes.Complex128DType(),
                reason="numpy lacks an ld2a implementation",
                strict=True,
            ),
        ),
    ],
)
# 定义测试函数，用于测试复数浮点数类型的转换
def test_cfloat_casts(typename):
    # 定义复数浮点数输入列表
    inp = [1.1 + 1.1j, 2.8 + 2.8j, -3.2 - 3.2j, 2.7e4 + 2.7e4j]
    # 将输入列表转换为指定类型的复数 NumPy 数组
    ainp = np.array(inp, dtype=typename)
    # 断言：将数组先转换为字符串类型，再转回指定类型，应与原数组相等
    assert_array_equal(ainp, ainp.astype("T").astype(typename))

    # 另一组复数浮点数输入数据
    inp = [0.1 + 0.1j]
    # 将输入列表转换为指定类型的复数 NumPy 数组，并转换为字符串类型
    sres = np.array(inp, dtype=typename).astype("T")
    # 再将字符串类型的数组转回指定类型
    res = sres.astype(typename)
    # 断言：应当与原始输入数组相等
    assert_array_equal(np.array(inp, dtype=typename), res)
    # 断言：转换后的字符串数组第一个元素应为 "(0.1+0.1j)"
    assert sres[0] == "(0.1+0.1j)"


# 定义测试函数，用于测试字符串数组的索引取值操作
def test_take(string_list):
    # 将字符串列表转换为通用字符串类型的 NumPy 数组
    sarr = np.array(string_list, dtype="T")
    # 使用 np.arange(len(string_list)) 进行索引取值
    res = sarr.take(np.arange(len(string_list)))
    # 断言：取出的结果应与原始数组相等
    assert_array_equal(sarr, res)

    # 进一步测试带有输出参数的索引取值操作
    out = np.empty(len(string_list), dtype="T")
    out[0] = "hello"
    # 使用 np.arange(len(string_list)) 进行索引取值，并将结果存入 out 数组
    res = sarr.take(np.arange(len(string_list)), out=out)
    # 断言：返回的结果应该是 out 数组本身
    assert res is out
    # 断言：取出的结果应与原始数组相等
    assert_array_equal(sarr, res)


# 使用 pytest 的参数化功能，对最小和最大函数进行测试
@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize(
    "ufunc_name,func",
    [
        ("min", min),
        ("max", max),
    ],
)
# 定义测试函数，测试最小和最大函数的行为是否符合 Python 内建的 min/max 函数
def test_ufuncs_minmax(string_list, ufunc_name, func, use_out):
    """Test that the min/max ufuncs match Python builtin min/max behavior."""
    # 将字符串列表转换为通用字符串类型的 NumPy 数组
    arr = np.array(string_list, dtype="T")
    # 将字符串列表转换为普通字符串类型的 NumPy 数组
    uarr = np.array(string_list, dtype=str)
    # 调用 Python 内建的 min/max 函数计算期望结果
    res = np.array(func(string_list), dtype="T")
    # 断言：使用 NumPy 中的 ufunc 函数，结果应与期望一致
    assert_array_equal(getattr(arr, ufunc_name)(), res)

    # 获取对应的 NumPy ufunc 函数对象
    ufunc = getattr(np, ufunc_name + "imum")

    if use_out:
        # 如果 use_out 为 True，则使用 out 参数存储结果
        res = ufunc(arr, arr, out=arr)
    else:
        # 否则直接调用 ufunc 函数计算结果
        res = ufunc(arr, arr)

    # 断言：使用 ufunc 函数计算的结果应与原始列表的结果一致
    assert_array_equal(uarr, res)
    # 断言：使用 NumPy 中的 ufunc 函数，结果应与期望一致
    assert_array_equal(getattr(arr, ufunc_name)(), func(string_list))


# 定义测试函数，测试最大值函数的回归问题
def test_max_regression():
    # 将字符串列表转换为通用字符串类型的 NumPy 数组
    arr = np.array(['y', 'y', 'z'], dtype="T")
    # 断言：数组中的最大值应为 'z'
    assert arr.max() == 'z'
# 使用 pytest.mark.parametrize 装饰器定义参数化测试，参数 use_out 分别为 True 和 False
# 参数化 other_strings 包含三个列表，每个列表包含不同的字符串元素
@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize(
    "other_strings",
    [
        ["abc", "def" * 500, "ghi" * 16, "🤣" * 100, "📵", "😰"],
        ["🚜", "🙃", "😾", "😹", "🚠", "🚌"],
        ["🥦", "¨", "⨯", "∰ ", "⨌ ", "⎶ "],
    ],
)
# 定义名为 test_ufunc_add 的测试函数，参数为 dtype, string_list, other_strings, use_out
def test_ufunc_add(dtype, string_list, other_strings, use_out):
    # 根据 string_list 和 other_strings 创建 NumPy 数组 arr1 和 arr2，数据类型为 dtype
    arr1 = np.array(string_list, dtype=dtype)
    arr2 = np.array(other_strings, dtype=dtype)
    # 创建结果数组 result，其中元素为 arr1 和 arr2 对应元素相加的结果，数据类型为 dtype
    result = np.array([a + b for a, b in zip(arr1, arr2)], dtype=dtype)

    # 根据 use_out 的值选择是否将结果存入 arr1，调用 np.add 函数进行数组加法操作
    if use_out:
        res = np.add(arr1, arr2, out=arr1)
    else:
        res = np.add(arr1, arr2)

    # 断言 res 数组与预期结果 result 相等
    assert_array_equal(res, result)

    # 若 dtype 没有属性 "na_object"，直接返回
    if not hasattr(dtype, "na_object"):
        return

    # 检查 dtype.na_object 是否为 float 类型的 NaN 或者字符串类型
    is_nan = isinstance(dtype.na_object, float) and np.isnan(dtype.na_object)
    is_str = isinstance(dtype.na_object, str)
    bool_errors = 0
    try:
        bool(dtype.na_object)
    except TypeError:
        bool_errors = 1

    # 创建新的 arr1 和 arr2 数组，分别加入 dtype.na_object 作为第一个和最后一个元素
    arr1 = np.array([dtype.na_object] + string_list, dtype=dtype)
    arr2 = np.array(other_strings + [dtype.na_object], dtype=dtype)

    # 若 is_nan 或 bool_errors 或 is_str 为真，调用 np.add 进行数组加法操作
    if is_nan or bool_errors or is_str:
        res = np.add(arr1, arr2)
        # 断言 res 数组中间部分与 arr1 和 arr2 中间部分的加法结果相等
        assert_array_equal(res[1:-1], arr1[1:-1] + arr2[1:-1])
        # 根据 is_str 的不同情况断言 res 的第一个和最后一个元素与预期结果相等
        if not is_str:
            assert res[0] is dtype.na_object and res[-1] is dtype.na_object
        else:
            assert res[0] == dtype.na_object + arr2[0]
            assert res[-1] == arr1[-1] + dtype.na_object
    else:
        # 若不满足前述条件，期望抛出 ValueError 异常
        with pytest.raises(ValueError):
            np.add(arr1, arr2)


# 定义名为 test_ufunc_add_reduce 的测试函数，参数为 dtype
def test_ufunc_add_reduce(dtype):
    # 创建包含字符串元素的 NumPy 数组 arr，数据类型为 dtype
    values = ["a", "this is a long string", "c"]
    arr = np.array(values, dtype=dtype)
    # 创建空的 out 数组，数据类型为 dtype
    out = np.empty((), dtype=dtype)

    # 创建期望结果 expected，为 arr 中所有字符串元素拼接而成的数组，数据类型为 dtype
    expected = np.array("".join(values), dtype=dtype)
    # 断言 np.add.reduce(arr) 的结果与 expected 相等
    assert_array_equal(np.add.reduce(arr), expected)

    # 使用 out 参数调用 np.add.reduce(arr)，结果存入 out 数组，断言 out 与 expected 相等
    np.add.reduce(arr, out=out)
    assert_array_equal(out, expected)


# 定义名为 test_add_promoter 的测试函数，参数为 string_list
def test_add_promoter(string_list):
    # 创建字符串类型的 NumPy 数组 arr，数据类型为 StringDType()
    arr = np.array(string_list, dtype=StringDType())
    # 创建 lresult 和 rresult 数组，分别为 arr 中每个字符串元素前后添加 "hello" 而得到的结果数组
    lresult = np.array(["hello" + s for s in string_list], dtype=StringDType())
    rresult = np.array([s + "hello" for s in string_list], dtype=StringDType())

    # 遍历操作符 op，分别断言 op + arr 和 arr + op 的结果与 lresult 和 rresult 相等
    for op in ["hello", np.str_("hello"), np.array(["hello"])]:
        assert_array_equal(op + arr, lresult)
        assert_array_equal(arr + op, rresult)


# 定义名为 test_add_promoter_reduce 的测试函数
def test_add_promoter_reduce():
    # 使用 pytest.raises 断言调用 np.add.reduce(np.array(["a", "b"], dtype="U")) 会抛出 TypeError 异常
    with pytest.raises(TypeError, match="the resolved dtypes are not"):
        np.add.reduce(np.array(["a", "b"], dtype="U"))

    # 调用 np.add.reduce(np.array(["a", "b"], dtype="U"), dtype=np.dtypes.StringDType) 确保在 *ufunc* 中使用 dtype=T 可行
    np.add.reduce(np.array(["a", "b"], dtype="U"), dtype=np.dtypes.StringDType)


# 定义名为 test_multiply_reduce 的测试函数
def test_multiply_reduce():
    # 创建重复次数的 NumPy 数组 repeats，初始值为 "school-🚌"，数据类型为 np.dtypes.StringDType
    repeats = np.array([2, 3, 4])
    val = "school-🚌"
    # 调用 np.multiply.reduce(repeats, initial=val, dtype=np.dtypes.StringDType) 进行 reduce 操作
    res = np.multiply.reduce(repeats, initial=val, dtype=np.dtypes.StringDType)
    # 断言语句，用于确保变量 res 的值等于 val 乘以 repeats 列表中所有元素的乘积
    assert res == val * np.prod(repeats)
# 定义一个测试函数，测试当输入为字符串数组时，调用 np.multiply 是否会引发特定异常
def test_multiply_two_string_raises():
    # 创建一个包含字符串数组的 NumPy 数组，指定数据类型为 "T"（字符串）
    arr = np.array(["hello", "world"], dtype="T")
    # 使用 pytest 的上下文管理器检查是否会引发指定的异常
    with pytest.raises(np._core._exceptions._UFuncNoLoopError):
        # 调用 np.multiply 尝试对字符串数组进行乘法运算，预期会引发异常
        np.multiply(arr, arr)


# 使用 pytest.mark.parametrize 来定义参数化测试，测试 np.multiply 函数的不同输入组合
@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize("other", [2, [2, 1, 3, 4, 1, 3]])
@pytest.mark.parametrize(
    "other_dtype",
    [
        None,
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "short",
        "int",
        "intp",
        "long",
        "longlong",
        "ushort",
        "uint",
        "uintp",
        "ulong",
        "ulonglong",
    ],
)
# 定义测试函数 test_ufunc_multiply，测试 NumPy 通用函数的乘法运算是否与 Python 内置行为一致
def test_ufunc_multiply(dtype, string_list, other, other_dtype, use_out):
    """Test the two-argument ufuncs match python builtin behavior."""
    # 根据指定的数据类型创建 NumPy 数组，内容为字符串列表
    arr = np.array(string_list, dtype=dtype)
    # 如果指定了 other_dtype，将其转换为 NumPy 的数据类型对象
    if other_dtype is not None:
        other_dtype = np.dtype(other_dtype)
    try:
        # 尝试获取 other 的长度，若成功，说明 other 是一个序列
        len(other)
        # 使用列表推导式计算字符串列表和 other 序列对应位置的乘积结果
        result = [s * o for s, o in zip(string_list, other)]
        # 将 other 转换为 NumPy 数组
        other = np.array(other)
        # 如果指定了 other_dtype，将 other 转换为该数据类型
        if other_dtype is not None:
            other = other.astype(other_dtype)
    except TypeError:
        # 如果 other 不是序列，而是一个单一值，处理异常情况
        if other_dtype is not None:
            other = other_dtype.type(other)
        result = [s * other for s in string_list]

    # 如果 use_out 为 True，测试使用 np.multiply 函数的 out 参数
    if use_out:
        # 备份原始数组 arr
        arr_cache = arr.copy()
        # 调用 np.multiply 进行乘法运算，结果存放在 arr 中
        lres = np.multiply(arr, other, out=arr)
        # 断言 lres 和预期结果 result 相等
        assert_array_equal(lres, result)
        # 恢复 arr 到原始值
        arr[:] = arr_cache
        # 断言 lres 和 arr 是同一个对象
        assert lres is arr
        # 使用原地操作符 *= 进行乘法运算
        arr *= other
        # 断言 arr 和预期结果 result 相等
        assert_array_equal(arr, result)
        # 恢复 arr 到原始值
        arr[:] = arr_cache
        # 使用 np.multiply 进行反向乘法运算，结果存放在 arr 中
        rres = np.multiply(other, arr, out=arr)
        # 断言 rres 和预期结果 result 相等
        assert rres is arr
        assert_array_equal(rres, result)
    else:
        # 如果 use_out 为 False，直接使用 * 操作符进行乘法运算
        lres = arr * other
        # 断言 lres 和预期结果 result 相等
        assert_array_equal(lres, result)
        # 反向乘法运算
        rres = other * arr
        # 断言 rres 和预期结果 result 相等
        assert_array_equal(rres, result)

    # 如果 dtype 具有属性 "na_object"，执行下列逻辑
    if not hasattr(dtype, "na_object"):
        return

    # 检查 dtype.na_object 是否为 NaN
    is_nan = np.isnan(np.array([dtype.na_object], dtype=dtype))[0]
    # 检查 dtype.na_object 是否为字符串
    is_str = isinstance(dtype.na_object, str)
    bool_errors = 0
    try:
        # 尝试将 dtype.na_object 转换为布尔值
        bool(dtype.na_object)
    except TypeError:
        # 捕获 TypeError 异常
        bool_errors = 1

    # 将字符串列表与 dtype.na_object 合并为新数组 arr
    arr = np.array(string_list + [dtype.na_object], dtype=dtype)

    try:
        # 尝试获取 other 的长度，若成功，说明 other 是一个序列
        len(other)
        # 向 other 中追加值 3
        other = np.append(other, 3)
        # 如果指定了 other_dtype，将 other 转换为该数据类型
        if other_dtype is not None:
            other = other.astype(other_dtype)
    except TypeError:
        pass

    # 如果 dtype.na_object 是 NaN 或者存在 bool_errors 或者是字符串
    if is_nan or bool_errors or is_str:
        # 对于每个 res 在 [arr * other, other * arr] 中
        for res in [arr * other, other * arr]:
            # 断言 res 的前面部分与预期结果 result 相等
            assert_array_equal(res[:-1], result)
            # 如果不是字符串类型，断言 res 的最后一个元素是 dtype.na_object
            if not is_str:
                assert res[-1] is dtype.na_object
            else:
                try:
                    # 尝试比较 res 的最后一个元素与 dtype.na_object * other[-1]
                    assert res[-1] == dtype.na_object * other[-1]
                except (IndexError, TypeError):
                    # 捕获 IndexError 或 TypeError 异常，比较 res 的最后一个元素与 dtype.na_object * other
                    assert res[-1] == dtype.na_object * other
    else:
        # 如果以上条件不满足，预期会引发 TypeError 异常
        with pytest.raises(TypeError):
            arr * other
        with pytest.raises(TypeError):
            other * arr


# 定义一个包含 datetime 输入的列表
DATETIME_INPUT = [
    np.datetime64("1923-04-14T12:43:12"),
    # 创建一个 numpy datetime64 对象，表示 "1994-06-21T14:43:15"
    np.datetime64("1994-06-21T14:43:15"),
    
    # 创建一个 numpy datetime64 对象，表示 "2001-10-15T04:10:32"
    np.datetime64("2001-10-15T04:10:32"),
    
    # 创建一个 numpy datetime64 对象，表示 "NaT" (Not a Time，表示缺失的时间值)
    np.datetime64("NaT"),
    
    # 创建一个 numpy datetime64 对象，表示 "1995-11-25T16:02:16"
    np.datetime64("1995-11-25T16:02:16"),
    
    # 创建一个 numpy datetime64 对象，表示 "2005-01-04T03:14:12"
    np.datetime64("2005-01-04T03:14:12"),
    
    # 创建一个 numpy datetime64 对象，表示 "2041-12-03T14:05:03"
    np.datetime64("2041-12-03T14:05:03"),
]

# 定义一个时间差输入列表，包含不同的 numpy.timedelta64 对象
TIMEDELTA_INPUT = [
    np.timedelta64(12358, "s"),  # 表示12358秒的时间差
    np.timedelta64(23, "s"),     # 表示23秒的时间差
    np.timedelta64(74, "s"),     # 表示74秒的时间差
    np.timedelta64("NaT"),       # 表示不确定的时间差
    np.timedelta64(23, "s"),     # 表示23秒的时间差
    np.timedelta64(73, "s"),     # 表示73秒的时间差
    np.timedelta64(7, "s"),      # 表示7秒的时间差
]

# 使用 pytest 的 parametrize 装饰器定义测试用例参数化
@pytest.mark.parametrize(
    "input_data, input_dtype",
    [
        (DATETIME_INPUT, "M8[s]"),  # 使用日期时间输入和'M8[s]'数据类型
        (TIMEDELTA_INPUT, "m8[s]")  # 使用时间差输入和'm8[s]'数据类型
    ]
)
def test_datetime_timedelta_cast(dtype, input_data, input_dtype):
    # 根据给定的输入数据和数据类型创建 numpy 数组 a
    a = np.array(input_data, dtype=input_dtype)

    # 检查 dtype 是否具有属性 'na_object'
    has_na = hasattr(dtype, "na_object")
    # 检查 dtype 的 'na_object' 是否为字符串
    is_str = isinstance(getattr(dtype, "na_object", None), str)

    # 如果没有 'na_object' 属性或者 'na_object' 是字符串，则删除第三个元素
    if not has_na or is_str:
        a = np.delete(a, 3)

    # 将数组 a 转换为指定的 dtype 类型，保存为 sa
    sa = a.astype(dtype)
    # 将 sa 转换回原始数据类型，保存为 ra
    ra = sa.astype(a.dtype)

    # 如果有 'na_object' 属性且 'na_object' 不是字符串
    if has_na and not is_str:
        # 断言 sa 的第四个元素为 dtype 的 'na_object'
        assert sa[3] is dtype.na_object
        # 断言 ra 的第四个元素是 NaT（不确定的时间）
        assert np.isnat(ra[3])

    # 断言数组 a 和 ra 相等
    assert_array_equal(a, ra)

    # 如果有 'na_object' 属性且 'na_object' 不是字符串
    if has_na and not is_str:
        # 不必担心如何比较 NaT 是如何转换的
        sa = np.delete(sa, 3)
        a = np.delete(a, 3)

    # 如果输入数据类型以 "M" 开头
    if input_dtype.startswith("M"):
        # 断言 sa 与 a.astype("U") 相等
        assert_array_equal(sa, a.astype("U"))
    else:
        # timedelta 到 unicode 的转换会产生不可循环的字符串，我们不希望在 stringdtype 中重现这种行为
        # 断言 sa 与 a.astype("int64").astype("U") 相等
        assert_array_equal(sa, a.astype("int64").astype("U"))


def test_nat_casts():
    # 构建字符串 'nat' 的所有大小写组合
    s = 'nat'
    all_nats = itertools.product(*zip(s.upper(), s.lower()))
    all_nats = list(map(''.join, all_nats))
    NaT_dt = np.datetime64('NaT')
    NaT_td = np.timedelta64('NaT')
    for na_object in [np._NoValue, None, np.nan, 'nat', '']:
        # numpy 将空字符串和所有大小写组合的 'nat' 视为 NaT
        dtype = StringDType(na_object=na_object)
        arr = np.array([''] + all_nats, dtype=dtype)
        dt_array = arr.astype('M8[s]')
        td_array = arr.astype('m8[s]')
        # 断言 dt_array 中的元素与 NaT_dt 相等
        assert_array_equal(dt_array, NaT_dt)
        # 断言 td_array 中的元素与 NaT_td 相等
        assert_array_equal(td_array, NaT_td)

        if na_object is np._NoValue:
            output_object = 'NaT'
        else:
            output_object = na_object

        for arr in [dt_array, td_array]:
            # 断言 arr 转换为指定 dtype 后与 output_object 相等
            assert_array_equal(
                arr.astype(dtype),
                np.array([output_object]*arr.size, dtype=dtype))


def test_nat_conversion():
    # 对于 numpy.datetime64 和 numpy.timedelta64 的 'NaT'，测试是否抛出 ValueError
    for nat in [np.datetime64("NaT", "s"), np.timedelta64("NaT", "s")]:
        with pytest.raises(ValueError, match="string coercion is disabled"):
            np.array(["a", nat], dtype=StringDType(coerce=False))


def test_growing_strings(dtype):
    # 扩展字符串会导致堆分配，测试确保我们正确处理所有可能的起始情况
    data = [
        "hello",  # 一个短字符串
        "abcdefghijklmnopqestuvwxyz",  # 一个中等长度的堆分配字符串
        "hello" * 200,  # 一个长的堆分配字符串
    ]

    # 创建一个包含不同 dtype 的 numpy 数组 arr 和 uarr
    arr = np.array(data, dtype=dtype)
    uarr = np.array(data, dtype=str)

    for _ in range(5):
        # 对 arr 和 uarr 执行字符串的扩展操作
        arr = arr + arr
        uarr = uarr + uarr
    # 使用 NumPy 的 assert_array_equal 函数比较两个数组 arr 和 uarr 是否完全相等
    assert_array_equal(arr, uarr)
# 根据条件跳过测试，如果在 WebAssembly 中运行，因为 wasm 不支持线程
@pytest.mark.skipif(IS_WASM, reason="no threading support in wasm")
def test_threaded_access_and_mutation(dtype, random_string_list):
    # 这个测试使用一个随机数生成器 (RNG)，如果存在线程 bug 可能会导致崩溃或死锁
    rng = np.random.default_rng(0x4D3D3D3)

    def func(arr):
        rnd = rng.random()
        # 在数组中随机写入数据、执行 ufunc，或者重新初始化数组
        if rnd < 0.25:
            num = np.random.randint(0, arr.size)
            arr[num] = arr[num] + "hello"
        elif rnd < 0.5:
            if rnd < 0.375:
                np.add(arr, arr)
            else:
                np.add(arr, arr, out=arr)
        elif rnd < 0.75:
            if rnd < 0.875:
                np.multiply(arr, np.int64(2))
            else:
                np.multiply(arr, np.int64(2), out=arr)
        else:
            arr[:] = random_string_list

    # 使用 ThreadPoolExecutor 创建最多 8 个工作线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        # 创建一个包含随机字符串的 NumPy 数组
        arr = np.array(random_string_list, dtype=dtype)
        # 提交函数 func 的多个任务到线程池
        futures = [tpe.submit(func, arr) for _ in range(500)]

        # 等待所有任务完成
        for f in futures:
            f.result()


# 一组用于测试的字符串数据
UFUNC_TEST_DATA = [
    "hello" * 10,
    "Ae¢☃€ 😊" * 20,
    "entry\nwith\nnewlines",
    "entry\twith\ttabs",
]


# 为字符串数组创建的测试 fixture
@pytest.fixture
def string_array(dtype):
    return np.array(UFUNC_TEST_DATA, dtype=dtype)


# 为 Unicode 字符串数组创建的测试 fixture
@pytest.fixture
def unicode_array():
    return np.array(UFUNC_TEST_DATA, dtype=np.str_)


# 一组保留 NaN 的字符串函数
NAN_PRESERVING_FUNCTIONS = [
    "capitalize",
    "expandtabs",
    "lower",
    "lstrip",
    "rstrip",
    "splitlines",
    "strip",
    "swapcase",
    "title",
    "upper",
]


# 返回布尔值输出的函数
BOOL_OUTPUT_FUNCTIONS = [
    "isalnum",
    "isalpha",
    "isdigit",
    "islower",
    "isspace",
    "istitle",
    "isupper",
    "isnumeric",
    "isdecimal",
]


# 返回单目函数的函数名列表
UNARY_FUNCTIONS = [
    "str_len",
    "capitalize",
    "expandtabs",
    "isalnum",
    "isalpha",
    "isdigit",
    "islower",
    "isspace",
    "istitle",
    "isupper",
    "lower",
    "lstrip",
    "rstrip",
    "splitlines",
    "strip",
    "swapcase",
    "title",
    "upper",
    "isnumeric",
    "isdecimal",
    "isalnum",
    "islower",
    "istitle",
    "isupper",
]


# 未实现的向量化字符串函数列表
UNIMPLEMENTED_VEC_STRING_FUNCTIONS = [
    "capitalize",
    "expandtabs",
    "lower",
    "splitlines",
    "swapcase",
    "title",
    "upper",
]


# 仅在 np.char 中存在的函数列表
ONLY_IN_NP_CHAR = [
    "join",
    "split",
    "rsplit",
    "splitlines"
]


# 参数化测试，测试单目函数
@pytest.mark.parametrize("function_name", UNARY_FUNCTIONS)
def test_unary(string_array, unicode_array, function_name):
    if function_name in ONLY_IN_NP_CHAR:
        func = getattr(np.char, function_name)
    else:
        func = getattr(np.strings, function_name)
    dtype = string_array.dtype
    # 对 string_array 和 unicode_array 应用指定的函数
    sres = func(string_array)
    ures = func(unicode_array)
    # 如果 sres 的 dtype 是 StringDType，则将 ures 转换为 StringDType
    if sres.dtype == StringDType():
        ures = ures.astype(StringDType())
    # 断言两个结果数组是否相等
    assert_array_equal(sres, ures)

    # 如果 dtype 没有 "na_object" 属性，则直接返回
    if not hasattr(dtype, "na_object"):
        return
    # 检查 dtype.na_object 是否为 NaN
    is_nan = np.isnan(np.array([dtype.na_object], dtype=dtype))[0]
    # 检查 dtype.na_object 是否为字符串类型
    is_str = isinstance(dtype.na_object, str)
    # 在 string_array 的开头插入 dtype.na_object 构成新的数组 na_arr
    na_arr = np.insert(string_array, 0, dtype.na_object)

    # 如果 function_name 在 UNIMPLEMENTED_VEC_STRING_FUNCTIONS 中
    if function_name in UNIMPLEMENTED_VEC_STRING_FUNCTIONS:
        if not is_str:
            # 为了避免这些错误，需要在 _vec_string 中添加 NA 支持
            # 检查 func(na_arr) 是否抛出 ValueError 或 TypeError 异常
            with pytest.raises((ValueError, TypeError)):
                func(na_arr)
        else:
            if function_name == "splitlines":
                # 断言 func(na_arr) 的第一个元素等于 func(dtype.na_object)[()]
                assert func(na_arr)[0] == func(dtype.na_object)[()]
            else:
                # 断言 func(na_arr) 的第一个元素等于 func(dtype.na_object)
                assert func(na_arr)[0] == func(dtype.na_object)
        return

    # 如果 function_name 是 "str_len" 并且 dtype.na_object 不是字符串
    if function_name == "str_len" and not is_str:
        # str_len 对于任何非字符串的 null 均会抛出 ValueError 异常，因为其结果为整数
        with pytest.raises(ValueError):
            func(na_arr)
        return

    # 如果 function_name 在 BOOL_OUTPUT_FUNCTIONS 中
    if function_name in BOOL_OUTPUT_FUNCTIONS:
        # 如果 dtype.na_object 是 NaN
        if is_nan:
            # 断言 func(na_arr) 的第一个元素是 np.False_
            assert func(na_arr)[0] is np.False_
        elif is_str:
            # 断言 func(na_arr) 的第一个元素等于 func(dtype.na_object)
            assert func(na_arr)[0] == func(dtype.na_object)
        else:
            # 检查 func(na_arr) 是否抛出 ValueError 异常
            with pytest.raises(ValueError):
                func(na_arr)
        return

    # 如果 dtype.na_object 不是 NaN 且不是字符串
    if not (is_nan or is_str):
        # 检查 func(na_arr) 是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            func(na_arr)
        return

    # 计算 func(na_arr) 的结果
    res = func(na_arr)
    # 如果 dtype.na_object 是 NaN 并且 function_name 在 NAN_PRESERVING_FUNCTIONS 中
    if is_nan and function_name in NAN_PRESERVING_FUNCTIONS:
        # 断言 res 的第一个元素是 dtype.na_object
        assert res[0] is dtype.na_object
    elif is_str:
        # 断言 res 的第一个元素等于 func(dtype.na_object)
        assert res[0] == func(dtype.na_object)
# Mark the test as expected to fail with a specific reason if a Unicode bug occurs, and fail strictly
unicode_bug_fail = pytest.mark.xfail(
    reason="unicode output width is buggy", strict=True
)

# Define a list of binary functions with their corresponding arguments
BINARY_FUNCTIONS = [
    ("add", (None, None)),
    ("multiply", (None, 2)),
    ("mod", ("format: %s", None)),
    ("center", (None, 25)),
    ("count", (None, "A")),
    ("encode", (None, "UTF-8")),
    ("endswith", (None, "lo")),
    ("find", (None, "A")),
    ("index", (None, "e")),
    ("join", ("-", None)),
    ("ljust", (None, 12)),
    ("lstrip", (None, "A")),
    ("partition", (None, "A")),
    ("replace", (None, "A", "B")),
    ("rfind", (None, "A")),
    ("rindex", (None, "e")),
    ("rjust", (None, 12)),
    ("rsplit", (None, "A")),
    ("rstrip", (None, "A")),
    ("rpartition", (None, "A")),
    ("split", (None, "A")),
    ("strip", (None, "A")),
    ("startswith", (None, "A")),
    ("zfill", (None, 12)),
]

# List of functions that pass through NaN or null values
PASSES_THROUGH_NAN_NULLS = [
    "add",
    "center",
    "ljust",
    "multiply",
    "replace",
    "rjust",
    "strip",
    "lstrip",
    "rstrip",
    "replace",  # Missing comma added
    "zfill",
]

# List of functions where null values are considered falsy
NULLS_ARE_FALSEY = [
    "startswith",
    "endswith",
]

# List of functions where null values always raise an error
NULLS_ALWAYS_ERROR = [
    "count",
    "find",
    "rfind",
]

# Combine lists to indicate which functions support null values in arguments
SUPPORTS_NULLS = (
    PASSES_THROUGH_NAN_NULLS +
    NULLS_ARE_FALSEY +
    NULLS_ALWAYS_ERROR
)

# Function to call a given function with specific arguments and handle null values
def call_func(func, args, array, sanitize=True):
    if args == (None, None):
        return func(array, array)
    if args[0] is None:
        if sanitize:
            # Sanitize arguments by converting them to NumPy arrays if they are strings
            san_args = tuple(
                np.array(arg, dtype=array.dtype) if isinstance(arg, str) else
                arg for arg in args[1:]
            )
        else:
            san_args = args[1:]
        return func(array, *san_args)
    if args[1] is None:
        return func(args[0], array)
    # Assertion for a condition that shouldn't happen
    assert 0

# Test function parameterized with binary functions to validate behavior across string and unicode arrays
@pytest.mark.parametrize("function_name, args", BINARY_FUNCTIONS)
def test_binary(string_array, unicode_array, function_name, args):
    if function_name in ONLY_IN_NP_CHAR:
        # Get function from np.char if it exists there, otherwise from np.strings
        func = getattr(np.char, function_name)
    else:
        func = getattr(np.strings, function_name)
    
    # Call function for string arrays and unicode arrays, potentially converting types
    sres = call_func(func, args, string_array)
    ures = call_func(func, args, unicode_array, sanitize=False)
    
    # Convert unicode result to StringDType if necessary
    if not isinstance(sres, tuple) and sres.dtype == StringDType():
        ures = ures.astype(StringDType())
    
    # Assert that results from string and unicode arrays are equal
    assert_array_equal(sres, ures)

    dtype = string_array.dtype
    # Check if the function supports null values, and if the dtype supports NA objects
    if function_name not in SUPPORTS_NULLS or not hasattr(dtype, "na_object"):
        return

    # Insert NA object into string array and check its properties
    na_arr = np.insert(string_array, 0, dtype.na_object)
    is_nan = np.isnan(np.array([dtype.na_object], dtype=dtype))[0]
    is_str = isinstance(dtype.na_object, str)
    should_error = not (is_nan or is_str)

    # Check conditions under which null values should raise errors
    if (
        (function_name in NULLS_ALWAYS_ERROR and not is_str)
        or (function_name in PASSES_THROUGH_NAN_NULLS and should_error)
        or (function_name in NULLS_ARE_FALSEY and should_error)
    ):
        # Ensure calling the function with NA array raises ValueError or TypeError
        with pytest.raises((ValueError, TypeError)):
            call_func(func, args, na_arr)
        return
    # 调用指定函数 `func`，传入参数 `args` 和 `na_arr`，并获取返回结果 `res`
    res = call_func(func, args, na_arr)

    # 如果 `is_str` 为真，则进行断言检查，验证第一个返回结果与调用仅针对 `na_arr[:1]` 的函数结果是否相同
    if is_str:
        assert res[0] == call_func(func, args, na_arr[:1])
    # 如果函数名 `function_name` 存在于 `NULLS_ARE_FALSEY` 中，则断言第一个返回结果为 `np.False_`
    elif function_name in NULLS_ARE_FALSEY:
        assert res[0] is np.False_
    # 如果函数名 `function_name` 存在于 `PASSES_THROUGH_NAN_NULLS` 中，则断言第一个返回结果为 `dtype.na_object`
    elif function_name in PASSES_THROUGH_NAN_NULLS:
        assert res[0] is dtype.na_object
    else:
        # 如果执行到这里，应该是不应该发生的情况
        assert 0
# 使用 pytest 框架的 mark.parametrize 装饰器，定义了一个参数化测试函数，测试 np.strings.find 和 np.strings.startswith 函数
# 分别对输入数组进行查找指定子字符串的操作，并验证预期的返回结果
@pytest.mark.parametrize("function, expected", [
    (np.strings.find, [[2, -1], [1, -1]]),
    (np.strings.startswith, [[False, False], [True, False]])])
# 参数化测试函数的另一组参数，测试不同的起始和结束位置的输入值
@pytest.mark.parametrize("start, stop", [
    (1, 4),                                     # 整数起始和结束位置
    (np.int8(1), np.int8(4)),                   # np.int8 类型的起始和结束位置
    (np.array([1, 1], dtype='u2'),              # 无符号 2 字节整数数组的起始位置
     np.array([4, 4], dtype='u2'))])            # 无符号 2 字节整数数组的结束位置
def test_non_default_start_stop(function, start, stop, expected):
    # 创建一个 2x2 的 numpy 数组，包含字符串数据，使用 'T' 表示对数组进行转置
    a = np.array([["--🐍--", "--🦜--"],
                  ["-🐍---", "-🦜---"]], "T")
    # 调用给定的字符串处理函数 function，在数组 a 中查找或者以指定字符串开始的位置索引
    indx = function(a, "🐍", start, stop)
    # 验证函数返回的索引数组与预期结果 expected 是否相等
    assert_array_equal(indx, expected)


# 参数化测试函数，测试替换字符串操作的非默认重复次数
@pytest.mark.parametrize("count", [2, np.int8(2), np.array([2, 2], 'u2')])
def test_replace_non_default_repeat(count):
    # 创建一个包含字符串数据的 numpy 数组，并使用 'T' 表示对数组进行转置
    a = np.array(["🐍--", "🦜-🦜-"], "T")
    # 调用 np.strings.replace 函数，将数组 a 中的指定子字符串替换为指定字符串，限制替换次数为 count
    result = np.strings.replace(a, "🦜-", "🦜†", count)
    # 验证替换操作后的结果数组与预期结果是否相等
    assert_array_equal(result, np.array(["🐍--", "🦜†🦜†"], "T"))


# 测试函数，验证 np.char.rjust、np.char.ljust 和相关函数的一致性
def test_strip_ljust_rjust_consistency(string_array, unicode_array):
    # 对字符串数组和 Unicode 数组分别进行右对齐操作，填充字符使其总长度为 1000
    rjs = np.char.rjust(string_array, 1000)
    rju = np.char.rjust(unicode_array, 1000)

    # 对字符串数组和 Unicode 数组分别进行左对齐操作，填充字符使其总长度为 1000
    ljs = np.char.ljust(string_array, 1000)
    lju = np.char.ljust(unicode_array, 1000)

    # 验证右对齐后去除左侧空白字符的结果数组是否相等，并将结果强制转换为 StringDType 类型
    assert_array_equal(
        np.char.lstrip(rjs),
        np.char.lstrip(rju).astype(StringDType()),
    )

    # 验证左对齐后去除右侧空白字符的结果数组是否相等，并将结果强制转换为 StringDType 类型
    assert_array_equal(
        np.char.rstrip(ljs),
        np.char.rstrip(lju).astype(StringDType()),
    )

    # 验证左右两侧去除空白字符的结果数组是否相等，并将结果强制转换为 StringDType 类型
    assert_array_equal(
        np.char.strip(ljs),
        np.char.strip(lju).astype(StringDType()),
    )

    # 验证右对齐后去除左右两侧空白字符的结果数组是否相等，并将结果强制转换为 StringDType 类型
    assert_array_equal(
        np.char.strip(rjs),
        np.char.strip(rju).astype(StringDType()),
    )


# 测试函数，验证未设置 NA 对象时的类型转换行为
def test_unset_na_coercion():
    # 使用未设置 NA 对象的 StringDType 创建数组 arr，包含字符串数据 "hello" 和 "world"
    inp = ["hello", "world"]
    arr = np.array(inp, dtype=StringDType(na_object=None))

    # 遍历不同的操作 dtype，进行字符串连接操作，验证结果数组是否符合预期
    for op_dtype in [None, StringDType(), StringDType(coerce=False),
                     StringDType(na_object=None)]:
        if op_dtype is None:
            op = "2"
        else:
            op = np.array("2", dtype=op_dtype)
        res = arr + op
        assert_array_equal(res, ["hello2", "world2"])

    # 使用设置了不同 NA 对象的 StringDType 进行字符串连接操作，验证是否引发 TypeError 异常
    for op_dtype in [StringDType(na_object=pd_NA), StringDType(na_object="")]:
        op = np.array("2", dtype=op_dtype)
        with pytest.raises(TypeError):
            arr + op

    # 使用不同的操作 dtype，比较数组 arr 和输入数组的内容是否相等
    for op_dtype in [None, StringDType(), StringDType(coerce=True),
                     StringDType(na_object=None)]:
        if op_dtype is None:
            op = inp
        else:
            op = np.array(inp, dtype=op_dtype)
        assert_array_equal(arr, op)
    # 循环遍历列表中的每个数据类型对象
    for op_dtype in [StringDType(na_object=pd_NA),
                     StringDType(na_object=np.nan)]:
        # 使用指定的数据类型对象创建一个新的 NumPy 数组
        op = np.array(inp, dtype=op_dtype)
        # 使用 pytest 的断言检查是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 检查创建的数组与 op 数组是否相等
            arr == op
class TestImplementation:
    """Check that strings are stored in the arena when possible.

    This tests implementation details, so should be adjusted if
    the implementation changes.
    """

    @classmethod
    def setup_class(self):
        # 定义常量，表示不同的状态和标志位
        self.MISSING = 0x80
        self.INITIALIZED = 0x40
        self.OUTSIDE_ARENA = 0x20
        self.LONG = 0x10
        # 创建一个 StringDType 类型的对象，na_object 使用 NaN 表示空缺值
        self.dtype = StringDType(na_object=np.nan)
        # 计算字符串的字节大小
        self.sizeofstr = self.dtype.itemsize
        # 指针大小为 sizeof(size_t)，在这里 sp 是 sizeofstr 的一半
        sp = self.dtype.itemsize // 2  # pointer size = sizeof(size_t)
        
        # 定义一个视图的数据类型 view_dtype
        # 在小端字节序系统中定义不同字段顺序的数据类型描述
        self.view_dtype = np.dtype([
            ('offset', f'u{sp}'),
            ('size', f'u{sp // 2}'),
            ('xsiz', f'V{sp // 2 - 1}'),
            ('size_and_flags', 'u1'),
        ] if sys.byteorder == 'little' else [
            ('size_and_flags', 'u1'),
            ('xsiz', f'V{sp // 2 - 1}'),
            ('size', f'u{sp // 2}'),
            ('offset', f'u{sp}'),
        ])
        
        # 初始化不同长度的字符串实例
        self.s_empty = ""
        self.s_short = "01234"
        self.s_medium = "abcdefghijklmnopqrstuvwxyz"
        self.s_long = "-=+" * 100
        
        # 创建一个 NumPy 数组 a，其中包含不同长度的字符串，使用 self.dtype 类型
        self.a = np.array(
            [self.s_empty, self.s_short, self.s_medium, self.s_long],
            self.dtype)

    def get_view(self, a):
        # 不能直接将 StringDType 视为其他类型，因为它具有引用。因此，使用一个 stride trick 的 hack。
        from numpy.lib._stride_tricks_impl import DummyArray
        # 复制 a 的数组接口信息，并使用 view_dtype 的描述符
        interface = dict(a.__array_interface__)
        interface['descr'] = self.view_dtype.descr
        interface['typestr'] = self.view_dtype.str
        # 返回使用 DummyArray 包装后的 ndarray
        return np.asarray(DummyArray(interface, base=a))

    def get_flags(self, a):
        # 获取视图的 size_and_flags 字段，并返回高 4 位的值
        return self.get_view(a)['size_and_flags'] & 0xf0

    def is_short(self, a):
        # 检查字符串是否短的辅助方法
        return self.get_flags(a) == self.INITIALIZED | self.OUTSIDE_ARENA

    def is_on_heap(self, a):
        # 检查字符串是否长字符串的辅助方法
        return self.get_flags(a) == (self.INITIALIZED
                                     | self.OUTSIDE_ARENA
                                     | self.LONG)

    def is_missing(self, a):
        # 检查字符串是否为缺失值的辅助方法
        return self.get_flags(a) & self.MISSING == self.MISSING

    def in_arena(self, a):
        # 检查字符串是否在 arena 内的辅助方法
        return (self.get_flags(a) & (self.INITIALIZED | self.OUTSIDE_ARENA)
                == self.INITIALIZED)
    # 定义测试设置的方法
    def test_setup(self):
        # 判断是否为短字符串
        is_short = self.is_short(self.a)
        # 计算字符串数组中每个字符串的长度
        length = np.strings.str_len(self.a)
        # 断言检查：确保 is_short 的结果符合预期（长度在0到15之间）
        assert_array_equal(is_short, (length > 0) & (length <= 15))
        # 断言检查：检查字符串数组中是否存在在Arena中的值
        assert_array_equal(self.in_arena(self.a), [False, False, True, True])
        # 断言检查：检查字符串数组中的值是否不在堆上
        assert_array_equal(self.is_on_heap(self.a), False)
        # 断言检查：检查字符串数组中的值是否不缺失
        assert_array_equal(self.is_missing(self.a), False)
        # 获取视图对象
        view = self.get_view(self.a)
        # 根据字符串是否为短字符串，选择合适的大小字段来构建sizes数组
        sizes = np.where(is_short, view['size_and_flags'] & 0xf,
                         view['size'])
        # 断言检查：确保sizes数组与字符串数组的长度一致
        assert_array_equal(sizes, np.strings.str_len(self.a))
        # 断言检查：检查xsiz字段是否正确设置为零填充
        assert_array_equal(view['xsiz'][2:],
                           np.void(b'\x00' * (self.sizeofstr // 4 - 1)))
        # 断言检查：检查中等长度字符串在Arena中长度的表现（1字节或8字节）
        offsets = view['offset']
        assert offsets[2] == 1
        assert offsets[3] == 1 + len(self.s_medium) + self.sizeofstr // 2

    # 定义测试空字符串的方法
    def test_empty(self):
        # 创建一个空的数组e，dtype由类的属性决定
        e = np.empty((3,), self.dtype)
        # 断言检查：确保get_flags函数返回0
        assert_array_equal(self.get_flags(e), 0)
        # 断言检查：确保数组e中的值是空字符串
        assert_array_equal(e, "")

    # 定义测试全零字符串的方法
    def test_zeros(self):
        # 创建一个全零数组z，dtype由类的属性决定
        z = np.zeros((2,), self.dtype)
        # 断言检查：确保get_flags函数返回0
        assert_array_equal(self.get_flags(z), 0)
        # 断言检查：确保数组z中的值是空字符串
        assert_array_equal(z, "")

    # 定义测试复制数组的方法
    def test_copy(self):
        # 复制字符串数组a，得到数组c
        c = self.a.copy()
        # 断言检查：确保复制后的数组c与原数组a的标志相同
        assert_array_equal(self.get_flags(c), self.get_flags(self.a))
        # 断言检查：确保数组c与原数组a相等
        assert_array_equal(c, self.a)
        # 获取复制后数组c的视图对象，检查offset字段的设置
        offsets = self.get_view(c)['offset']
        assert offsets[2] == 1
        assert offsets[3] == 1 + len(self.s_medium) + self.sizeofstr // 2

    # 定义测试Arena使用和设置的方法
    def test_arena_use_with_setting(self):
        # 创建一个与数组a形状相同的全零数组c
        c = np.zeros_like(self.a)
        # 断言检查：确保get_flags函数返回0
        assert_array_equal(self.get_flags(c), 0)
        # 将数组a的值复制到数组c中
        c[:] = self.a
        # 断言检查：确保数组c的标志与数组a的标志相同
        assert_array_equal(self.get_flags(c), self.get_flags(self.a))
        # 断言检查：确保数组c与数组a相等
        assert_array_equal(c, self.a)

    # 定义测试Arena重用和设置的方法
    def test_arena_reuse_with_setting(self):
        # 复制字符串数组a，得到数组c
        c = self.a.copy()
        # 将数组a的值复制到数组c中
        c[:] = self.a
        # 断言检查：确保数组c的标志与数组a的标志相同
        assert_array_equal(self.get_flags(c), self.get_flags(self.a))
        # 断言检查：确保数组c与数组a相等
        assert_array_equal(c, self.a)

    # 定义测试在缺失后重用Arena的方法
    def test_arena_reuse_after_missing(self):
        # 复制字符串数组a，得到数组c
        c = self.a.copy()
        # 将数组c中的所有值设置为NaN
        c[:] = np.nan
        # 断言检查：确保数组c中的所有值都是缺失的
        assert np.all(self.is_missing(c))
        # 将原始字符串数组a的值重新放回数组c
        c[:] = self.a
        # 断言检查：确保数组c的标志与数组a的标志相同
        assert_array_equal(self.get_flags(c), self.get_flags(self.a))
        # 断言检查：确保数组c与数组a相等
        assert_array_equal(c, self.a)

    # 定义测试在清空后重用Arena的方法
    def test_arena_reuse_after_empty(self):
        # 复制字符串数组a，得到数组c
        c = self.a.copy()
        # 将数组c中的所有值设置为空字符串
        c[:] = ""
        # 断言检查：确保数组c中的所有值都是空字符串
        assert_array_equal(c, "")
        # 将原始字符串数组a的值重新放回数组c
        c[:] = self.a
        # 断言检查：确保数组c的标志与数组a的标志相同
        assert_array_equal(self.get_flags(c), self.get_flags(self.a))
        # 断言检查：确保数组c与数组a相等
        assert_array_equal(c, self.a)
    def test_arena_reuse_for_shorter(self):
        c = self.a.copy()
        # A string slightly shorter than the shortest in the arena
        # should be used for all strings in the arena.
        c[:] = self.s_medium[:-1]
        assert_array_equal(c, self.s_medium[:-1])
        # first empty string in original was never initialized, so
        # filling it in now leaves it initialized inside the arena.
        # second string started as a short string so it can never live
        # in the arena.
        in_arena = np.array([True, False, True, True])
        assert_array_equal(self.in_arena(c), in_arena)
        # But when a short string is replaced, it will go on the heap.
        assert_array_equal(self.is_short(c), False)
        assert_array_equal(self.is_on_heap(c), ~in_arena)
        # We can put the originals back, and they'll still fit,
        # and short strings are back as short strings
        c[:] = self.a
        assert_array_equal(c, self.a)
        assert_array_equal(self.in_arena(c), in_arena)
        assert_array_equal(self.is_short(c), self.is_short(self.a))
        assert_array_equal(self.is_on_heap(c), False)

    def test_arena_reuse_if_possible(self):
        c = self.a.copy()
        # A slightly longer string will not fit in the arena for
        # the medium string, but will fit for the longer one.
        c[:] = self.s_medium + "±"
        assert_array_equal(c, self.s_medium + "±")
        in_arena_exp = np.strings.str_len(self.a) >= len(self.s_medium) + 1
        # first entry started uninitialized and empty, so filling it leaves
        # it in the arena
        in_arena_exp[0] = True
        assert not np.all(in_arena_exp == self.in_arena(self.a))
        assert_array_equal(self.in_arena(c), in_arena_exp)
        assert_array_equal(self.is_short(c), False)
        assert_array_equal(self.is_on_heap(c), ~in_arena_exp)
        # And once outside arena, it stays outside, since offset is lost.
        # But short strings are used again.
        c[:] = self.a
        is_short_exp = self.is_short(self.a)
        assert_array_equal(c, self.a)
        assert_array_equal(self.in_arena(c), in_arena_exp)
        assert_array_equal(self.is_short(c), is_short_exp)
        assert_array_equal(self.is_on_heap(c), ~in_arena_exp & ~is_short_exp)

    def test_arena_no_reuse_after_short(self):
        c = self.a.copy()
        # If we replace a string with a short string, it cannot
        # go into the arena after because the offset is lost.
        c[:] = self.s_short
        assert_array_equal(c, self.s_short)
        assert_array_equal(self.in_arena(c), False)
        c[:] = self.a
        assert_array_equal(c, self.a)
        assert_array_equal(self.in_arena(c), False)
        assert_array_equal(self.is_on_heap(c), self.in_arena(self.a))



        # 根据长度不同测试字符串是否能够复用内存空间
        def test_arena_reuse_for_shorter(self):
            c = self.a.copy()
            # 将稍短于竞技场中最短字符串的字符串用于竞技场中所有字符串。
            c[:] = self.s_medium[:-1]
            assert_array_equal(c, self.s_medium[:-1])
            # 原始数据中的第一个空字符串从未初始化，因此现在填充后仍在竞技场中初始化。
            # 第二个字符串起初作为短字符串，因此永远不能存在于竞技场中。
            in_arena = np.array([True, False, True, True])
            assert_array_equal(self.in_arena(c), in_arena)
            # 但是当一个短字符串被替换时，它将存储在堆中。
            assert_array_equal(self.is_short(c), False)
            assert_array_equal(self.is_on_heap(c), ~in_arena)
            # 我们可以把原始数据放回去，它们仍然适合，
            # 短字符串再次成为短字符串
            c[:] = self.a
            assert_array_equal(c, self.a)
            assert_array_equal(self.in_arena(c), in_arena)
            assert_array_equal(self.is_short(c), self.is_short(self.a))
            assert_array_equal(self.is_on_heap(c), False)

        # 如果可能，测试竞技场重用
        def test_arena_reuse_if_possible(self):
            c = self.a.copy()
            # 稍长的字符串将不适合竞技场中的中等字符串，
            # 但适合更长的字符串。
            c[:] = self.s_medium + "±"
            assert_array_equal(c, self.s_medium + "±")
            in_arena_exp = np.strings.str_len(self.a) >= len(self.s_medium) + 1
            # 第一个条目起始未初始化和空，因此填充后保留在竞技场中
            in_arena_exp[0] = True
            assert not np.all(in_arena_exp == self.in_arena(self.a))
            assert_array_equal(self.in_arena(c), in_arena_exp)
            assert_array_equal(self.is_short(c), False)
            assert_array_equal(self.is_on_heap(c), ~in_arena_exp)
            # 一旦离开竞技场，由于偏移丢失，它将保持在外面。
            # 但短字符串会再次被使用。
            c[:] = self.a
            is_short_exp = self.is_short(self.a)
            assert_array_equal(c, self.a)
            assert_array_equal(self.in_arena(c), in_arena_exp)
            assert_array_equal(self.is_short(c), is_short_exp)
            assert_array_equal(self.is_on_heap(c), ~in_arena_exp & ~is_short_exp)

        # 测试短字符串后不再重用竞技场
        def test_arena_no_reuse_after_short(self):
            c = self.a.copy()
            # 如果我们用短字符串替换字符串，那么它不能
            # 在之后进入竞技场，因为偏移丢失了。
            c[:] = self.s_short
            assert_array_equal(c, self.s_short)
            assert_array_equal(self.in_arena(c), False)
            c[:] = self.a
            assert_array_equal(c, self.a)
            assert_array_equal(self.in_arena(c), False)
            assert_array_equal(self.is_on_heap(c), self.in_arena(self.a))
```