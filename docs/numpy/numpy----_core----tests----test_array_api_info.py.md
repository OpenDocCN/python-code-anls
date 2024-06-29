# `.\numpy\numpy\_core\tests\test_array_api_info.py`

```py
# 导入 numpy 库并导入 pytest 库
import numpy as np
import pytest

# 调用 numpy 的 __array_namespace_info__() 方法获取信息并赋值给变量 info
info = np.__array_namespace_info__()


# 定义测试函数 test_capabilities，用于测试 capabilities 方法
def test_capabilities():
    # 调用 capabilities 方法获取 capabilities 字典
    caps = info.capabilities()
    # 断言 boolean indexing 功能为 True
    assert caps["boolean indexing"] == True
    # 断言 data-dependent shapes 功能为 True
    assert caps["data-dependent shapes"] == True

    # 下面这段代码将在 2024.12 版本的数组 API 标准中添加
    # assert caps["max rank"] == 64
    # np.zeros((1,)*64)
    # 使用 pytest 检测 ValueError 异常是否被触发
    # with pytest.raises(ValueError):
    #     np.zeros((1,)*65)


# 定义测试函数 test_default_device，用于测试 default_device 方法
def test_default_device():
    # 断言 default_device 方法返回值为 "cpu"，并且与 np.asarray(0).device 相等
    assert info.default_device() == "cpu" == np.asarray(0).device


# 定义测试函数 test_default_dtypes，用于测试 default_dtypes 方法
def test_default_dtypes():
    # 调用 default_dtypes 方法获取 default_dtypes 字典
    dtypes = info.default_dtypes()
    # 断言 real floating 数据类型为 np.float64，并且 np.asarray(0.0).dtype 与之相等
    assert dtypes["real floating"] == np.float64 == np.asarray(0.0).dtype
    # 断言 complex floating 数据类型为 np.complex128，并且 np.asarray(0.0j).dtype 与之相等
    assert dtypes["complex floating"] == np.complex128 == np.asarray(0.0j).dtype
    # 断言 integral 数据类型为 np.intp，并且 np.asarray(0).dtype 与之相等
    assert dtypes["integral"] == np.intp == np.asarray(0).dtype
    # 断言 indexing 数据类型为 np.intp，并且 np.argmax(np.zeros(10)).dtype 与之相等
    assert dtypes["indexing"] == np.intp == np.argmax(np.zeros(10)).dtype

    # 使用 pytest 检测 ValueError 异常是否被触发，并且匹配 "Device not understood" 字符串
    with pytest.raises(ValueError, match="Device not understood"):
        info.default_dtypes(device="gpu")


# 定义测试函数 test_dtypes_all，用于测试 dtypes 方法
def test_dtypes_all():
    # 调用 dtypes 方法获取 dtypes 字典
    dtypes = info.dtypes()
    # 断言 dtypes 字典是否符合预期的数据类型映射关系
    assert dtypes == {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }


# 定义 dtype_categories 字典，用于指定不同数据类型类别及其对应的映射关系
dtype_categories = {
    "bool": {"bool": np.bool_},
    "signed integer": {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    },
    "unsigned integer": {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    },
    "integral": ("signed integer", "unsigned integer"),
    "real floating": {"float32": np.float32, "float64": np.float64},
    "complex floating": {"complex64": np.complex64, "complex128":
                         np.complex128},
    "numeric": ("integral", "real floating", "complex floating"),
}


# 使用 pytest 的 parametrize 装饰器进行参数化测试
@pytest.mark.parametrize("kind", dtype_categories)
def test_dtypes_kind(kind):
    # 获取预期结果
    expected = dtype_categories[kind]
    # 如果 expected 是 tuple 类型，比较两种方式获取的结果是否相同
    if isinstance(expected, tuple):
        assert info.dtypes(kind=kind) == info.dtypes(kind=expected)
    else:
        assert info.dtypes(kind=kind) == expected


# 定义测试函数 test_dtypes_tuple，用于测试 dtypes 方法，指定 kind 参数为 ("bool", "integral")
def test_dtypes_tuple():
    dtypes = info.dtypes(kind=("bool", "integral"))
    # 断言 dtypes 字典是否符合预期的数据类型映射关系
    assert dtypes == {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }


# 定义测试函数 test_dtypes_invalid_kind，用于测试 dtypes 方法，传入不支持的 kind 参数
def test_dtypes_invalid_kind():
    # 使用 pytest 检测 ValueError 异常是否被触发，并且匹配 "unsupported kind" 字符串
    with pytest.raises(ValueError, match="unsupported kind"):
        info.dtypes(kind="invalid")


# 定义测试函数 test_dtypes_invalid_device，用于测试 dtypes 方法，传入不支持的 device 参数
def test_dtypes_invalid_device():
    # 这部分代码在后续补充
    pass
    # 使用 pytest 来验证是否会抛出 ValueError 异常，并检查异常消息是否包含 "Device not understood"
    with pytest.raises(ValueError, match="Device not understood"):
        # 调用 info 对象的 dtypes 方法，并传入参数 device="gpu"
        info.dtypes(device="gpu")
# 定义一个名为 test_devices 的函数，用于测试某个功能或特性
def test_devices():
    # 断言检查调用 info.devices() 的返回值是否等于列表 ["cpu"]
    assert info.devices() == ["cpu"]
```