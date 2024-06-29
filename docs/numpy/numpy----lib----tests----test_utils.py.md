# `.\numpy\numpy\lib\tests\test_utils.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_raises_regex  # 导入 assert_raises_regex 函数
import numpy.lib._utils_impl as _utils_impl  # 导入 _utils_impl 模块

from io import StringIO  # 从 io 模块导入 StringIO 类


def test_assert_raises_regex_context_manager():
    with assert_raises_regex(ValueError, 'no deprecation warning'):  # 使用 assert_raises_regex 上下文管理器检查是否抛出 ValueError 异常，并检查错误消息是否包含特定文本
        raise ValueError('no deprecation warning')


def test_info_method_heading():
    # info(class) should only print "Methods:" heading if methods exist
    # 定义两个类来测试：NoPublicMethods 没有公共方法，WithPublicMethods 有一个公共方法
    class NoPublicMethods:
        pass

    class WithPublicMethods:
        def first_method():
            pass

    def _has_method_heading(cls):
        out = StringIO()  # 创建一个 StringIO 对象用于捕获输出
        np.info(cls, output=out)  # 使用 np.info 函数打印类信息到 StringIO 对象
        return 'Methods:' in out.getvalue()  # 检查输出字符串中是否包含 "Methods:" 标题

    assert _has_method_heading(WithPublicMethods)  # 测试 WithPublicMethods 类应该打印出 "Methods:" 标题
    assert not _has_method_heading(NoPublicMethods)  # 测试 NoPublicMethods 类不应该打印出 "Methods:" 标题


def test_drop_metadata():
    def _compare_dtypes(dt1, dt2):
        return np.can_cast(dt1, dt2, casting='no')  # 比较两个 dtype 是否可以强制转换

    # structured dtype
    dt = np.dtype([('l1', [('l2', np.dtype('S8', metadata={'msg': 'toto'}))])],
                  metadata={'msg': 'titi'})  # 创建一个结构化 dtype，并设置 metadata
    dt_m = _utils_impl.drop_metadata(dt)  # 调用 _utils_impl.drop_metadata 函数去除 metadata
    assert _compare_dtypes(dt, dt_m) is True  # 检查去除 metadata 后 dtype 是否保持可转换性
    assert dt_m.metadata is None  # 检查去除 metadata 后顶层 dtype 的 metadata 是否为 None
    assert dt_m['l1'].metadata is None  # 检查去除 metadata 后子 dtype 的 metadata 是否为 None

    # alignment
    dt = np.dtype([('x', '<f8'), ('y', '<i4')],
                  align=True,
                  metadata={'msg': 'toto'})  # 创建一个带 alignment 和 metadata 的 dtype
    dt_m = _utils_impl.drop_metadata(dt)  # 调用 _utils_impl.drop_metadata 函数去除 metadata
    assert _compare_dtypes(dt, dt_m) is True  # 检查去除 metadata 后 dtype 是否保持可转换性
    assert dt_m.metadata is None  # 检查去除 metadata 后顶层 dtype 的 metadata 是否为 None

    # subdtype
    dt = np.dtype('8f',
                  metadata={'msg': 'toto'})  # 创建一个带 metadata 的 subdtype
    dt_m = _utils_impl.drop_metadata(dt)  # 调用 _utils_impl.drop_metadata 函数去除 metadata
    assert _compare_dtypes(dt, dt_m) is True  # 检查去除 metadata 后 dtype 是否保持可转换性
    assert dt_m.metadata is None  # 检查去除 metadata 后 dtype 的 metadata 是否为 None

    # scalar
    dt = np.dtype('uint32',
                  metadata={'msg': 'toto'})  # 创建一个带 metadata 的标量 dtype
    dt_m = _utils_impl.drop_metadata(dt)  # 调用 _utils_impl.drop_metadata 函数去除 metadata
    assert _compare_dtypes(dt, dt_m) is True  # 检查去除 metadata 后 dtype 是否保持可转换性
    assert dt_m.metadata is None  # 检查去除 metadata 后 dtype 的 metadata 是否为 None


@pytest.mark.parametrize("dtype",
        [np.dtype("i,i,i,i")[["f1", "f3"]],
        np.dtype("f8"),
        np.dtype("10i")])
def test_drop_metadata_identity_and_copy(dtype):
    # If there is no metadata, the identity is preserved:
    assert _utils_impl.drop_metadata(dtype) is dtype  # 如果没有 metadata，则保持 dtype 的身份不变

    # If there is any, it is dropped (subforms are checked above)
    dtype = np.dtype(dtype, metadata={1: 2})  # 给 dtype 添加一个 metadata
    assert _utils_impl.drop_metadata(dtype).metadata is None  # 检查去除 metadata 后是否为 None
```