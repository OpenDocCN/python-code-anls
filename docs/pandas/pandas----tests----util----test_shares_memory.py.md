# `D:\src\scipysrc\pandas\pandas\tests\util\test_shares_memory.py`

```
# 导入测试装饰器模块 `_test_decorators`，用于测试工具函数和类的装饰器
import pandas.util._test_decorators as td

# 导入 Pandas 主模块，并命名为 pd
import pandas as pd
# 导入 Pandas 测试模块 `_testing`，并命名为 tm
import pandas._testing as tm


# 定义一个名为 test_shares_memory_interval 的测试函数
def test_shares_memory_interval():
    # 创建一个包含整数区间的对象，范围是 [1, 5)
    obj = pd.interval_range(1, 5)

    # 断言：检查对象是否与自身共享内存
    assert tm.shares_memory(obj, obj)
    # 断言：检查对象是否与其数据的内存共享
    assert tm.shares_memory(obj, obj._data)
    # 断言：检查对象是否与其反向索引的内存共享
    assert tm.shares_memory(obj, obj[::-1])
    # 断言：检查对象是否与其前两个元素的内存共享
    assert tm.shares_memory(obj, obj[:2])

    # 断言：检查对象与其数据副本的内存是否不共享
    assert not tm.shares_memory(obj, obj._data.copy())


# 使用装饰器 `skip_if_no`，如果没有安装 "pyarrow" 模块则跳过测试
@td.skip_if_no("pyarrow")
# 定义一个名为 test_shares_memory_string 的测试函数
def test_shares_memory_string():
    # GH#55823
    # 导入 pyarrow 库，并命名为 pa
    import pyarrow as pa

    # 创建一个包含字符串数组的对象，数据是 ["a", "b"]，类型为 "string[pyarrow]"
    obj = pd.array(["a", "b"], dtype="string[pyarrow]")
    # 断言：检查对象是否与自身共享内存
    assert tm.shares_memory(obj, obj)

    # 创建一个包含字符串数组的对象，数据是 ["a", "b"]，类型为 "string[pyarrow_numpy]"
    obj = pd.array(["a", "b"], dtype="string[pyarrow_numpy]")
    # 断言：检查对象是否与自身共享内存
    assert tm.shares_memory(obj, obj)

    # 创建一个包含字符串数组的对象，数据是 ["a", "b"]，类型为 pandas.ArrowDtype(pa.string())
    obj = pd.array(["a", "b"], dtype=pd.ArrowDtype(pa.string()))
    # 断言：检查对象是否与自身共享内存
    assert tm.shares_memory(obj, obj)
```