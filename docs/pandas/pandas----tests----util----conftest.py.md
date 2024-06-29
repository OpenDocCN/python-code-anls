# `D:\src\scipysrc\pandas\pandas\tests\util\conftest.py`

```
# 导入 pytest 模块，用于测试框架
import pytest

# 定义一个 pytest 的 fixture，返回 True 或 False，用于确定是否检查数据类型是否相同
@pytest.fixture(params=[True, False])
def check_dtype(request):
    """
    Fixture returning `True` or `False`, determining whether to check
    if the `dtype` is identical or not, when comparing two data structures,
    e.g. `Series`, `SparseArray` or `DataFrame`.
    """
    return request.param

# 定义一个 pytest 的 fixture，返回 True 或 False，用于确定是否精确比较浮点数
@pytest.fixture(params=[True, False])
def check_exact(request):
    """
    Fixture returning `True` or `False`, determining whether to
    compare floating point numbers exactly or not.
    """
    return request.param

# 定义一个 pytest 的 fixture，返回 True 或 False，用于确定是否检查索引类型是否相同
@pytest.fixture(params=[True, False])
def check_index_type(request):
    """
    Fixture returning `True` or `False`, determining whether to check
    if the `Index` types are identical or not.
    """
    return request.param

# 定义一个 pytest 的 fixture，返回 0.5e-3 或 0.5e-5，用作相对容差
@pytest.fixture(params=[0.5e-3, 0.5e-5])
def rtol(request):
    """
    Fixture returning 0.5e-3 or 0.5e-5. Those values are used as relative tolerance.
    """
    return request.param

# 定义一个 pytest 的 fixture，返回 True 或 False，用于确定是否精确比较内部的 `Categorical`
@pytest.fixture(params=[True, False])
def check_categorical(request):
    """
    Fixture returning `True` or `False`, determining whether to
    compare internal `Categorical` exactly or not.
    """
    return request.param
```