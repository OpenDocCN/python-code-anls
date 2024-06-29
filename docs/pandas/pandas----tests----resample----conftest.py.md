# `D:\src\scipysrc\pandas\pandas\tests\resample\conftest.py`

```
import pytest
# 定义各种支持的降采样方法列表
downsample_methods = [
    "min",      # 取最小值
    "max",      # 取最大值
    "first",    # 取第一个值
    "last",     # 取最后一个值
    "sum",      # 求和
    "mean",     # 求平均值
    "sem",      # 计算标准误差
    "median",   # 计算中位数
    "prod",     # 计算积
    "var",      # 计算方差
    "std",      # 计算标准差
    "ohlc",     # 对开盘价、最高价、最低价和收盘价进行重采样
    "quantile", # 计算分位数
]
# 定义各种支持的升采样方法列表
upsample_methods = ["count", "size"]
# 定义支持的序列方法列表
series_methods = ["nunique"]
# 将所有的降采样、升采样和序列方法合并到一个列表中
resample_methods = downsample_methods + upsample_methods + series_methods

# 为降采样方法创建的 pytest fixture，参数化使用
@pytest.fixture(params=downsample_methods)
def downsample_method(request):
    """Fixture for parametrization of Grouper downsample methods."""
    return request.param

# 为重采样方法创建的 pytest fixture，参数化使用
@pytest.fixture(params=resample_methods)
def resample_method(request):
    """Fixture for parametrization of Grouper resample methods."""
    return request.param
```