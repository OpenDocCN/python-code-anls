# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_dtypes.py`

```
import numpy as np

# 定义一个测试类 TestSeriesDtypes
class TestSeriesDtypes:
    # 定义一个测试方法 test_dtype，接受一个 datetime_series 参数
    def test_dtype(self, datetime_series):
        # 断言 datetime_series 的数据类型为 np.float64
        assert datetime_series.dtype == np.dtype("float64")
        # 断言 datetime_series 的数据类型为 np.float64（复数形式）
        assert datetime_series.dtypes == np.dtype("float64")
```