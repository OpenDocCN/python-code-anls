# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_autocorr.py`

```
import numpy as np

# 定义一个测试自相关的类
class TestAutoCorr:
    
    # 定义测试自相关的方法，传入一个日期时间序列 datetime_series
    def test_autocorr(self, datetime_series):
        
        # 直接调用自相关函数，没有指定滞后参数
        corr1 = datetime_series.autocorr()
        
        # 指定滞后参数为1后再调用自相关函数
        corr2 = datetime_series.autocorr(lag=1)
        
        # 如果日期时间序列的长度小于等于2，自相关结果应为 NaN
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            # 如果长度大于2，自相关结果应相等
            assert corr1 == corr2
        
        # 选择一个随机的滞后期数 n，范围在1到序列长度减2之间
        n = 1 + np.random.default_rng(2).integers(max(1, len(datetime_series) - 2))
        
        # 计算序列和滞后n期的相关性
        corr1 = datetime_series.corr(datetime_series.shift(n))
        
        # 再次调用自相关函数，指定滞后参数为n
        corr2 = datetime_series.autocorr(lag=n)
        
        # 如果日期时间序列的长度小于等于2，自相关结果应为 NaN
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            # 如果长度大于2，自相关结果应相等
            assert corr1 == corr2
```