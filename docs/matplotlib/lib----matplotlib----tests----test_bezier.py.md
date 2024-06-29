# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_bezier.py`

```py
"""
Tests specific to the bezier module.
"""

# 导入需要测试的函数
from matplotlib.bezier import inside_circle, split_bezier_intersecting_with_closedpath


# 定义测试函数，测试大数值情况下的贝塞尔曲线拆分
def test_split_bezier_with_large_values():
    # 这些数字来源于 gh-27753
    # 定义箭头路径的点集
    arrow_path = [(96950809781500.0, 804.7503795623779),
                  (96950809781500.0, 859.6242585800646),
                  (96950809781500.0, 914.4981375977513)]
    
    # 判断特定点是否在圆内，返回布尔值
    in_f = inside_circle(96950809781500.0, 804.7503795623779, 0.06)
    
    # 调用函数，拆分与闭合路径相交的贝塞尔曲线
    split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    
    # 我们的测试目的是确保这些操作能够完成
    # 失败情况是由于浮点精度导致的无限循环
    # 如果出现这种情况，pytest 会超时
```