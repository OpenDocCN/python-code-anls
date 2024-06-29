# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_gridspec.py`

```py
import matplotlib.gridspec as gridspec  # 导入matplotlib的gridspec模块，用于创建子图网格布局
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import pytest  # 导入pytest模块，用于编写和运行单元测试

# 定义一个单元测试函数，测试GridSpec对象的相等性
def test_equal():
    gs = gridspec.GridSpec(2, 1)  # 创建一个2行1列的GridSpec对象
    assert gs[0, 0] == gs[0, 0]  # 断言GridSpec对象中的一个子网格与自身相等
    assert gs[:, 0] == gs[:, 0]  # 断言GridSpec对象中的整列子网格与自身相等

# 定义一个单元测试函数，测试GridSpec对象的宽度比例设置
def test_width_ratios():
    """
    Addresses issue #5835.
    See at https://github.com/matplotlib/matplotlib/issues/5835.
    """
    with pytest.raises(ValueError):  # 使用pytest检测是否会引发 ValueError 异常
        gridspec.GridSpec(1, 1, width_ratios=[2, 1, 3])  # 创建一个GridSpec对象，并设置宽度比例

# 定义一个单元测试函数，测试GridSpec对象的高度比例设置
def test_height_ratios():
    """
    Addresses issue #5835.
    See at https://github.com/matplotlib/matplotlib/issues/5835.
    """
    with pytest.raises(ValueError):  # 使用pytest检测是否会引发 ValueError 异常
        gridspec.GridSpec(1, 1, height_ratios=[2, 1, 3])  # 创建一个GridSpec对象，并设置高度比例

# 定义一个单元测试函数，测试GridSpec对象的字符串表示形式
def test_repr():
    ss = gridspec.GridSpec(3, 3)[2, 1:3]  # 获取GridSpec对象中的子网格范围
    assert repr(ss) == "GridSpec(3, 3)[2:3, 1:3]"  # 断言GridSpec对象的字符串表示形式是否符合预期

    ss = gridspec.GridSpec(2, 2,  # 创建一个2行2列的GridSpec对象
                           height_ratios=(3, 1),  # 设置高度比例
                           width_ratios=(1, 3))   # 设置宽度比例
    assert repr(ss) == \
        "GridSpec(2, 2, height_ratios=(3, 1), width_ratios=(1, 3))"  # 断言GridSpec对象的字符串表示形式是否符合预期

# 定义一个单元测试函数，测试GridSpecFromSubplotSpec函数的参数设置
def test_subplotspec_args():
    fig, axs = plt.subplots(1, 2)  # 创建一个包含两个子图的Figure对象
    # 应当成功创建GridSpec对象：
    gs = gridspec.GridSpecFromSubplotSpec(2, 1,
                                          subplot_spec=axs[0].get_subplotspec())  # 根据子图的SubplotSpec创建GridSpec对象
    assert gs.get_topmost_subplotspec() == axs[0].get_subplotspec()  # 断言获取的GridSpec对象的最顶层SubplotSpec与预期相等
    # 应当引发TypeError异常，因为subplot_spec参数不是SubplotSpec类型：
    with pytest.raises(TypeError, match="subplot_spec must be type SubplotSpec"):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=axs[0])  # 错误地使用一个Axes对象作为subplot_spec参数
    # 应当引发TypeError异常，因为subplot_spec参数不是SubplotSpec类型：
    with pytest.raises(TypeError, match="subplot_spec must be type SubplotSpec"):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=axs)  # 错误地使用一个Axes对象数组作为subplot_spec参数
```