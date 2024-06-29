# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_container.py`

```
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，用于测试 stem 图形对象的创建和删除
def test_stem_remove():
    # 获取当前图形的轴对象
    ax = plt.gca()
    # 在轴上绘制 stem 图形，并获取返回的对象
    st = ax.stem([1, 2], [1, 2])
    # 删除之前绘制的 stem 图形对象
    st.remove()

# 定义一个函数，用于测试 errorbar 图形对象的创建和删除
def test_errorbar_remove():
    # Regression test for a bug that caused remove to fail when using
    # fmt='none'

    # 获取当前图形的轴对象
    ax = plt.gca()

    # 创建 errorbar 图形对象，并获取返回的对象
    eb = ax.errorbar([1], [1])
    # 删除之前绘制的 errorbar 图形对象
    eb.remove()

    # 创建带有 x 轴误差条的 errorbar 图形对象，并删除
    eb = ax.errorbar([1], [1], xerr=1)
    eb.remove()

    # 创建带有 y 轴误差条的 errorbar 图形对象，并删除
    eb = ax.errorbar([1], [1], yerr=2)
    eb.remove()

    # 创建同时带有 x 和 y 轴误差条的 errorbar 图形对象，并删除
    eb = ax.errorbar([1], [1], xerr=[2], yerr=2)
    eb.remove()

    # 创建使用 fmt='none' 参数的 errorbar 图形对象，并删除
    eb = ax.errorbar([1], [1], fmt='none')
    eb.remove()

# 定义一个函数，测试非字符串标签在 bar 图形中的使用情况
def test_nonstring_label():
    # Test for #26824
    # 绘制带有非字符串标签的条形图
    plt.bar(np.arange(10), np.random.rand(10), label=1)
    # 添加图例
    plt.legend()
```