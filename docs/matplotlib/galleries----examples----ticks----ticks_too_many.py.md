# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\ticks_too_many.py`

```py
# %%
# Example 1: Strings can lead to an unexpected order of number ticks
# ------------------------------------------------------------------

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值操作

fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(6, 2.5))  # 创建包含两个子图的画布
x = ['1', '5', '2', '3']  # 定义 x 轴数据，是一个字符串列表
y = [1, 4, 2, 3]  # 定义 y 轴数据
ax[0].plot(x, y, 'd')  # 在第一个子图上绘制散点图，使用字符串作为 x 轴数据
ax[0].tick_params(axis='x', color='r', labelcolor='r')  # 设置 x 轴的刻度参数，包括颜色和标签颜色
ax[0].set_xlabel('Categories')  # 设置 x 轴标签
ax[0].set_title('Ticks seem out of order / misplaced')  # 设置子图标题

# convert to numbers:
x = np.asarray(x, dtype='float')  # 将 x 转换为浮点数类型的 numpy 数组
ax[1].plot(x, y, 'd')  # 在第二个子图上绘制散点图，此时 x 轴数据为数字类型
ax[1].set_xlabel('Floats')  # 设置 x 轴标签
ax[1].set_title('Ticks as expected')  # 设置子图标题

# %%
# Example 2: Strings can lead to very many ticks
# ----------------------------------------------
# If *x* has 100 elements, all strings, then we would have 100 (unreadable)
# ticks, and again the solution is to convert the strings to floats:

fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))  # 创建包含两个子图的画布
x = [f'{xx}' for xx in np.arange(100)]  # 生成一个包含 100 个字符串的列表，每个元素为其索引值的字符串表示
y = np.arange(100)  # 生成一个包含 0 到 99 的整数数组
ax[0].plot(x, y)  # 在第一个子图上绘制折线图，使用字符串作为 x 轴数据
ax[0].tick_params(axis='x', color='r', labelcolor='r')  # 设置 x 轴的刻度参数，包括颜色和标签颜色
ax[0].set_title('Too many ticks')  # 设置子图标题
ax[0].set_xlabel('Categories')  # 设置 x 轴标签

ax[1].plot(np.asarray(x, float), y)  # 在第二个子图上绘制折线图，此时将 x 转换为浮点数类型的 numpy 数组
ax[1].set_title('x converted to numbers')  # 设置子图标题
ax[1].set_xlabel('Floats')  # 设置 x 轴标签

# %%
# Example 3: Strings can lead to an unexpected order of datetime ticks
# --------------------------------------------------------------------
# A common case is when dates are read from a CSV file, they need to be
# converted from strings to datetime objects to get the proper date locators
# and formatters.

fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(6, 2.75))  # 创建包含两个子图的画布
x = ['2021-10-01', '2021-11-02', '2021-12-03', '2021-09-01']  # 定义包含日期字符串的列表
y = [0, 2, 3, 1]  # 定义 y 轴数据
ax[0].plot(x, y, 'd')  # 在第一个子图上绘制散点图，使用字符串作为 x 轴数据
ax[0].tick_params(axis='x', labelrotation=90, color='r', labelcolor='r')  # 设置 x 轴的刻度参数，包括旋转角度、颜色和标签颜色
ax[0].set_title('Dates out of order')  # 设置子图标题

# convert to datetime64
x = np.asarray(x, dtype='datetime64[s]')  # 将 x 转换为 datetime64[s] 类型的 numpy 数组
ax[1].plot(x, y, 'd')  # 在第二个子图上绘制散点图，此时 x 轴数据为 datetime 类型
ax[1].tick_params(axis='x', labelrotation=90)  # 设置 x 轴的刻度参数，包括旋转角度
ax[1].set_title('x converted to datetimes')  # 设置子图标题

plt.show()  # 显示图形
```