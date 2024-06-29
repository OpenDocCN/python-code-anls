# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\simple_legend01.py`

```py
"""
===============
Simple Legend01
===============

"""
# 导入matplotlib.pyplot模块，并简称为plt
import matplotlib.pyplot as plt

# 创建一个新的Figure对象
fig = plt.figure()

# 在Figure对象上添加一个2x2的子图，并选取第一个子图
ax = fig.add_subplot(211)
# 在第一个子图上绘制一条曲线，设置标签为'test1'
ax.plot([1, 2, 3], label="test1")
# 在第一个子图上继续绘制另一条曲线，设置标签为'test2'
ax.plot([3, 2, 1], label="test2")
# 在子图的上方放置图例，使其扩展以充分利用给定的边界框
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncols=2, mode="expand", borderaxespad=0.)

# 在Figure对象上添加一个2x2的子图，并选取第三个子图
ax = fig.add_subplot(223)
# 在第三个子图上绘制一条曲线，设置标签为'test1'
ax.plot([1, 2, 3], label="test1")
# 在第三个子图上继续绘制另一条曲线，设置标签为'test2'
ax.plot([3, 2, 1], label="test2")
# 在子图的右侧放置图例
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 显示图形
plt.show()
```