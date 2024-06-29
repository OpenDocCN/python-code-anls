# `D:\src\scipysrc\matplotlib\galleries\examples\misc\findobj_demo.py`

```
"""
============
Findobj Demo
============

Recursively find all objects that match some criteria
"""
# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 导入 matplotlib.text 模块，并简写为 text
import matplotlib.text as text

# 创建 numpy 数组 a, b, c, d，分别存储范围从 0 到 3，步长为 0.02 的值
a = np.arange(0, 3, .02)
b = np.arange(0, 3, .02)
# 计算数组 a 的指数值，并存储在数组 c 中
c = np.exp(a)
# 将数组 c 反向排序，并存储在数组 d 中
d = c[::-1]

# 创建一个新的图形 fig 和轴对象 ax
fig, ax = plt.subplots()
# 绘制三条曲线，并设置不同的线型和标签
plt.plot(a, c, 'k--', a, d, 'k:', a, c + d, 'k')
# 添加图例，说明每条曲线的含义，并设置位置为上中，带有阴影效果
plt.legend(('Model length', 'Data length', 'Total message length'),
           loc='upper center', shadow=True)
# 设置 y 轴的数值范围
plt.ylim([-1, 20])
# 不显示网格线
plt.grid(False)
# 设置 x 轴标签
plt.xlabel('Model complexity --->')
# 设置 y 轴标签
plt.ylabel('Message length --->')
# 设置图形标题
plt.title('Minimum Message Length')


# 定义一个函数 myfunc，用于检查对象是否具有 'set_color' 属性但没有 'set_facecolor' 属性
def myfunc(x):
    return hasattr(x, 'set_color') and not hasattr(x, 'set_facecolor')

# 使用 fig.findobj 方法，递归查找所有符合 myfunc 函数条件的对象，并将它们的颜色设置为蓝色
for o in fig.findobj(myfunc):
    o.set_color('blue')

# 使用 fig.findobj 方法，查找所有是 text.Text 类的对象实例，并将它们的字体风格设置为斜体
for o in fig.findobj(text.Text):
    o.set_fontstyle('italic')


# 显示绘制的图形
plt.show()
```