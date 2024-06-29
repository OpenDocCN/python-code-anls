# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\text_rotation_relative_to_line.py`

```
"""
==============================
Text Rotation Relative To Line
==============================

Text objects in matplotlib are normally rotated with respect to the
screen coordinate system (i.e., 45 degrees rotation plots text along a
line that is in between horizontal and vertical no matter how the axes
are changed).  However, at times one wants to rotate text with respect
to something on the plot.  In this case, the correct angle won't be
the angle of that object in the plot coordinate system, but the angle
that that object APPEARS in the screen coordinate system.  This angle
can be determined automatically by setting the parameter
*transform_rotates_text*, as shown in the example below.
"""

# 导入 matplotlib 的 pyplot 模块，命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，命名为 np
import numpy as np

# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()

# 绘制对角线 (45 度)
h = ax.plot(range(0, 10), range(0, 10))

# 设置 x 轴的显示范围，使其不再看起来是 45 度
ax.set_xlim([-10, 20])

# 设置要绘制文本的位置
l1 = np.array((1, 1))
l2 = np.array((5, 5))

# 设定文本旋转的角度
angle = 45

# 绘制第一个文本，但角度旋转不正确
th1 = ax.text(*l1, 'text not rotated correctly', fontsize=16,
              rotation=angle, rotation_mode='anchor')

# 绘制第二个文本，使用 transform_rotates_text 参数来自动确定正确的旋转角度
th2 = ax.text(*l2, 'text rotated correctly', fontsize=16,
              rotation=angle, rotation_mode='anchor',
              transform_rotates_text=True)

# 显示图形
plt.show()
```