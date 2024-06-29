# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\line_with_text.py`

```py
"""
=======================
Artist within an artist
=======================

Override basic methods so an artist can contain another
artist.  In this case, the line contains a Text instance to label it.
"""
# 导入 matplotlib 库中需要的模块
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.lines as lines
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms

# 自定义 MyLine 类，继承自 matplotlib 中的 Line2D 类
class MyLine(lines.Line2D):
    def __init__(self, *args, **kwargs):
        # 当线条数据设置时更新位置
        self.text = mtext.Text(0, 0, '')  # 创建一个 Text 实例作为标签
        super().__init__(*args, **kwargs)

        # 只能在线条初始化之后才能访问标签属性
        self.text.set_text(self.get_label())

    def set_figure(self, figure):
        self.text.set_figure(figure)
        super().set_figure(figure)

    # 重写 Axes 属性的 setter 方法，以便在子元素上设置 Axes
    @lines.Line2D.axes.setter
    def axes(self, new_axes):
        self.text.axes = new_axes
        lines.Line2D.axes.fset(self, new_axes)  # 调用超类的属性 setter 方法

    def set_transform(self, transform):
        # 设置文本的变换，增加2像素的偏移
        texttrans = transform + mtransforms.Affine2D().translate(2, 2)
        self.text.set_transform(texttrans)
        super().set_transform(transform)

    def set_data(self, x, y):
        if len(x):
            self.text.set_position((x[-1], y[-1]))  # 将标签位置设置为线条末端位置

        super().set_data(x, y)

    def draw(self, renderer):
        # 在线条末端绘制标签，并增加2像素的偏移
        super().draw(renderer)
        self.text.draw(renderer)

# 设置随机种子以便重现结果
np.random.seed(19680801)

# 创建图形和坐标轴
fig, ax = plt.subplots()
x, y = np.random.rand(2, 20)
# 创建 MyLine 实例并设置属性
line = MyLine(x, y, mfc='red', ms=12, label='line label')
line.text.set_color('red')
line.text.set_fontsize(16)

# 将 MyLine 实例添加到坐标轴
ax.add_line(line)

plt.show()
```