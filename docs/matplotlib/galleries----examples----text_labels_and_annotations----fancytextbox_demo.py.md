# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\fancytextbox_demo.py`

```py
"""
==================
Styling text boxes
==================

This example shows how to style text boxes using *bbox* parameters.
"""
# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 在指定坐标 (0.6, 0.7) 处绘制文本 "eggs"，设置文本大小为 50，旋转角度为 30 度，
# 水平对齐方式为居中，垂直对齐方式为居中，设置文本框样式为圆角矩形，
# 外边框颜色为 (1., 0.5, 0.5)，填充颜色为 (1., 0.8, 0.8)
plt.text(0.6, 0.7, "eggs", size=50, rotation=30.,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

# 在指定坐标 (0.55, 0.6) 处绘制文本 "spam"，设置文本大小为 50，旋转角度为 -25 度，
# 水平对齐方式为右对齐，垂直对齐方式为顶部对齐，设置文本框样式为方形，
# 外边框颜色和填充颜色与上面相同
plt.text(0.55, 0.6, "spam", size=50, rotation=-25.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

# 显示绘制的图形
plt.show()
```