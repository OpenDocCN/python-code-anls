# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\angles_on_bracket_arrows.py`

```py
"""
===================================
Angle annotations on bracket arrows
===================================

This example shows how to add angle annotations to bracket arrow styles
created using `.FancyArrowPatch`. *angleA* and *angleB* are measured from a
vertical line as positive (to the left) or negative (to the right). Blue
`.FancyArrowPatch` arrows indicate the directions of *angleA* and *angleB*
from the vertical and axes text annotate the angle sizes.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库

from matplotlib.patches import FancyArrowPatch  # 从 matplotlib.patches 模块导入 FancyArrowPatch 类


def get_point_of_rotated_vertical(origin, line_length, degrees):
    """Return xy coordinates of the vertical line end rotated by degrees."""
    rad = np.deg2rad(-degrees)  # 将角度转换为弧度
    return [origin[0] + line_length * np.sin(rad),  # 计算旋转后垂直线末端的 x 坐标
            origin[1] + line_length * np.cos(rad)]  # 计算旋转后垂直线末端的 y 坐标


fig, ax = plt.subplots()  # 创建一个新的图形和轴
ax.set(xlim=(0, 6), ylim=(-1, 5))  # 设置 x 轴和 y 轴的范围
ax.set_title("Orientation of the bracket arrows relative to angleA and angleB")  # 设置图表标题

style = ']-['  # 定义箭头样式
for i, angle in enumerate([-40, 0, 60]):
    y = 2*i  # 计算箭头垂直位置
    arrow_centers = ((1, y), (5, y))  # 定义箭头中心点的位置
    vlines = ((1, y + 0.5), (5, y + 0.5))  # 定义垂直线的位置
    anglesAB = (angle, -angle)  # 设置 angleA 和 angleB 的角度值
    bracketstyle = f"{style}, angleA={anglesAB[0]}, angleB={anglesAB[1]}"  # 定义箭头样式字符串
    bracket = FancyArrowPatch(*arrow_centers, arrowstyle=bracketstyle,  # 创建 FancyArrowPatch 对象
                              mutation_scale=42)
    ax.add_patch(bracket)  # 将箭头添加到图表
    ax.text(3, y + 0.05, bracketstyle, ha="center", va="bottom", fontsize=14)  # 在箭头旁添加样式文字说明
    ax.vlines([line[0] for line in vlines], [y, y], [line[1] for line in vlines],  # 在图表上绘制垂直线
              linestyles="--", color="C0")
    # 获取 A 和 B 处绘制箭头的顶部坐标
    patch_tops = [get_point_of_rotated_vertical(center, 0.5, angle)
                  for center, angle in zip(arrow_centers, anglesAB)]
    # 定义注释箭头的连接方向
    connection_dirs = (1, -1) if angle > 0 else (-1, 1)
    # 添加箭头和注释文本
    arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8"
    for vline, dir, patch_top, angle in zip(vlines, connection_dirs,
                                            patch_tops, anglesAB):
        kw = dict(connectionstyle=f"arc3,rad={dir * 0.5}",
                  arrowstyle=arrowstyle, color="C0")
        ax.add_patch(FancyArrowPatch(vline, patch_top, **kw))
        ax.text(vline[0] - dir * 0.15, y + 0.7, f'{angle}°', ha="center",
                va="center")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.ArrowStyle`
```