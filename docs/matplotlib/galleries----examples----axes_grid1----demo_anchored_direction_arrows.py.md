# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_anchored_direction_arrows.py`

```
"""
========================
Anchored Direction Arrow
========================

"""
# 导入 matplotlib 的 pyplot 模块并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简写为 np
import numpy as np

# 导入 matplotlib 的字体管理模块并命名为 fm
import matplotlib.font_manager as fm
# 从 axes_grid1 中导入 AnchoredDirectionArrows 类
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

# 设置随机数种子以便结果可重现
np.random.seed(19680801)

# 创建一个图形和一个轴对象
fig, ax = plt.subplots()
# 在轴对象上显示一个 10x10 的随机数组
ax.imshow(np.random.random((10, 10)))

# 创建一个简单的箭头示例并添加到轴对象中
simple_arrow = AnchoredDirectionArrows(ax.transAxes, 'X', 'Y')
ax.add_artist(simple_arrow)

# 创建一个高对比度的箭头示例并添加到轴对象中
high_contrast_part_1 = AnchoredDirectionArrows(
                            ax.transAxes,
                            '111', r'11$\overline{2}$',
                            loc='upper right',
                            arrow_props={'ec': 'w', 'fc': 'none', 'alpha': 1,
                                         'lw': 2}
                            )
ax.add_artist(high_contrast_part_1)

# 创建第二个高对比度的箭头示例并添加到轴对象中
high_contrast_part_2 = AnchoredDirectionArrows(
                            ax.transAxes,
                            '111', r'11$\overline{2}$',
                            loc='upper right',
                            arrow_props={'ec': 'none', 'fc': 'k'},
                            text_props={'ec': 'w', 'fc': 'k', 'lw': 0.4}
                            )
ax.add_artist(high_contrast_part_2)

# 创建一个旋转的箭头示例并添加到轴对象中
fontprops = fm.FontProperties(family='serif')
rotated_arrow = AnchoredDirectionArrows(
                    ax.transAxes,
                    '30', '120',
                    loc='center',
                    color='w',
                    angle=30,
                    fontproperties=fontprops
                    )
ax.add_artist(rotated_arrow)

# 改变箭头方向示例
a1 = AnchoredDirectionArrows(
        ax.transAxes, 'A', 'B', loc='lower center',
        length=-0.15,
        sep_x=0.03, sep_y=0.03,
        color='r'
    )
ax.add_artist(a1)

a2 = AnchoredDirectionArrows(
        ax.transAxes, 'A', ' B', loc='lower left',
        aspect_ratio=-1,
        sep_x=0.01, sep_y=-0.02,
        color='orange'
        )
ax.add_artist(a2)

a3 = AnchoredDirectionArrows(
        ax.transAxes, ' A', 'B', loc='lower right',
        length=-0.15,
        aspect_ratio=-1,
        sep_y=-0.1, sep_x=0.04,
        color='cyan'
        )
ax.add_artist(a3)

# 显示图形
plt.show()
```