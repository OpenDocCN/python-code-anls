# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\pathpatch3d.py`

```
# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并重命名为 plt
import numpy as np  # 导入 numpy 库并重命名为 np

from matplotlib.patches import Circle, PathPatch  # 从 matplotlib.patches 模块导入 Circle 和 PathPatch 类
from matplotlib.text import TextPath  # 从 matplotlib.text 模块导入 TextPath 类
from matplotlib.transforms import Affine2D  # 从 matplotlib.transforms 模块导入 Affine2D 类
import mpl_toolkits.mplot3d.art3d as art3d  # 导入 mplot3d.art3d 模块并重命名为 art3d

def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    """
    在 Axes *ax* 上绘制字符串 *s*，位置为 *xyz*，大小为 *size*，角度为 *angle*。
    *zdir* 指定哪个轴作为第三维度。*usetex* 是一个布尔值，指示是否通过 LaTeX 进行处理。
    其他关键字参数将被传递给 `.transform_path`。

    注意：zdir 影响 xyz 的解释。
    """
    x, y, z = xyz  # 将参数 xyz 解包为 x, y, z
    if zdir == "y":
        xy1, z1 = (x, z), y  # 如果 zdir 是 "y"，则调整坐标顺序
    elif zdir == "x":
        xy1, z1 = (y, z), x  # 如果 zdir 是 "x"，则调整坐标顺序
    else:
        xy1, z1 = (x, y), z  # 否则保持原始坐标顺序

    # 创建文本路径对象
    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    # 创建仿射变换对象，进行旋转和平移
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    # 创建路径补丁对象，并应用关键字参数
    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    # 将路径补丁对象添加到 Axes *ax* 中
    ax.add_patch(p1)
    # 将二维路径补丁对象转换为三维，指定 z 和 zdir
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


# 创建一个新的 3D 图形对象
fig = plt.figure()
# 向图形对象添加一个 3D 子图
ax = fig.add_subplot(projection='3d')

# 在 x=0 的墙面绘制一个圆
p = Circle((5, 5), 3)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")

# 手动标记坐标轴
text3d(ax, (4, -2, 0), "X-axis", zdir="z", size=.5, usetex=False,
       ec="none", fc="k")
text3d(ax, (12, 4, 0), "Y-axis", zdir="z", size=.5, usetex=False,
       angle=np.pi / 2, ec="none", fc="k")
text3d(ax, (12, 10, 4), "Z-axis", zdir="y", size=.5, usetex=False,
       angle=np.pi / 2, ec="none", fc="k")

# 在 z=0 的地面上写一个 LaTeX 公式
text3d(ax, (1, 5, 0),
       r"$\displaystyle G_{\mu\nu} + \Lambda g_{\mu\nu} = "
       r"\frac{8\pi G}{c^4} T_{\mu\nu}  $",
       zdir="z", size=1, usetex=True,
       ec="none", fc="k")

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)

# 显示图形
plt.show()
```