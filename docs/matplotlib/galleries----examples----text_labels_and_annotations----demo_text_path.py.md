# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\demo_text_path.py`

```
"""
======================
Using a text as a Path
======================

`~matplotlib.text.TextPath` creates a `.Path` that is the outline of the
characters of a text. The resulting path can be employed e.g. as a clip path
for an image.
"""

import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy数值计算库

from matplotlib.cbook import get_sample_data  # 导入获取示例数据的函数
from matplotlib.image import BboxImage  # 导入边界框图像类
from matplotlib.offsetbox import (AnchoredOffsetbox, AnnotationBbox,
                                  AuxTransformBox)  # 导入偏移框相关类
from matplotlib.patches import PathPatch, Shadow  # 导入路径补丁类和阴影类
from matplotlib.text import TextPath  # 导入文本路径类
from matplotlib.transforms import IdentityTransform  # 导入单位转换类


class PathClippedImagePatch(PathPatch):
    """
    The given image is used to draw the face of the patch. Internally,
    it uses BboxImage whose clippath set to the path of the patch.

    FIXME : The result is currently dpi dependent.
    """

    def __init__(self, path, bbox_image, **kwargs):
        super().__init__(path, **kwargs)
        self.bbox_image = BboxImage(
            self.get_window_extent, norm=None, origin=None)
        self.bbox_image.set_data(bbox_image)

    def set_facecolor(self, color):
        """Simply ignore facecolor."""
        super().set_facecolor("none")

    def draw(self, renderer=None):
        # the clip path must be updated every draw. any solution? -JJ
        self.bbox_image.set_clip_path(self._path, self.get_transform())
        self.bbox_image.draw(renderer)
        super().draw(renderer)


if __name__ == "__main__":

    fig, (ax1, ax2) = plt.subplots(2)  # 创建包含两个子图的图形对象

    # EXAMPLE 1

    arr = plt.imread(get_sample_data("grace_hopper.jpg"))  # 读取示例图片数据

    text_path = TextPath((0, 0), "!?", size=150)  # 创建文本路径对象
    p = PathClippedImagePatch(text_path, arr, ec="k")  # 创建被裁剪图像路径补丁对象

    # make offset box
    offsetbox = AuxTransformBox(IdentityTransform())  # 创建基于单位变换的偏移框对象
    offsetbox.add_artist(p)  # 将路径补丁对象添加到偏移框中

    # make anchored offset box
    ao = AnchoredOffsetbox(loc='upper left', child=offsetbox, frameon=True,
                           borderpad=0.2)  # 创建锚定的偏移框对象
    ax1.add_artist(ao)  # 将锚定的偏移框对象添加到第一个子图中

    # another text
    for usetex, ypos, string in [
            (False, 0.25, r"textpath supports mathtext"),
            (True, 0.05, r"textpath supports \TeX"),
    ]:
        text_path = TextPath((0, 0), string, size=20, usetex=usetex)  # 创建文本路径对象

        p1 = PathPatch(text_path, ec="w", lw=3, fc="w", alpha=0.9)  # 创建路径补丁对象
        p2 = PathPatch(text_path, ec="none", fc="k")  # 创建路径补丁对象

        offsetbox2 = AuxTransformBox(IdentityTransform())  # 创建基于单位变换的偏移框对象
        offsetbox2.add_artist(p1)  # 将路径补丁对象添加到偏移框中
        offsetbox2.add_artist(p2)  # 将路径补丁对象添加到偏移框中

        ab = AnnotationBbox(offsetbox2, (0.95, ypos),
                            xycoords='axes fraction',
                            boxcoords="offset points",
                            box_alignment=(1., 0.),
                            frameon=False,
                            )  # 创建注释框对象
        ax1.add_artist(ab)  # 将注释框对象添加到第一个子图中

    ax1.imshow([[0, 1, 2], [1, 2, 3]], cmap=plt.cm.gist_gray_r,
               interpolation="bilinear", aspect="auto")  # 在第一个子图中显示灰度图像

    # EXAMPLE 2

    arr = np.arange(256).reshape(1, 256)  # 创建256x1的数组
    for usetex, xpos, string in [
            (False, 0.25,
             r"$\left[\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}\right]$!"),
            (True, 0.75,
             r"$\displaystyle\left[\sum_{n=1}^\infty"
             r"\frac{-e^{i\pi}}{2^n}\right]$!"),
    ]:
        # 创建一个文本路径对象，用指定的字符串和参数进行初始化
        text_path = TextPath((0, 0), string, size=40, usetex=usetex)
        # 创建一个路径裁剪图像补丁对象，使用给定的文本路径和图像数组
        text_patch = PathClippedImagePatch(text_path, arr, ec="none")
        # 创建一个阴影对象，作为文本补丁的阴影效果，设置参数如阴影位置、颜色等
        shadow1 = Shadow(text_patch, 1, -1, fc="none", ec="0.6", lw=3)
        # 创建第二个阴影对象，设置参数如阴影位置、颜色等
        shadow2 = Shadow(text_patch, 1, -1, fc="0.3", ec="none")

        # 创建一个辅助变换框对象，使用单位变换初始化
        offsetbox = AuxTransformBox(IdentityTransform())
        # 将第一个阴影对象添加到辅助变换框中
        offsetbox.add_artist(shadow1)
        # 将第二个阴影对象添加到辅助变换框中
        offsetbox.add_artist(shadow2)
        # 将文本补丁对象添加到辅助变换框中
        offsetbox.add_artist(text_patch)

        # 使用AnnotationBbox将锚定的偏移框放置到图中的特定位置
        ab = AnnotationBbox(offsetbox, (xpos, 0.5), box_alignment=(0.5, 0.5))

        # 将注释框添加到指定的轴对象ax2中
        ax2.add_artist(ab)

    # 设置轴对象ax2的x轴范围
    ax2.set_xlim(0, 1)
    # 设置轴对象ax2的y轴范围
    ax2.set_ylim(0, 1)

    # 显示绘制的图形
    plt.show()
```