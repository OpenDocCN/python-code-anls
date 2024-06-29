# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\mathtext_asarray.py`

```py
"""
=======================
Convert texts to images
=======================
"""

# 导入所需模块和库
from io import BytesIO  # 导入 BytesIO 类，用于处理二进制数据流
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，并使用 plt 别名
from matplotlib.figure import Figure  # 从 matplotlib.figure 模块导入 Figure 类
from matplotlib.transforms import IdentityTransform  # 导入 IdentityTransform 类，用于坐标变换


def text_to_rgba(s, *, dpi, **kwargs):
    # 将文本字符串转换为图像的函数
    # 创建一个空的背景透明的 Figure 对象
    fig = Figure(facecolor="none")
    # 在 Figure 对象上绘制文本字符串
    fig.text(0, 0, s, **kwargs)
    # 使用 BytesIO 创建一个临时缓冲区
    with BytesIO() as buf:
        # 将 Figure 对象保存为 PNG 格式到缓冲区中，保留紧密的边界框并且填充为零
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)  # 将缓冲区指针移动到起始位置
        rgba = plt.imread(buf)  # 使用 plt.imread 从缓冲区中读取图像数据
    return rgba  # 返回读取的 RGBA 数据


fig = plt.figure()  # 创建一个新的图形对象
# 将文本字符串转换为 RGBA 图像数据
rgba1 = text_to_rgba(r"IQ: $\sigma_i=15$", color="blue", fontsize=20, dpi=200)
rgba2 = text_to_rgba(r"some other string", color="red", fontsize=20, dpi=200)
# 将转换后的文本图像数据绘制到 Figure 对象中的指定位置
fig.figimage(rgba1, 100, 50)
fig.figimage(rgba2, 100, 150)

# 可以直接使用像素坐标在 Figure 对象上绘制文本
fig.text(100, 250, r"IQ: $\sigma_i=15$", color="blue", fontsize=20,
         transform=IdentityTransform())
fig.text(100, 350, r"some other string", color="red", fontsize=20,
         transform=IdentityTransform())

plt.show()  # 显示绘制好的 Figure 对象

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.figimage`
#    - `matplotlib.figure.Figure.text`
#    - `matplotlib.transforms.IdentityTransform`
#    - `matplotlib.image.imread`
```