# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\font_file.py`

```
"""
====================
Using ttf font files
====================

Although it is usually not a good idea to explicitly point to a single ttf file
for a font instance, you can do so by passing a `pathlib.Path` instance as the
*font* parameter.  Note that passing paths as `str`\s is intentionally not
supported, but you can simply wrap `str`\s in `pathlib.Path`\s as needed.

Here, we use the Computer Modern roman font (``cmr10``) shipped with
Matplotlib.

For a more flexible solution, see
:doc:`/gallery/text_labels_and_annotations/font_family_rc` and
:doc:`/gallery/text_labels_and_annotations/fonts_demo`.
"""

# 导入所需模块
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 指定字体文件的路径
fpath = Path(mpl.get_data_path(), "fonts/ttf/cmr10.ttf")

# 设置图表标题，使用指定的字体文件
ax.set_title(f'This is a special font: {fpath.name}', font=fpath)

# 设置默认的 X 轴标签
ax.set_xlabel('This is the default font')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.set_title`
```