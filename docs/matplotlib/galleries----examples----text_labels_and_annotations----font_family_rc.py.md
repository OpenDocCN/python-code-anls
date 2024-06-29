# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\font_family_rc.py`

```py
"""
===========================
Configuring the font family
===========================

You can explicitly set which font family is picked up, either by specifying
family names of fonts installed on user's system, or generic-families
(e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
or a combination of both.
(see :ref:`text_props`)

In the example below, we are overriding the default sans-serif generic family
to include a specific (Tahoma) font. (Note that the best way to achieve this
would simply be to prepend 'Tahoma' in 'font.family')

The default family is set with the font.family rcparam,
e.g. ::

  rcParams['font.family'] = 'sans-serif'

and for the font.family you set a list of font styles to try to find
in order::

  rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                 'Lucida Grande', 'Verdana']

.. redirect-from:: /gallery/font_family_rc_sgskip

The ``font.family`` defaults are OS dependent and can be viewed with:
"""
# 引入 matplotlib.pyplot 库，并获取默认的 sans-serif 和 monospace 字体
import matplotlib.pyplot as plt

# 打印当前配置的 sans-serif 字体的第一个字体
print(plt.rcParams["font.sans-serif"][0])
# 打印当前配置的 monospace 字体的第一个字体
print(plt.rcParams["font.monospace"][0])

# %%
# 选择默认的 sans-serif 字体

def print_text(text):
    # 创建一个大小为 6x1 的图形，并设置背景色为浅绿色
    fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
    # 在图形上添加文本，居中显示，文本内容为传入的参数 text，字体大小为 40
    ax.text(0.5, 0.5, text, ha='center', va='center', size=40)
    # 关闭坐标轴
    ax.axis("off")
    # 显示图形
    plt.show()

# 设置默认的字体系列为 sans-serif，并打印 "Hello World! 01"
plt.rcParams["font.family"] = "sans-serif"
print_text("Hello World! 01")

# %%
# 选择 sans-serif 字体并指定为 "Nimbus Sans"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
print_text("Hello World! 02")

# %%
# 选择默认的 monospace 字体

plt.rcParams["font.family"] = "monospace"
print_text("Hello World! 03")

# %%
# 选择 monospace 字体并指定为 "FreeMono"

plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["FreeMono"]
print_text("Hello World! 04")
```