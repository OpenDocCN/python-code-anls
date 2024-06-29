# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\demo_text_rotation_mode.py`

```py
r"""
==================
Text Rotation Mode
==================

This example illustrates the effect of ``rotation_mode`` on the positioning
of rotated text.

Rotated `.Text`\s are created by passing the parameter ``rotation`` to
the constructor or the Axes' method `~.axes.Axes.text`.

The actual positioning depends on the additional parameters
``horizontalalignment``, ``verticalalignment`` and ``rotation_mode``.
``rotation_mode`` determines the order of rotation and alignment:

- ``rotation_mode='default'`` (or None) first rotates the text and then aligns
  the bounding box of the rotated text.
- ``rotation_mode='anchor'`` aligns the unrotated text and then rotates the
  text around the point of alignment.

.. redirect-from:: /gallery/text_labels_and_annotations/text_rotation
"""

import matplotlib.pyplot as plt

# 定义一个函数用于测试不同的 rotation_mode 对于文本旋转效果的影响
def test_rotation_mode(fig, mode):
    # 水平对齐方式列表
    ha_list = ["left", "center", "right"]
    # 垂直对齐方式列表
    va_list = ["top", "center", "baseline", "bottom"]
    # 创建子图网格，每个子图使用相同的 x 和 y 轴，确保比例相同
    axs = fig.subplots(len(va_list), len(ha_list), sharex=True, sharey=True,
                       subplot_kw=dict(aspect=1),
                       gridspec_kw=dict(hspace=0, wspace=0))

    # 设置每个子图的 x 轴标签和标题
    for ha, ax in zip(ha_list, axs[-1, :]):
        ax.set_xlabel(ha)
    # 设置每个子图的 y 轴标签
    for va, ax in zip(va_list, axs[:, 0]):
        ax.set_ylabel(va)
    # 设置中间子图的标题，显示当前 rotation_mode 的取值
    axs[0, 1].set_title(f"rotation_mode='{mode}'", size="large")

    # 根据不同的 rotation_mode 设置不同的文本框样式
    kw = (
        {} if mode == "default" else
        {"bbox": dict(boxstyle="square,pad=0.", ec="none", fc="C1", alpha=0.3)}
    )

    # 在每个子图中添加旋转和对齐设置的文本
    texts = {}
    for i, va in enumerate(va_list):
        for j, ha in enumerate(ha_list):
            ax = axs[i, j]
            # 准备子图布局
            ax.set(xticks=[], yticks=[])
            ax.axvline(0.5, color="skyblue", zorder=0)
            ax.axhline(0.5, color="skyblue", zorder=0)
            ax.plot(0.5, 0.5, color="C0", marker="o", zorder=1)
            # 添加带有旋转和对齐设置的文本
            tx = ax.text(0.5, 0.5, "Tpg",
                         size="x-large", rotation=40,
                         horizontalalignment=ha, verticalalignment=va,
                         rotation_mode=mode, **kw)
            texts[ax] = tx

    # 如果 rotation_mode 为 'default'，突出显示文本框
    if mode == "default":
        fig.canvas.draw()
        for ax, text in texts.items():
            # 获取文本框的边界框
            bb = text.get_window_extent().transformed(ax.transData.inverted())
            # 在子图中添加一个矩形框，表示文本框的位置和大小
            rect = plt.Rectangle((bb.x0, bb.y0), bb.width, bb.height,
                                 facecolor="C1", alpha=0.3, zorder=2)
            ax.add_patch(rect)

fig = plt.figure(figsize=(8, 5))
subfigs = fig.subfigures(1, 2)
# 在两个子图中分别测试 'default' 和 'anchor' 两种 rotation_mode
test_rotation_mode(subfigs[0], "default")
test_rotation_mode(subfigs[1], "anchor")
plt.show()
```