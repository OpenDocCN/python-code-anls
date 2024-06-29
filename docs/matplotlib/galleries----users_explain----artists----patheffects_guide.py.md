# `D:\src\scipysrc\matplotlib\galleries\users_explain\artists\patheffects_guide.py`

```
"""
.. redirect-from:: /tutorials/advance/patheffects_guide

.. _patheffects_guide:

==================
Path effects guide
==================

Defining paths that objects follow on a canvas.

.. py:currentmodule:: matplotlib.patheffects

Matplotlib's :mod:`.patheffects` module provides functionality to apply a
multiple draw stage to any Artist which can be rendered via a `.path.Path`.

Artists which can have a path effect applied to them include `.patches.Patch`,
`.lines.Line2D`, `.collections.Collection` and even `.text.Text`. Each artist's
path effects can be controlled via the `.Artist.set_path_effects` method,
which takes an iterable of `AbstractPathEffect` instances.

The simplest path effect is the `Normal` effect, which simply draws the artist
without any effect:
"""

# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt

# 导入 matplotlib 的 patheffects 模块，并简称为 path_effects
import matplotlib.patheffects as path_effects

# 创建一个大小为 (5, 1.5) 的新图形对象
fig = plt.figure(figsize=(5, 1.5))

# 在图形上添加文本，位置为 (0.5, 0.5)，内容为 'Hello path effects world!\nThis is the normal path effect.\nPretty dull, huh?'
# 水平对齐方式为居中，垂直对齐方式为居中，字体大小为 20
text = fig.text(0.5, 0.5, 'Hello path effects world!\nThis is the normal '
                          'path effect.\nPretty dull, huh?',
                ha='center', va='center', size=20)

# 给文本对象设置路径效果，使用 Normal 路径效果，即不应用任何特效
text.set_path_effects([path_effects.Normal()])

# 显示图形
plt.show()

# %%
# Whilst the plot doesn't look any different to what you would expect without
# any path effects, the drawing of the text has now been changed to use the
# path effects framework, opening up the possibilities for more interesting
# examples.
#
# Adding a shadow
# ---------------
#
# A far more interesting path effect than `Normal` is the drop-shadow, which we
# can apply to any of our path based artists. The classes `SimplePatchShadow`
# and `SimpleLineShadow` do precisely this by drawing either a filled patch or
# a line patch below the original artist:

# 导入 matplotlib 的 patheffects 模块，并简称为 path_effects
import matplotlib.patheffects as path_effects

# 在图上添加文本，位置为 (0.5, 0.5)，内容为 'Hello path effects world!'
# 使用 withSimplePatchShadow() 添加简单的填充阴影效果
text = plt.text(0.5, 0.5, 'Hello path effects world!',
                path_effects=[path_effects.withSimplePatchShadow()])

# 绘制一条蓝色的线，坐标点为 [0, 3, 2, 5]，线宽为 5，使用 SimpleLineShadow 添加简单的线阴影效果，然后使用 Normal 路径效果
plt.plot([0, 3, 2, 5], linewidth=5, color='blue',
         path_effects=[path_effects.SimpleLineShadow(),
                       path_effects.Normal()])

# 显示图形
plt.show()

# %%
# Notice the two approaches to setting the path effects in this example. The
# first uses the ``with*`` classes to include the desired functionality
# automatically followed with the "normal" effect, whereas the latter
# explicitly defines the two path effects to draw.
#
# Making an Artist stand out
# --------------------------
#
# One nice way of making artists visually stand out is to draw an outline in
# a bold color below the actual artist. The :class:`Stroke` path effect makes
# this a relatively simple task:

# 创建一个大小为 (7, 1) 的新图形对象
fig = plt.figure(figsize=(7, 1))

# 在图形上添加文本，位置为 (0.5, 0.5)，内容为 'This text stands out because of\nits black border.'
# 文本颜色为白色，水平对齐方式为居中，垂直对齐方式为居中，字体大小为 30
text = fig.text(0.5, 0.5, 'This text stands out because of\n'
                          'its black border.', color='white',
                          ha='center', va='center', size=30)

# 给文本对象设置路径效果，使用 Stroke 路径效果来绘制黑色边框，边框宽度为 3，然后使用 Normal 路径效果
text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                       path_effects.Normal()])

# 显示图形
plt.show()

# %%
# 创建一个大小为 8.5x1 英寸的新图形对象
fig = plt.figure(figsize=(8.5, 1))

# 在图形对象上添加一个文本对象，文本内容为 'Hatch shadow'，位于左上角 (0.02, 0.5)，字体大小为 75，粗细为 1000，垂直对齐方式为居中
t = fig.text(0.02, 0.5, 'Hatch shadow', fontsize=75, weight=1000, va='center')

# 设置文本对象的路径效果，包括两个 `PathPatchEffect`：
#   1. 偏移量为 (4, -4)，使用 'xxxx' 格子线填充，背景颜色为灰色
#   2. 边框颜色为白色，线宽为 1.1，背景颜色为黑色
t.set_path_effects([
    path_effects.PathPatchEffect(offset=(4, -4), hatch='xxxx', facecolor='gray'),
    path_effects.PathPatchEffect(edgecolor='white', linewidth=1.1, facecolor='black')])

# 显示图形对象
plt.show()
```