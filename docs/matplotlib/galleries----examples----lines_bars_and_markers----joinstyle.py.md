# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\joinstyle.py`

```py
"""
=========
JoinStyle
=========

The `matplotlib._enums.JoinStyle` controls how Matplotlib draws the corners
where two different line segments meet. For more details, see the
`~matplotlib._enums.JoinStyle` docs.
"""

# 导入 Matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 从 matplotlib._enums 模块中导入 JoinStyle 枚举
from matplotlib._enums import JoinStyle

# 调用 JoinStyle 对象的 demo() 方法，展示不同连接样式的效果
JoinStyle.demo()

# 显示绘制的图形
plt.show()
```