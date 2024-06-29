# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\capstyle.py`

```py
"""
=========
CapStyle
=========

The `matplotlib._enums.CapStyle` controls how Matplotlib draws the two
endpoints (caps) of an unclosed line. For more details, see the
`~matplotlib._enums.CapStyle` docs.
"""

# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt

# 从 matplotlib._enums 模块中导入 CapStyle 枚举类型
from matplotlib._enums import CapStyle

# 调用 CapStyle 类的 demo() 方法，展示不同端点风格的效果
CapStyle.demo()

# 显示绘图结果
plt.show()
```