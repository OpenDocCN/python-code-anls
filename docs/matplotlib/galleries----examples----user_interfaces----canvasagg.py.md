# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\canvasagg.py`

```py
# 导入所需模块
from PIL import Image  # 导入PIL库中的Image模块
import numpy as np  # 导入numpy库，并使用np作为别名
from matplotlib.backends.backend_agg import FigureCanvasAgg  # 从matplotlib中导入FigureCanvasAgg类
from matplotlib.figure import Figure  # 从matplotlib中导入Figure类

# 创建一个Figure对象，设置其大小为5x4英寸，分辨率为100dpi
fig = Figure(figsize=(5, 4), dpi=100)
# 创建一个与Figure对象关联的Agg后端的Canvas对象
canvas = FigureCanvasAgg(fig)

# 在Figure对象上添加一个子图
ax = fig.add_subplot()
ax.plot([1, 2, 3])  # 在子图上绘制简单的折线图

# 选项1：将Figure对象保存为PNG格式的图像文件
fig.savefig("test.png")

# 选项2：获取渲染缓冲区的内存视图，并将其转换为numpy数组
canvas.draw()
rgba = np.asarray(canvas.buffer_rgba())
# 将numpy数组转换为PIL Image对象
im = Image.fromarray(rgba)
# 将PIL Image对象保存为BMP格式的图像文件
im.save("test.bmp")

# 如果需要使用ImageMagick的`display`工具显示图像，请取消下面一行的注释
# im.show()
```