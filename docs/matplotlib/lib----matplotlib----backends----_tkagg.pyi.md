# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_tkagg.pyi`

```py
import numpy as np
from numpy.typing import NDArray

# 定义两个全局变量，表示图片叠加的不同合成规则
TK_PHOTO_COMPOSITE_OVERLAY: int
TK_PHOTO_COMPOSITE_SET: int

# 定义函数 blit，用于在给定的照片上执行图像合成操作
def blit(
    interp: int,  # 插值方法的选择参数
    photo_name: str,  # 图片的名称
    data: NDArray[np.uint8],  # 包含图片数据的 NumPy 数组
    comp_rule: int,  # 合成规则，指定如何将数据合成到图片上
    offset: tuple[int, int, int, int],  # 偏移量，用于指定数据在图片上的位置
    bbox: tuple[int, int, int, int],  # 边界框，用于限定数据在图片上的操作区域
) -> None: ...
    # 函数主体未定义，这里用省略号表示

# 定义函数 enable_dpi_awareness，用于启用 DPI 感知功能
def enable_dpi_awareness(frame_handle: int, interp: int) -> bool | None: ...
    # 函数主体未定义，这里用省略号表示


这段代码定义了两个全局变量和两个函数声明，但函数体并没有给出。
```