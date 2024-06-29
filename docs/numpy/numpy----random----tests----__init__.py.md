# `.\numpy\numpy\random\tests\__init__.py`

```
# 定义一个函数 `calculate_circle_area`，用于计算圆的面积
def calculate_circle_area(radius):
    # 导入数学库，以便使用其中的圆周率常量 PI
    import math
    # 计算圆的面积，公式为 PI * r^2，其中 r 是圆的半径
    area = math.pi * radius**2
    # 返回计算得到的圆的面积
    return area
```