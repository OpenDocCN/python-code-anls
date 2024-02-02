# `arknights-mower\arknights_mower\utils\typealias.py`

```py
# 导入必要的类型注解模块
from typing import Dict, List, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 numpy 类型注解
from numpy.typing import NDArray

# 定义 Image 类型为 int8 类型的数组
Image = NDArray[np.int8]
# 定义 Pixel 类型为包含三个 int 类型的元组
Pixel = Tuple[int, int, int]

# 定义 GrayImage 类型为 int8 类型的数组
GrayImage = NDArray[np.int8]
# 定义 GrayPixel 类型为 int 类型
GrayPixel = int

# 定义 Recognizer 模块的类型
Range = Tuple[int, int]
Coordinate = Tuple[int, int]
Scope = Tuple[Coordinate, Coordinate]
Slice = Tuple[Range, Range]
Rectangle = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
Location = Union[Rectangle, Scope, Coordinate]

# 定义 Matcher 模块的类型
Hash = List[int]
Score = Tuple[float, float, float, float]

# 定义 Operation Plan 模块的类型
OpePlan = Tuple[str, int]

# 定义 BaseConstruct Plan 模块的类型
BasePlan = Dict[str, List[str]]

# 定义 Parameter 模块的类型
ParamArgs = List[str]
```