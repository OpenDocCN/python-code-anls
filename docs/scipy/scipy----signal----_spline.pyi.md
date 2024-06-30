# `D:\src\scipysrc\scipy\scipy\signal\_spline.pyi`

```
# 导入 numpy 库，并引入 NDArray 类型定义
import numpy as np
from numpy.typing import NDArray

# 定义 FloatingArray 类型为 np.float32 或 np.float64 的数组类型
FloatingArray = NDArray[np.float32] | NDArray[np.float64]
# 定义 ComplexArray 类型为 np.complex64 或 np.complex128 的数组类型
ComplexArray = NDArray[np.complex64] | NDArray[np.complex128]
# 定义 FloatingComplexArray 类型为 FloatingArray 或 ComplexArray 的数组类型
FloatingComplexArray = FloatingArray | ComplexArray


# 定义 symiirorder1_ic 函数，接受一个 FloatingComplexArray 类型的信号，
# 以及一个 float 类型的 c0、z1 和 precision 参数，返回一个 FloatingComplexArray 类型的结果
def symiirorder1_ic(signal: FloatingComplexArray,
                    c0: float,
                    z1: float,
                    precision: float) -> FloatingComplexArray:
    ...


# 定义 symiirorder2_ic_fwd 函数，接受一个 FloatingArray 类型的信号，
# 以及一个 float 类型的 r、omega 和 precision 参数，返回一个 FloatingArray 类型的结果
def symiirorder2_ic_fwd(signal: FloatingArray,
                        r: float,
                        omega: float,
                        precision: float) -> FloatingArray:
    ...


# 定义 symiirorder2_ic_bwd 函数，接受一个 FloatingArray 类型的信号，
# 以及一个 float 类型的 r、omega 和 precision 参数，返回一个 FloatingArray 类型的结果
def symiirorder2_ic_bwd(signal: FloatingArray,
                        r: float,
                        omega: float,
                        precision: float) -> FloatingArray:
    ...


# 定义 sepfir2d 函数，接受一个 FloatingComplexArray 类型的 input 参数，
# 以及两个 FloatingComplexArray 类型的 hrow 和 hcol 参数，返回一个 FloatingComplexArray 类型的结果
def sepfir2d(input: FloatingComplexArray,
             hrow: FloatingComplexArray,
             hcol: FloatingComplexArray) -> FloatingComplexArray:
    ...
```