# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\lib_polynomial.pyi`

```py
import numpy as np  # 导入NumPy库，用于科学计算
import numpy.typing as npt  # 导入NumPy的类型标注模块，用于类型注解

AR_f8: npt.NDArray[np.float64]  # AR_f8是一个NumPy数组，包含np.float64类型的元素
AR_c16: npt.NDArray[np.complex128]  # AR_c16是一个NumPy数组，包含np.complex128类型的元素
AR_O: npt.NDArray[np.object_]  # AR_O是一个NumPy数组，包含np.object_类型的元素
AR_U: npt.NDArray[np.str_]  # AR_U是一个NumPy数组，包含np.str_类型的元素

poly_obj: np.poly1d  # poly_obj是一个NumPy多项式对象

np.polymul(AR_f8, AR_U)  # E: 不兼容的类型错误，尝试对AR_f8和AR_U进行多项式乘法
np.polydiv(AR_f8, AR_U)  # E: 不兼容的类型错误，尝试对AR_f8和AR_U进行多项式除法

5**poly_obj  # E: 没有匹配的重载变体，尝试对5和poly_obj进行乘方运算

np.polyint(AR_U)  # E: 不兼容的类型错误，尝试对AR_U进行多项式积分
np.polyint(AR_f8, m=1j)  # E: 没有匹配的重载变体，尝试对AR_f8进行多项式积分，使用复数1j作为参数

np.polyder(AR_U)  # E: 不兼容的类型错误，尝试对AR_U进行多项式求导
np.polyder(AR_f8, m=1j)  # E: 没有匹配的重载变体，尝试对AR_f8进行多项式求导，使用复数1j作为参数

np.polyfit(AR_O, AR_f8, 1)  # E: 不兼容的类型错误，尝试使用AR_O和AR_f8进行多项式拟合
np.polyfit(AR_f8, AR_f8, 1, rcond=1j)  # E: 没有匹配的重载变体，尝试使用AR_f8和AR_f8进行多项式拟合，使用复数rcond=1j作为参数
np.polyfit(AR_f8, AR_f8, 1, w=AR_c16)  # E: 不兼容的类型错误，尝试使用AR_f8和AR_f8进行多项式拟合，使用AR_c16作为权重参数
np.polyfit(AR_f8, AR_f8, 1, cov="bob")  # E: 没有匹配的重载变体，尝试使用AR_f8和AR_f8进行多项式拟合，使用字符串"bob"作为cov参数

np.polyval(AR_f8, AR_U)  # E: 不兼容的类型错误，尝试对AR_f8和AR_U进行多项式求值
np.polyadd(AR_f8, AR_U)  # E: 不兼容的类型错误，尝试对AR_f8和AR_U进行多项式加法
np.polysub(AR_f8, AR_U)  # E: 不兼容的类型错误，尝试对AR_f8和AR_U进行多项式减法
```