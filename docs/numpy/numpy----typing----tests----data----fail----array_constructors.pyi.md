# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\array_constructors.pyi`

```py
import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型提示模块

a: npt.NDArray[np.float64]  # 声明变量 a 是一个 NumPy 数组，包含浮点数
generator = (i for i in range(10))  # 创建一个生成器，产生范围在 0 到 9 的整数

np.require(a, requirements=1)  # E: No overload variant
np.require(a, requirements="TEST")  # E: incompatible type

np.zeros("test")  # E: incompatible type
np.zeros()  # E: require at least one argument

np.ones("test")  # E: incompatible type
np.ones()  # E: require at least one argument

np.array(0, float, True)  # E: No overload variant

np.linspace(None, 'bob')  # E: No overload variant
np.linspace(0, 2, num=10.0)  # E: No overload variant
np.linspace(0, 2, endpoint='True')  # E: No overload variant
np.linspace(0, 2, retstep=b'False')  # E: No overload variant
np.linspace(0, 2, dtype=0)  # E: No overload variant
np.linspace(0, 2, axis=None)  # E: No overload variant

np.logspace(None, 'bob')  # E: No overload variant
np.logspace(0, 2, base=None)  # E: No overload variant

np.geomspace(None, 'bob')  # E: No overload variant

np.stack(generator)  # E: No overload variant
np.hstack({1, 2})  # E: No overload variant
np.vstack(1)  # E: No overload variant

np.array([1], like=1)  # E: No overload variant
```