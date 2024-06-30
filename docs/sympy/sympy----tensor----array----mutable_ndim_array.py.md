# `D:\src\scipysrc\sympy\sympy\tensor\array\mutable_ndim_array.py`

```
# 从 sympy 库中导入 NDimArray 类
from sympy.tensor.array.ndim_array import NDimArray

# 定义 MutableNDimArray 类，继承自 NDimArray 类
class MutableNDimArray(NDimArray):

    # 定义 as_immutable 方法，但只是声明，未实现具体功能，需要在子类中具体实现
    def as_immutable(self):
        # 抛出未实现的抽象方法异常，提醒子类必须实现该方法
        raise NotImplementedError("abstract method")

    # 定义 as_mutable 方法，返回对象自身，表明对象是可变的
    def as_mutable(self):
        return self

    # 定义 _sympy_ 方法，返回对象调用 as_immutable 方法的结果
    def _sympy_(self):
        return self.as_immutable()
```