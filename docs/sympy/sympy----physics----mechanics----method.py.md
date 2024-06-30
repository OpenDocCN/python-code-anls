# `D:\src\scipysrc\sympy\sympy\physics\mechanics\method.py`

```
# 导入ABC和abstractmethod，用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod

# 定义一个名为_Methods的抽象基类，继承自ABC
class _Methods(ABC):
    """Abstract Base Class for all methods."""
    
    # 定义抽象方法q，没有具体实现
    @abstractmethod
    def q(self):
        pass

    # 定义抽象方法u，没有具体实现
    @abstractmethod
    def u(self):
        pass

    # 定义抽象方法bodies，没有具体实现
    @abstractmethod
    def bodies(self):
        pass

    # 定义抽象方法loads，没有具体实现
    @abstractmethod
    def loads(self):
        pass

    # 定义抽象方法mass_matrix，没有具体实现
    @abstractmethod
    def mass_matrix(self):
        pass

    # 定义抽象方法forcing，没有具体实现
    @abstractmethod
    def forcing(self):
        pass

    # 定义抽象方法mass_matrix_full，没有具体实现
    @abstractmethod
    def mass_matrix_full(self):
        pass

    # 定义抽象方法forcing_full，没有具体实现
    @abstractmethod
    def forcing_full(self):
        pass

    # 定义一个非抽象方法_form_eoms，抛出NotImplementedError异常，要求子类实现此方法
    def _form_eoms(self):
        raise NotImplementedError("Subclasses must implement this.")
```