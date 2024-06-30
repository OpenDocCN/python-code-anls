# `D:\src\scipysrc\sympy\sympy\physics\mechanics\body_base.py`

```
from abc import ABC, abstractmethod
from sympy import Symbol, sympify
from sympy.physics.vector import Point

__all__ = ['BodyBase']

class BodyBase(ABC):
    """Abstract class for body type objects."""

    def __init__(self, name, masscenter=None, mass=None):
        # 如果名称不是字符串类型，则引发类型错误异常
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        # 如果未提供质量，默认创建一个以名称为后缀的符号对象
        if mass is None:
            mass = Symbol(f'{name}_mass')
        # 如果未提供质心，默认创建一个以名称为后缀的点对象
        if masscenter is None:
            masscenter = Point(f'{name}_masscenter')
        self.mass = mass
        self.masscenter = masscenter
        self.potential_energy = 0
        self.points = []

    def __str__(self):
        # 返回对象的名称字符串表示形式
        return self.name

    def __repr__(self):
        # 返回对象的详细字符串表示形式，包括名称、质心和质量
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, mass={repr(self.mass)})')

    @property
    def name(self):
        """The name of the body."""
        # 返回对象的名称
        return self._name

    @property
    def masscenter(self):
        """The body's center of mass."""
        # 返回对象的质心
        return self._masscenter

    @masscenter.setter
    def masscenter(self, point):
        # 设置对象的质心，确保传入的参数是一个 Point 对象
        if not isinstance(point, Point):
            raise TypeError("The body's center of mass must be a Point object.")
        self._masscenter = point

    @property
    def mass(self):
        """The body's mass."""
        # 返回对象的质量
        return self._mass

    @mass.setter
    def mass(self, mass):
        # 设置对象的质量，使用 sympify 将输入转换为 SymPy 符号对象
        self._mass = sympify(mass)

    @property
    def potential_energy(self):
        """The potential energy of the body."""
        # 返回对象的势能
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, scalar):
        # 设置对象的势能，使用 sympify 将输入转换为 SymPy 符号对象
        self._potential_energy = sympify(scalar)

    @abstractmethod
    def kinetic_energy(self, frame):
        # 抽象方法，需要子类实现，计算对象的动能

    @abstractmethod
    def linear_momentum(self, frame):
        # 抽象方法，需要子类实现，计算对象的线动量

    @abstractmethod
    def angular_momentum(self, point, frame):
        # 抽象方法，需要子类实现，计算对象相对于给定点的角动量

    @abstractmethod
    def parallel_axis(self, point, frame):
        # 抽象方法，需要子类实现，计算对象关于给定点的平行轴定理
```