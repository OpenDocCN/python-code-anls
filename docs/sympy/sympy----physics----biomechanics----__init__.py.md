# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\__init__.py`

```
# SymPy 的生物力学扩展模块。

包含了与生物力学相关的构造，允许用户将使用 `sympy.physics.mechanics` 创建的多体模型扩展到涉及肌肉肌腱和激活动力学的生物力学或肌肉骨骼模型中。

从本地导入 ActivationBase、FirstOrderActivationDeGroote2016、ZerothOrderActivation 三个类。
from .activation import (
   ActivationBase,
   FirstOrderActivationDeGroote2016,
   ZerothOrderActivation,
)

从本地导入 CharacteristicCurveCollection、CharacteristicCurveFunction、以及几种肌肉肌腱相关的力长度和力速度特性模型。
from .curve import (
   CharacteristicCurveCollection,
   CharacteristicCurveFunction,
   FiberForceLengthActiveDeGroote2016,
   FiberForceLengthPassiveDeGroote2016,
   FiberForceLengthPassiveInverseDeGroote2016,
   FiberForceVelocityDeGroote2016,
   FiberForceVelocityInverseDeGroote2016,
   TendonForceLengthDeGroote2016,
   TendonForceLengthInverseDeGroote2016,
)

从本地导入 MusculotendonBase、MusculotendonDeGroote2016、MusculotendonFormulation 三个肌肉肌腱模型的基类和特定形式。
from .musculotendon import (
   MusculotendonBase,
   MusculotendonDeGroote2016,
   MusculotendonFormulation,
)

定义了一个公开的列表 `__all__`，包含了所有需要导出的符号和类名，用于模块级别的导入。
__all__ = [
   # 肌肉肌腱特性曲线函数
   'CharacteristicCurveCollection',
   'CharacteristicCurveFunction',
   'FiberForceLengthActiveDeGroote2016',
   'FiberForceLengthPassiveDeGroote2016',
   'FiberForceLengthPassiveInverseDeGroote2016',
   'FiberForceVelocityDeGroote2016',
   'FiberForceVelocityInverseDeGroote2016',
   'TendonForceLengthDeGroote2016',
   'TendonForceLengthInverseDeGroote2016',

   # 激活动力学类
   'ActivationBase',
   'FirstOrderActivationDeGroote2016',
   'ZerothOrderActivation',

   # 肌肉肌腱模型类
   'MusculotendonBase',
   'MusculotendonDeGroote2016',
   'MusculotendonFormulation',
]
```