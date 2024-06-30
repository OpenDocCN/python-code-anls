# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_dimensionsystem.py`

```
from sympy.core.symbol import symbols
from sympy.matrices.dense import (Matrix, eye)
from sympy.physics.units.definitions.dimension_definitions import (
    action, current, length, mass, time,
    velocity)
from sympy.physics.units.dimensions import DimensionSystem

# 定义测试函数，用于测试 DimensionSystem 类的扩展功能
def test_extend():
    # 创建一个包含长度和时间维度的 DimensionSystem 对象
    ms = DimensionSystem((length, time), (velocity,))
    # 对该对象进行扩展，添加质量和动作维度
    mks = ms.extend((mass,), (action,))
    # 预期的结果 DimensionSystem 对象
    res = DimensionSystem((length, time, mass), (velocity, action))
    # 断言扩展后的 base_dims 和 derived_dims 与预期结果一致
    assert mks.base_dims == res.base_dims
    assert mks.derived_dims == res.derived_dims

# 定义测试函数，用于测试 DimensionSystem 类的 list_can_dims 方法
def test_list_dims():
    # 创建一个包含长度、时间和质量维度的 DimensionSystem 对象
    dimsys = DimensionSystem((length, time, mass))
    # 断言返回的规范维度列表与预期一致
    assert dimsys.list_can_dims == (length, mass, time)

# 定义测试函数，用于测试 DimensionSystem 类的 dim_can_vector 方法
def test_dim_can_vector():
    # 创建一个包含长度、质量和时间维度，速度和动作派生维度的 DimensionSystem 对象
    dimsys = DimensionSystem(
        [length, mass, time],
        [velocity, action],
        {
            velocity: {length: 1, time: -1}
        }
    )
    # 断言长度维度的规范向量与预期一致
    assert dimsys.dim_can_vector(length) == Matrix([1, 0, 0])
    # 断言速度维度的规范向量与预期一致
    assert dimsys.dim_can_vector(velocity) == Matrix([1, 0, -1])

    # 创建一个包含长度、速度和动作维度，质量和时间派生维度的 DimensionSystem 对象
    dimsys = DimensionSystem(
        (length, velocity, action),
        (mass, time),
        {
            time: {length: 1, velocity: -1}
        }
    )
    # 断言长度维度的规范向量与预期一致
    assert dimsys.dim_can_vector(length) == Matrix([0, 1, 0])
    # 断言速度维度的规范向量与预期一致
    assert dimsys.dim_can_vector(velocity) == Matrix([0, 0, 1])
    # 断言时间维度的规范向量与预期一致
    assert dimsys.dim_can_vector(time) == Matrix([0, 1, -1])

    # 创建一个包含长度、质量和时间维度，速度和动作派生维度的 DimensionSystem 对象
    dimsys = DimensionSystem(
        (length, mass, time),
        (velocity, action),
        {velocity: {length: 1, time: -1},
         action: {mass: 1, length: 2, time: -1}}
    )
    # 断言长度维度的规范向量与预期一致
    assert dimsys.dim_vector(length) == Matrix([1, 0, 0])
    # 断言速度维度的规范向量与预期一致
    assert dimsys.dim_vector(velocity) == Matrix([1, 0, -1])

# 定义测试函数，用于测试 DimensionSystem 类的 inv_can_transf_matrix 属性
def test_inv_can_transf_matrix():
    # 创建一个包含长度、质量和时间维度的 DimensionSystem 对象
    dimsys = DimensionSystem((length, mass, time))
    # 断言反规范转换矩阵与单位矩阵一致
    assert dimsys.inv_can_transf_matrix == eye(3)

# 定义测试函数，用于测试 DimensionSystem 类的 can_transf_matrix 属性
def test_can_transf_matrix():
    # 创建一个包含长度、质量和时间维度的 DimensionSystem 对象
    dimsys = DimensionSystem((length, mass, time))
    # 断言规范转换矩阵与单位矩阵一致
    assert dimsys.can_transf_matrix == eye(3)

    # 创建一个包含长度、速度和动作维度，速度与长度和时间的关系的 DimensionSystem 对象
    dimsys = DimensionSystem((length, velocity, action))
    # 断言规范转换矩阵与单位矩阵一致
    assert dimsys.can_transf_matrix == eye(3)

    # 创建一个包含长度和时间维度，速度与长度和时间的关系的 DimensionSystem 对象
    dimsys = DimensionSystem((length, time), (velocity,), {velocity: {length: 1, time: -1}})
    # 断言规范转换矩阵与单位矩阵一致
    assert dimsys.can_transf_matrix == eye(2)

# 定义测试函数，用于测试 DimensionSystem 类的 is_consistent 方法
def test_is_consistent():
    # 断言包含长度和时间维度的 DimensionSystem 对象是一致的
    assert DimensionSystem((length, time)).is_consistent is True

# 定义测试函数，用于测试 DimensionSystem 类的 print_dim_base 方法
def test_print_dim_base():
    # 创建一个包含长度、时间、质量和电流维度，动作派生维度及其维度关系的 DimensionSystem 对象
    mksa = DimensionSystem(
        (length, time, mass, current),
        (action,),
        {action: {mass: 1, length: 2, time: -1}})
    # 定义符号 L, M, T
    L, M, T = symbols("L M T")
    # 断言动作维度的基本维度表示与预期一致
    assert mksa.print_dim_base(action) == L**2*M/T

# 定义测试函数，用于测试 DimensionSystem 类的 dim 属性
def test_dim():
    # 创建一个包含长度、质量和时间维度，速度和动作派生维度及其维度关系的 DimensionSystem 对象
    dimsys = DimensionSystem(
        (length, mass, time),
        (velocity, action),
        {velocity: {length: 1, time: -1},
         action: {mass: 1, length: 2, time: -1}}
    )
    # 断言对象的维度为 3
    assert dimsys.dim == 3
```