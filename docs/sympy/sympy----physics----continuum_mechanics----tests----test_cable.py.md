# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\tests\test_cable.py`

```
# 引入来自Sympy库的连续力学模块中的Cable类
from sympy.physics.continuum_mechanics.cable import Cable
# 从Sympy核心符号模块中引入Symbol符号
from sympy.core.symbol import Symbol

# 定义测试函数test_cable
def test_cable():
    # 创建一个名为c的Cable对象，指定其支持点和长度
    c = Cable(('A', 0, 10), ('B', 10, 10))
    # 断言支持点字典的初始状态
    assert c.supports == {'A': [0, 10], 'B': [10, 10]}
    # 断言左侧支持点的初始状态
    assert c.left_support == [0, 10]
    # 断言右侧支持点的初始状态
    assert c.right_support == [10, 10]
    # 断言加载情况的初始状态
    assert c.loads == {'distributed': {}, 'point_load': {}}
    # 断言加载位置的初始状态
    assert c.loads_position == {}
    # 断言长度的初始状态
    assert c.length == 0
    # 断言反力的初始状态，使用Symbol符号进行表示
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # 对change_support方法进行测试
    c.change_support('A', ('C', 12, 3))
    # 断言支持点变更后的状态
    assert c.supports == {'B': [10, 10], 'C': [12, 3]}
    # 断言左侧支持点变更后的状态
    assert c.left_support == [10, 10]
    # 断言右侧支持点变更后的状态
    assert c.right_support == [12, 3]
    # 断言反力变更后的状态，增加了新的支持点C
    assert c.reaction_loads == {Symbol("R_B_x"): 0, Symbol("R_B_y"): 0, Symbol("R_C_x"): 0, Symbol("R_C_y"): 0}

    # 再次调用change_support方法进行测试
    c.change_support('C', ('A', 0, 10))

    # 对apply_load方法应用点载荷进行测试
    c.apply_load(-1, ('X', 2, 5, 3, 30))
    c.apply_load(-1, ('Y', 5, 8, 5, 60))
    # 断言加载情况，包括分布载荷和点载荷的状态
    assert c.loads == {'distributed': {}, 'point_load': {'X': [3, 30], 'Y': [5, 60]}}
    # 断言加载位置的状态
    assert c.loads_position == {'X': [2, 5], 'Y': [5, 8]}
    # 断言长度的初始状态
    assert c.length == 0
    # 断言反力的初始状态，使用Symbol符号进行表示
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # 对remove_loads方法进行测试
    c.remove_loads('X')
    # 断言加载情况，点载荷中移除了X
    assert c.loads == {'distributed': {}, 'point_load': {'Y': [5, 60]}}
    # 断言加载位置中移除了X
    assert c.loads_position == {'Y': [5, 8]}
    # 断言长度的初始状态
    assert c.length == 0
    # 断言反力的初始状态，使用Symbol符号进行表示
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    c.remove_loads('Y')

    # 对apply_load方法应用分布载荷进行测试
    c.apply_load(0, ('Z', 9))
    # 断言加载情况，包括分布载荷的状态
    assert c.loads == {'distributed': {'Z': 9}, 'point_load': {}}
    # 断言加载位置的状态
    assert c.loads_position == {}
    # 断言长度的初始状态
    assert c.length == 0
    # 断言反力的初始状态，使用Symbol符号进行表示
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # 对apply_length方法进行测试
    c.apply_length(20)
    # 断言长度的状态
    assert c.length == 20

    del c
    # 对solve方法进行测试
    # 对于点载荷的情况进行测试
    c = Cable(("A", 0, 10), ("B", 5.5, 8))
    c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
    c.apply_load(-1, ('X', 4, 6, 8, 270))
    c.solve()
    # 断言张力的计算结果，使用Symbol符号进行表示
    #assert c.tension == {Symbol("Z_X"): 4.79150773600774, Symbol("X_B"): 6.78571428571429, Symbol("A_Z"): 6.89488895397307}
    assert abs(c.tension[Symbol("A_Z")] - 6.89488895397307) < 10e-12
    assert abs(c.tension[Symbol("Z_X")] - 4.79150773600774) < 10e-12
    assert abs(c.tension[Symbol("X_B")] - 6.78571428571429) < 10e-12
    # 断言反力的计算结果，使用Symbol符号进行表示
    #assert c.reaction_loads == {Symbol("R_A_x"): -4.06504065040650, Symbol("R_A_y"): 5.56910569105691, Symbol("R_B_x"): 4.06504065040650, Symbol("R_B_y"): 5.43089430894309}
    assert abs(c.reaction_loads[Symbol("R_A_x")] + 4.06504065040650) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_A_y")] - 5.56910569105691) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 4.06504065040650) < 10e-12
    # 断言：验证 c.reaction_loads 字典中的 Symbol("R_B_y") 对应的值是否接近 5.43089430894309
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 5.43089430894309) < 10e-12
    # 断言：验证 c.length 属性是否接近 8.25609584845190
    assert abs(c.length - 8.25609584845190) < 10e-12

    # 删除变量 c，释放内存空间
    del c

    # 测试 solve 方法
    # 为 c 对象创建一个 Cable 类的实例，定义了两个端点 ("A", 0, 40) 和 ("B", 100, 20)
    c = Cable(("A", 0, 40), ("B", 100, 20))
    # 在端点 0 处施加均布载荷 ("X", 850)
    c.apply_load(0, ("X", 850))
    # 解算力学系统，参数为力的比例系数和初始位移
    c.solve(58.58, 0)

    # 断言：验证在特定点处的拉力是否接近给定值
    assert abs(c.tension_at(0) - 61717.4130533677) < 10e-11
    assert abs(c.tension_at(40) - 39738.0809048449) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_x")] - 36465.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_y")] + 49793.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 44399.9537590861) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 42868.2071025955) < 10e-11
```