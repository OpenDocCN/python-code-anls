# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\tests\test_truss.py`

```
from sympy.core.symbol import Symbol, symbols  # 导入 Symbol 类和 symbols 函数
from sympy.physics.continuum_mechanics.truss import Truss  # 导入 Truss 类
from sympy import sqrt  # 导入 sqrt 函数


def test_truss():  # 定义测试函数 test_truss
    A = Symbol('A')  # 创建符号 A
    B = Symbol('B')  # 创建符号 B
    C = Symbol('C')  # 创建符号 C
    AB, BC, AC = symbols('AB, BC, AC')  # 创建符号 AB, BC, AC
    P = Symbol('P')  # 创建符号 P

    t = Truss()  # 创建 Truss 类实例 t
    assert t.nodes == []  # 断言节点为空列表
    assert t.node_labels == []  # 断言节点标签为空列表
    assert t.node_positions == []  # 断言节点位置为空列表
    assert t.members == {}  # 断言连接件字典为空
    assert t.loads == {}  # 断言载荷字典为空
    assert t.supports == {}  # 断言支撑字典为空
    assert t.reaction_loads == {}  # 断言反力载荷字典为空
    assert t.internal_forces == {}  # 断言内力字典为空

    # testing the add_node method
    t.add_node((A, 0, 0), (B, 2, 2), (C, 3, 0))  # 调用 add_node 方法添加节点
    assert t.nodes == [(A, 0, 0), (B, 2, 2), (C, 3, 0)]  # 断言节点列表包含指定节点
    assert t.node_labels == [A, B, C]  # 断言节点标签列表包含指定标签
    assert t.node_positions == [(0, 0), (2, 2), (3, 0)]  # 断言节点位置列表包含指定位置
    assert t.loads == {}  # 断言载荷字典为空
    assert t.supports == {}  # 断言支撑字典为空
    assert t.reaction_loads == {}  # 断言反力载荷字典为空

    # testing the remove_node method
    t.remove_node(C)  # 调用 remove_node 方法移除节点 C
    assert t.nodes == [(A, 0, 0), (B, 2, 2)]  # 断言节点列表移除了节点 C
    assert t.node_labels == [A, B]  # 断言节点标签列表移除了节点 C 的标签
    assert t.node_positions == [(0, 0), (2, 2)]  # 断言节点位置列表移除了节点 C 的位置
    assert t.loads == {}  # 断言载荷字典为空
    assert t.supports == {}  # 断言支撑字典为空

    t.add_node((C, 3, 0))  # 再次调用 add_node 方法添加节点 C

    # testing the add_member method
    t.add_member((AB, A, B), (BC, B, C), (AC, A, C))  # 调用 add_member 方法添加连接件
    assert t.members == {AB: [A, B], BC: [B, C], AC: [A, C]}  # 断言连接件字典包含指定连接件
    assert t.internal_forces == {AB: 0, BC: 0, AC: 0}  # 断言内力字典初始化为零

    # testing the remove_member method
    t.remove_member(BC)  # 调用 remove_member 方法移除连接件 BC
    assert t.members == {AB: [A, B], AC: [A, C]}  # 断言连接件字典移除了连接件 BC
    assert t.internal_forces == {AB: 0, AC: 0}  # 断言内力字典保持不变

    t.add_member((BC, B, C))  # 再次调用 add_member 方法添加连接件 BC

    D, CD = symbols('D, CD')  # 创建符号 D 和 CD

    # testing the change_label methods
    t.change_node_label((B, D))  # 调用 change_node_label 方法改变节点 B 的标签为 D
    assert t.nodes == [(A, 0, 0), (D, 2, 2), (C, 3, 0)]  # 断言节点列表更新为新标签
    assert t.node_labels == [A, D, C]  # 断言节点标签列表更新为新标签
    assert t.loads == {}  # 断言载荷字典为空
    assert t.supports == {}  # 断言支撑字典为空
    assert t.members == {AB: [A, D], BC: [D, C], AC: [A, C]}  # 断言连接件字典更新为新标签

    t.change_member_label((BC, CD))  # 调用 change_member_label 方法改变连接件 BC 的标签为 CD
    assert t.members == {AB: [A, D], CD: [D, C], AC: [A, C]}  # 断言连接件字典更新为新标签
    assert t.internal_forces == {AB: 0, CD: 0, AC: 0}  # 断言内力字典保持不变

    # testing the apply_load method
    t.apply_load((A, P, 90), (A, P/4, 90), (A, 2*P,45), (D, P/2, 90))  # 调用 apply_load 方法添加载荷
    assert t.loads == {A: [[P, 90], [P/4, 90], [2*P, 45]], D: [[P/2, 90]]}  # 断言载荷字典包含指定载荷
    assert t.loads[A] == [[P, 90], [P/4, 90], [2*P, 45]]  # 断言节点 A 的载荷列表包含指定载荷

    # testing the remove_load method
    t.remove_load((A, P/4, 90))  # 调用 remove_load 方法移除载荷 (A, P/4, 90)
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90]]}  # 断言载荷字典移除了指定载荷
    assert t.loads[A] == [[P, 90], [2*P, 45]]  # 断言节点 A 的载荷列表移除了指定载荷

    # testing the apply_support method
    t.apply_support((A, "pinned"), (D, "roller"))  # 调用 apply_support 方法添加支撑
    assert t.supports == {A: 'pinned', D: 'roller'}  # 断言支撑字典包含指定支撑
    assert t.reaction_loads == {}  # 断言反力载荷字典为空
    assert t.loads == {A: [[P, 90], [2*P, 45], [Symbol('R_A_x'), 0], [Symbol('R_A_y'), 90]],  D: [[P/2, 90], [Symbol('R_D_y'), 90]]}  # 断言载荷字典包含反力载荷

    # testing the remove_support method
    t.remove_support(A)  # 调用 remove_support 方法移除支撑 A
    assert t.supports == {D: 'roller'}  # 断言支撑字典移除了支撑 A
    assert t.reaction_loads == {}  # 断言反力载荷字典为空
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90], [Symbol('R_D_y'), 90]]}  # 断言载荷字典保持不变

    t.apply_support((A, "pinned"))  # 再次调用 apply_support 方法添加支撑 A

    # testing the solve method
    # 调用结构体 `t` 的 solve 方法，求解结构的静力平衡
    t.solve()
    # 断言结构体 `t` 中支反力字典中的水平力 R_A_x 等于 -sqrt(2) 乘以 P
    assert t.reaction_loads['R_A_x'] == -sqrt(2)*P
    # 断言结构体 `t` 中支反力字典中的竖直力 R_A_y 等于 -sqrt(2) 乘以 P 减去 P
    assert t.reaction_loads['R_A_y'] == -sqrt(2)*P - P
    # 断言结构体 `t` 中支反力字典中的竖直力 R_D_y 等于 -P 的一半
    assert t.reaction_loads['R_D_y'] == -P/2
    # 断言结构体 `t` 中的内力字典中 AB 元素的值除以 P 等于 0
    assert t.internal_forces[AB]/P == 0
    # 断言结构体 `t` 中的内力字典中 CD 元素的值等于 0
    assert t.internal_forces[CD] == 0
    # 断言结构体 `t` 中的内力字典中 AC 元素的值等于 0
    assert t.internal_forces[AC] == 0
```