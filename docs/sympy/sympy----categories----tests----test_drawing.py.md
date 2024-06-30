# `D:\src\scipysrc\sympy\sympy\categories\tests\test_drawing.py`

```
from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
                              Diagram, XypicDiagramDrawer, xypic_draw_diagram)
from sympy.sets.sets import FiniteSet

def test_GrowableGrid():
    # 创建一个 1x2 的可增长网格对象
    grid = _GrowableGrid(1, 2)

    # 检查网格的尺寸
    assert grid.width == 1
    assert grid.height == 2

    # 检查元素初始化情况
    assert grid[0, 0] is None
    assert grid[1, 0] is None

    # 设置元素的值
    grid[0, 0] = 1
    grid[1, 0] = "two"

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"

    # 检查添加行操作
    grid.append_row()

    assert grid.width == 1
    assert grid.height == 3

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"
    assert grid[2, 0] is None

    # 检查添加列操作
    grid.append_column()
    assert grid.width == 2
    assert grid.height == 3

    assert grid[0, 0] == 1
    assert grid[1, 0] == "two"
    assert grid[2, 0] is None

    assert grid[0, 1] is None
    assert grid[1, 1] is None
    assert grid[2, 1] is None

    # 重新创建一个 1x2 的可增长网格对象，并设置初始值
    grid = _GrowableGrid(1, 2)
    grid[0, 0] = 1
    grid[1, 0] = "two"

    # 检查前置添加行操作
    grid.prepend_row()
    assert grid.width == 1
    assert grid.height == 3

    assert grid[0, 0] is None
    assert grid[1, 0] == 1
    assert grid[2, 0] == "two"

    # 检查前置添加列操作
    grid.prepend_column()
    assert grid.width == 2
    assert grid.height == 3

    assert grid[0, 0] is None
    assert grid[1, 0] is None
    assert grid[2, 0] is None

    assert grid[0, 1] is None
    assert grid[1, 1] == 1
    assert grid[2, 1] == "two"


def test_DiagramGrid():
    # 创建一些对象和态射
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(D, A, "h")
    k = NamedMorphism(D, B, "k")

    # 创建一个一态射的图示
    d = Diagram([f])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid.morphisms == {f: FiniteSet()}

    # 创建一个三角形图示
    d = Diagram([f, g], {g * f: "unique"})
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] == C
    assert grid[1, 1] is None
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(),
                              g * f: FiniteSet("unique")}

    # 创建一个带有"回路"态射的三角形图示
    l_A = NamedMorphism(A, A, "l_A")
    d = Diagram([f, g, l_A])
    grid = DiagramGrid(d)

    assert grid.width == 2
    assert grid.height == 2
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), l_A: FiniteSet()}
    # 创建一个简单的图表对象，包含对象 f, g, h, k
    d = Diagram([f, g, h, k])
    # 使用图表对象创建一个网格对象
    grid = DiagramGrid(d)

    # 断言网格的宽度为 3
    assert grid.width == 3
    # 断言网格的高度为 2
    assert grid.height == 2
    # 断言网格中特定位置的对象
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == D
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] is None
    # 断言网格中的所有 morphisms 都是空集合
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    # 断言网格对象转换为字符串的结果
    assert str(grid) == '[[Object("A"), Object("B"), Object("D")], ' \
        '[None, Object("C"), None]]'

    # 创建一个链状的 morphism 序列
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    k = NamedMorphism(D, E, "k")
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)

    # 断言网格的宽度为 3
    assert grid.width == 3
    # 断言网格的高度为 3
    assert grid.height == 3
    # 断言网格中特定位置的对象
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == C
    assert grid[1, 2] == D
    assert grid[2, 0] is None
    assert grid[2, 1] is None
    assert grid[2, 2] == E
    # 断言网格中的所有 morphisms 都是空集合
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    # 创建一个正方形的 morphism 序列
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, D, "g")
    h = NamedMorphism(A, C, "h")
    k = NamedMorphism(C, D, "k")
    d = Diagram([f, g, h, k])
    grid = DiagramGrid(d)

    # 断言网格的宽度为 2
    assert grid.width == 2
    # 断言网格的高度为 2
    assert grid.height == 2
    # 断言网格中特定位置的对象
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[1, 0] == C
    assert grid[1, 1] == D
    # 断言网格中的所有 morphisms 都是空集合
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              k: FiniteSet()}

    # 创建一个由于创建五引理测试时的拼写错误而产生的奇怪图表，
    # 但它却可以用来停止算法中的一个额外问题。
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    # 创建一系列 morphisms，应该在带撇的对象之间
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    j = NamedMorphism(A, B, "j")   # 此处应该是 A_ 到 B_ 的 morphism，可能是错误
    k = NamedMorphism(B, C, "k")   # 此处应该是 B_ 到 C_ 的 morphism，可能是错误
    l = NamedMorphism(C, D, "l")   # 此处应该是 C_ 到 D_ 的 morphism，可能是错误
    m = NamedMorphism(D, E, "m")   # 此处应该是 D_ 到 E_ 的 morphism，可能是错误

    o = NamedMorphism(A, A_, "o")
    p = NamedMorphism(B, B_, "p")
    q = NamedMorphism(C, C_, "q")
    r = NamedMorphism(D, D_, "r")
    s = NamedMorphism(E, E_, "s")

    d = Diagram([f, g, h, i, j, k, l, m, o, p, q, r, s])
    grid = DiagramGrid(d)

    # 断言网格的宽度为 3
    assert grid.width == 3
    # 断言网格的高度为 4
    assert grid.height == 4
    # 断言网格中特定位置的对象
    assert grid[0, 0] is None
    assert grid[0, 1] == A
    assert grid[0, 2] == A_
    assert grid[1, 0] == C
    assert grid[1, 1] == B
    assert grid[1, 2] == B_
    assert grid[2, 0] == C_
    assert grid[2, 1] == D
    assert grid[2, 2] == D_
    assert grid[3, 0] is None
    assert grid[3, 1] == E
    # Assert that the element at position (3, 2) in grid is equal to E_
    assert grid[3, 2] == E_

    # Initialize an empty dictionary morphisms
    morphisms = {}

    # Iterate over a list of morphisms [f, g, h, ..., r, s] and assign each morphism
    # as a key in morphisms dictionary with an empty FiniteSet as its value
    for m in [f, g, h, i, j, k, l, m, o, p, q, r, s]:
        morphisms[m] = FiniteSet()

    # Assert that grid.morphisms is equal to morphisms
    assert grid.morphisms == morphisms

    # Define objects A1 to A8 representing vertices of a cube
    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")
    A4 = Object("A4")
    A5 = Object("A5")
    A6 = Object("A6")
    A7 = Object("A7")
    A8 = Object("A8")

    # Define morphisms representing faces of the cube
    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A1, A3, "f2")
    f3 = NamedMorphism(A2, A4, "f3")
    f4 = NamedMorphism(A3, A4, "f3")

    # Define morphisms for the bottom face of the cube
    f5 = NamedMorphism(A5, A6, "f5")
    f6 = NamedMorphism(A5, A7, "f6")
    f7 = NamedMorphism(A6, A8, "f7")
    f8 = NamedMorphism(A7, A8, "f8")

    # Define remaining morphisms for the cube
    f9 = NamedMorphism(A1, A5, "f9")
    f10 = NamedMorphism(A2, A6, "f10")
    f11 = NamedMorphism(A3, A7, "f11")
    f12 = NamedMorphism(A4, A8, "f11")

    # Create a diagram d with all defined morphisms
    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])

    # Create a grid based on the diagram d
    grid = DiagramGrid(d)

    # Assert statements to check dimensions and elements of grid
    assert grid.width == 4
    assert grid.height == 3
    assert grid[0, 0] is None
    assert grid[0, 1] == A5
    assert grid[0, 2] == A6
    assert grid[0, 3] is None
    assert grid[1, 0] is None
    assert grid[1, 1] == A1
    assert grid[1, 2] == A2
    assert grid[1, 3] is None
    assert grid[2, 0] == A7
    assert grid[2, 1] == A3
    assert grid[2, 2] == A4
    assert grid[2, 3] == A8

    # Initialize an empty dictionary morphisms
    morphisms = {}

    # Iterate over a list of morphisms [f1, f2, ..., f12] and assign each morphism
    # as a key in morphisms dictionary with an empty FiniteSet as its value
    for m in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]:
        morphisms[m] = FiniteSet()

    # Assert that grid.morphisms is equal to morphisms
    assert grid.morphisms == morphisms

    # Define objects A, B, C, D, E representing vertices of a line diagram
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    # Define morphisms for the line diagram
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    # Create a diagram d with all defined morphisms
    d = Diagram([f, g, h, i])

    # Create a grid based on the diagram d with sequential layout
    grid = DiagramGrid(d, layout="sequential")

    # Assert statements to check dimensions and elements of grid
    assert grid.width == 5
    assert grid.height == 1
    assert grid[0, 0] == A
    assert grid[0, 1] == B
    assert grid[0, 2] == C
    assert grid[0, 3] == D
    assert grid[0, 4] == E

    # Assert that grid.morphisms is a dictionary with keys as morphisms f, g, h, i
    # and values as empty FiniteSet objects
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              i: FiniteSet()}

    # Create a transposed version of the grid
    grid = DiagramGrid(d, layout="sequential", transpose=True)

    # Assert statements to check dimensions and elements of transposed grid
    assert grid.width == 1
    assert grid.height == 5
    assert grid[0, 0] == A
    assert grid[1, 0] == B
    assert grid[2, 0] == C
    assert grid[3, 0] == D
    assert grid[4, 0] == E

    # Assert that grid.morphisms is a dictionary with keys as morphisms f, g, h, i
    # and values as empty FiniteSet objects
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), h: FiniteSet(),
                              i: FiniteSet()}

    # Define morphisms for a pullback diagram
    m1 = NamedMorphism(A, B, "m1")
    m2 = NamedMorphism(A, C, "m2")
    s1 = NamedMorphism(B, D, "s1")
    s2 = NamedMorphism(C, D, "s2")
    f1 = NamedMorphism(E, B, "f1")
    f2 = NamedMorphism(E, C, "f2")
    g = NamedMorphism(E, A, "g")

    # Create a diagram d with all defined morphisms, and g labeled as unique
    d = Diagram([m1, m2, s1, s2, f1, f2], {g: "unique"})

    # Create a grid based on the diagram d
    grid = DiagramGrid(d)
    # 确保网格的宽度为3
    assert grid.width == 3
    # 确保网格的高度为2
    assert grid.height == 2
    # 确保网格位置 (0, 0) 上的元素为 A
    assert grid[0, 0] == A
    # 确保网格位置 (0, 1) 上的元素为 B
    assert grid[0, 1] == B
    # 确保网格位置 (0, 2) 上的元素为 E
    assert grid[0, 2] == E
    # 确保网格位置 (1, 0) 上的元素为 C
    assert grid[1, 0] == C
    # 确保网格位置 (1, 1) 上的元素为 D
    assert grid[1, 1] == D
    # 确保网格位置 (1, 2) 为空
    assert grid[1, 2] is None

    # 创建一个包含默认映射的字典
    morphisms = {g: FiniteSet("unique")}
    # 遍历映射列表，并将每个映射初始化为空的有限集合
    for m in [m1, m2, s1, s2, f1, f2]:
        morphisms[m] = FiniteSet()
    # 确保网格的映射等于初始化后的映射字典
    assert grid.morphisms == morphisms

    # 使用顺序布局创建一个 DiagramGrid 实例，用于压力测试
    grid = DiagramGrid(d, layout="sequential")

    # 确保网格的宽度为5
    assert grid.width == 5
    # 确保网格的高度为1
    assert grid.height == 1
    # 确保网格位置 (0, 0) 上的元素为 D
    assert grid[0, 0] == D
    # 确保网格位置 (0, 1) 上的元素为 B
    assert grid[0, 1] == B
    # 确保网格位置 (0, 2) 上的元素为 A
    assert grid[0, 2] == A
    # 确保网格位置 (0, 3) 上的元素为 C
    assert grid[0, 3] == C
    # 确保网格位置 (0, 4) 上的元素为 E
    assert grid[0, 4] == E
    # 确保网格的映射等于初始化后的映射字典
    assert grid.morphisms == morphisms

    # 使用对象分组创建一个 DiagramGrid 实例
    grid = DiagramGrid(d, groups=FiniteSet(E, FiniteSet(A, B, C, D)))

    # 确保网格的宽度为3
    assert grid.width == 3
    # 确保网格的高度为2
    assert grid.height == 2
    # 确保网格位置 (0, 0) 上的元素为 E
    assert grid[0, 0] == E
    # 确保网格位置 (0, 1) 上的元素为 A
    assert grid[0, 1] == A
    # 确保网格位置 (0, 2) 上的元素为 B
    assert grid[0, 2] == B
    # 确保网格位置 (1, 0) 为空
    assert grid[1, 0] is None
    # 确保网格位置 (1, 1) 上的元素为 C
    assert grid[1, 1] == C
    # 确保网格位置 (1, 2) 上的元素为 D
    assert grid[1, 2] == D
    # 确保网格的映射等于初始化后的映射字典
    assert grid.morphisms == morphisms

    # 创建五引理的对象
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    # 创建命名的态射
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    j = NamedMorphism(A_, B_, "j")
    k = NamedMorphism(B_, C_, "k")
    l = NamedMorphism(C_, D_, "l")
    m = NamedMorphism(D_, E_, "m")

    o = NamedMorphism(A, A_, "o")
    p = NamedMorphism(B, B_, "p")
    q = NamedMorphism(C, C_, "q")
    r = NamedMorphism(D, D_, "r")
    s = NamedMorphism(E, E_, "s")

    # 创建一个包含所有命名态射的 Diagram 对象
    d = Diagram([f, g, h, i, j, k, l, m, o, p, q, r, s])
    # 创建一个基于 Diagram 对象的 DiagramGrid 实例
    grid = DiagramGrid(d)

    # 确保网格的宽度为5
    assert grid.width == 5
    # 确保网格的高度为3
    assert grid.height == 3
    # 确保网格位置 (0, 0) 为空
    assert grid[0, 0] is None
    # 确保网格位置 (0, 1) 上的元素为 A
    assert grid[0, 1] == A
    # 确保网格位置 (0, 2) 上的元素为 A_
    assert grid[0, 2] == A_
    # 确保网格位置 (0, 3) 为空
    assert grid[0, 3] is None
    # 确保网格位置 (0, 4) 为空
    assert grid[0, 4] is None
    # 确保网格位置 (1, 0) 上的元素为 C
    assert grid[1, 0] == C
    # 确保网格位置 (1, 1) 上的元素为 B
    assert grid[1, 1] == B
    # 确保网格位置 (1, 2) 上的元素为 B_
    assert grid[1, 2] == B_
    # 确保网格位置 (1, 3) 上的元素为 C_
    assert grid[1, 3] == C_
    # 确保网格位置 (1, 4) 为空
    assert grid[1, 4] is None
    # 确保网格位置 (2, 0) 上的元素为 D
    assert grid[2, 0] == D
    # 确保网格位置 (2, 1) 上的元素为 E
    assert grid[2, 1] == E
    # 确保网格位置 (2, 2) 为空
    assert grid[2, 2] is None
    # 确保网格位置 (2, 3) 上的元素为 D_
    assert grid[2, 3] == D_
    # 确保网格位置 (2, 4) 上的元素为 E_
    assert grid[2, 4] == E_

    # 创建一个空的映射字典
    morphisms = {}
    # 遍历所有命名态射，并将每个初始化为空的有限集合加入映射字典
    for m in [f, g, h, i, j, k, l, m, o, p, q, r, s]:
        morphisms[m] = FiniteSet()
    # 确保网格的映射等于初始化后的映射字典
    assert grid.morphisms == morphisms

    # 使用对象分组创建一个 DiagramGrid 实例
    grid = DiagramGrid(d, FiniteSet(
        FiniteSet(A, B, C, D, E), FiniteSet(A_, B_, C_, D_, E_)))

    # 确保网格的宽度为6
    assert grid.width == 6
    # 确保网
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 4] == C_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 5] == D_
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 0] is None
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 1] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[2, 2] == E
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 3] is None
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 4] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[2, 5] == E_
    
    # 断言：验证网格对象的 morphisms 属性是否与预期值相等
    assert grid.morphisms == morphisms
    
    # 测试五引理，使用对象分组和混合容器表示组
    grid = DiagramGrid(d, [(A, B, C, D, E), {A_, B_, C_, D_, E_}])
    
    # 断言：验证网格对象的宽度是否等于预期值
    assert grid.width == 6
    
    # 断言：验证网格对象的高度是否等于预期值
    assert grid.height == 3
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 0] == A
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 1] == B
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[0, 2] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 3] == A_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 4] == B_
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[0, 5] is None
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[1, 0] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 1] == C
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 2] == D
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[1, 3] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 4] == C_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 5] == D_
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 0] is None
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 1] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[2, 2] == E
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 3] is None
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[2, 4] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[2, 5] == E_
    
    # 断言：验证网格对象的 morphisms 属性是否与预期值相等
    assert grid.morphisms == morphisms
    
    # 测试五引理，使用对象分组和提示
    grid = DiagramGrid(d, {
        FiniteSet(A, B, C, D, E): {"layout": "sequential",
                                   "transpose": True},
        FiniteSet(A_, B_, C_, D_, E_): {"layout": "sequential",
                                        "transpose": True}},
        transpose=True)
    
    # 断言：验证网格对象的宽度是否等于预期值
    assert grid.width == 5
    
    # 断言：验证网格对象的高度是否等于预期值
    assert grid.height == 2
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 0] == A
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 1] == B
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 2] == C
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 3] == D
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 4] == E
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 0] == A_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 1] == B_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 2] == C_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 3] == D_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 4] == E_
    
    # 断言：验证网格对象的 morphisms 属性是否与预期值相等
    assert grid.morphisms == morphisms
    
    # 一个包含两个三角形的不连通图
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    f_ = NamedMorphism(A_, B_, "f")
    g_ = NamedMorphism(B_, C_, "g")
    d = Diagram([f, g, f_, g_], {g * f: "unique", g_ * f_: "unique"})
    grid = DiagramGrid(d)
    
    # 断言：验证网格对象的宽度是否等于预期值
    assert grid.width == 4
    
    # 断言：验证网格对象的高度是否等于预期值
    assert grid.height == 2
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 0] == A
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 1] == B
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 2] == A_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[0, 3] == B_
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 0] == C
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[1, 1] is None
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert grid[1, 2] == C_
    
    # 断言：验证网格中特定位置的值是否为 None
    assert grid[1, 3] is None
    
    # 断言：验证网格对象的 morphisms 属性是否与预期值相等
    assert grid.morphisms == {f: FiniteSet(), g: FiniteSet(), f_: FiniteSet(),
                              g_: FiniteSet(), g * f: FiniteSet("unique"),
                              g_ * f_: FiniteSet("unique")}
    
    # 一个包含两个态射的不连通图
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(C, D, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)
    
    # 断言：验证网格对象的宽度是否等于预期值
    assert grid.width == 4
    
    # 断言：验证网格对象的高度是否等于预期值
    assert grid.height == 1
    
    # 断言：验证网格中特定位置的值是否等于预期值
    assert
    # 创建一个包含单个对象 f 的图表 Diagram 对象
    d = Diagram([f])
    # 使用 DiagramGrid 类创建一个网格 grid，用于显示图表 d 的布局
    grid = DiagramGrid(d)

    # 断言网格的宽度为 1
    assert grid.width == 1
    # 断言网格的高度为 1
    assert grid.height == 1
    # 断言网格位置 (0, 0) 处的对象为 A
    assert grid[0, 0] == A

    # 测试一个由两个不连接的对象组成的图表
    # 创建一个命名态射 g，连接 B 到 B，命名为 "g"
    g = NamedMorphism(B, B, "g")
    # 创建一个包含对象 f 和 g 的新图表 Diagram 对象
    d = Diagram([f, g])
    # 使用 DiagramGrid 类创建一个网格 grid，用于显示新图表 d 的布局
    grid = DiagramGrid(d)

    # 断言网格的宽度为 2
    assert grid.width == 2
    # 断言网格的高度为 1
    assert grid.height == 1
    # 断言网格位置 (0, 0) 处的对象为 A
    assert grid[0, 0] == A
    # 断言网格位置 (0, 1) 处的对象为 B
    assert grid[0, 1] == B
def test_DiagramGrid_pseudopod():
    # Test a diagram in which even growing a pseudopod does not
    # eventually help.
    # 创建对象 A, B, C, D, E, F, A', B', C', D', E' 分别表示不同的对象
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")
    F = Object("F")
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")

    # 创建命名的态射对象 f1 到 f10，将它们添加到列表中
    f1 = NamedMorphism(A, B, "f1")
    f2 = NamedMorphism(A, C, "f2")
    f3 = NamedMorphism(A, D, "f3")
    f4 = NamedMorphism(A, E, "f4")
    f5 = NamedMorphism(A, A_, "f5")
    f6 = NamedMorphism(A, B_, "f6")
    f7 = NamedMorphism(A, C_, "f7")
    f8 = NamedMorphism(A, D_, "f8")
    f9 = NamedMorphism(A, E_, "f9")
    f10 = NamedMorphism(A, F, "f10")
    
    # 创建包含所有命名态射的图表对象
    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
    
    # 创建图表网格对象，基于先前创建的图表对象
    grid = DiagramGrid(d)

    # 断言测试网格的宽度和高度是否符合预期
    assert grid.width == 5
    assert grid.height == 3
    
    # 断言网格中特定位置的对象是否正确
    assert grid[0, 0] == E
    assert grid[0, 1] == C
    assert grid[0, 2] == C_
    assert grid[0, 3] == E_
    assert grid[0, 4] == F
    assert grid[1, 0] == D
    assert grid[1, 1] == A
    assert grid[1, 2] == A_
    assert grid[1, 3] is None
    assert grid[1, 4] is None
    assert grid[2, 0] == D_
    assert grid[2, 1] == B
    assert grid[2, 2] == B_
    assert grid[2, 3] is None
    assert grid[2, 4] is None

    # 创建空字典用于存储态射对象
    morphisms = {}
    # 将所有命名态射添加到空集合中
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]:
        morphisms[f] = FiniteSet()
    
    # 断言网格对象的态射属性是否符合预期
    assert grid.morphisms == morphisms


def test_ArrowStringDescription():
    # 创建箭头字符串描述对象，测试其不同参数配置下的字符串表示
    astr = ArrowStringDescription("cm", "", None, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "^", 12, "", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar@/^12cm/[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "", "d", "r", "_", "f")
    assert str(astr) == "\\ar[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    assert str(astr) == "\\ar@(r,u)[dr]_{f}"

    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    assert str(astr) == "\\ar@(r,u)[dr]_{f}"

    # 修改箭头风格属性并断言其字符串表示是否符合预期
    astr = ArrowStringDescription("cm", "", 12, "r", "u", "d", "r", "_", "f")
    astr.arrow_style = "{-->}"
    assert str(astr) == "\\ar@(r,u)@{-->}[dr]_{f}"

    astr = ArrowStringDescription("cm", "_", 12, "", "", "d", "r", "_", "f")
    astr.arrow_style = "{-->}"
    assert str(astr) == "\\ar@/_12cm/@{-->}[dr]_{f}"


def test_XypicDiagramDrawer_line():
    # 创建线性图表对象
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    # 创建命名的态射对象并添加到列表中
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")
    
    # 创建线性图表对象
    d = Diagram([f, g, h, i])
    # 创建具有顺序布局的图表网格对象
    grid = DiagramGrid(d, layout="sequential")
    # 创建 XypicDiagramDrawer 对象用于绘制图表
    drawer = XypicDiagramDrawer()
    # 使用断言验证通过给定的数据结构 `d` 和 `grid` 绘制的结果是否符合预期
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]^{f} & B \\ar[r]^{g} & C \\ar[r]^{h} & D \\ar[r]^{i} & E \n" \
        "}\n"

    # 创建一个新的 `DiagramGrid` 对象，使用 `d` 并进行转置布局
    grid = DiagramGrid(d, layout="sequential", transpose=True)
    # 创建一个 `XypicDiagramDrawer` 对象
    drawer = XypicDiagramDrawer()
    # 使用断言验证通过给定的数据结构 `d` 和转置后的 `grid` 绘制的结果是否符合预期
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\\\\n" \
        "B \\ar[d]^{g} \\\\\n" \
        "C \\ar[d]^{h} \\\\\n" \
        "D \\ar[d]^{i} \\\\\n" \
        "E \n" \
        "}\n"
def test_XypicDiagramDrawer_cube():
    # 定义立方体的顶面顶点
    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")
    A4 = Object("A4")

    # 定义立方体的侧面顶点
    A5 = Object("A5")
    A6 = Object("A6")
    A7 = Object("A7")
    A8 = Object("A8")

    # 定义立方体顶面的箭头（morphisms）
    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A1, A3, "f2")
    f3 = NamedMorphism(A2, A4, "f3")
    f4 = NamedMorphism(A3, A4, "f3")  # f4 似乎应为 NamedMorphism(A3, A4, "f4")

    # 定义立方体底面的箭头（morphisms）
    f5 = NamedMorphism(A5, A6, "f5")
    f6 = NamedMorphism(A5, A7, "f6")
    f7 = NamedMorphism(A6, A8, "f7")
    f8 = NamedMorphism(A7, A8, "f8")
    # 创建一个从对象 A7 到 A8 的命名态射 f8

    # The remaining morphisms.
    f9 = NamedMorphism(A1, A5, "f9")
    # 创建一个从对象 A1 到 A5 的命名态射 f9
    f10 = NamedMorphism(A2, A6, "f10")
    # 创建一个从对象 A2 到 A6 的命名态射 f10
    f11 = NamedMorphism(A3, A7, "f11")
    # 创建一个从对象 A3 到 A7 的命名态射 f11
    f12 = NamedMorphism(A4, A8, "f11")
    # 创建一个从对象 A4 到 A8 的命名态射 f12，注意到名称应为 f12 而非 f11

    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    # 创建一个包含给定命名态射列表的图表对象 d

    grid = DiagramGrid(d)
    # 创建一个基于图表 d 的默认网格布局对象 grid

    drawer = XypicDiagramDrawer()
    # 创建一个 XypicDiagramDrawer 类型的图表绘制器对象 drawer

    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "& A_{5} \\ar[r]^{f_{5}} \\ar[ldd]_{f_{6}} & A_{6} \\ar[rdd]^{f_{7}} " \
        "& \\\\\n" \
        "& A_{1} \\ar[r]^{f_{1}} \\ar[d]^{f_{2}} \\ar[u]^{f_{9}} & A_{2} " \
        "\\ar[d]^{f_{3}} \\ar[u]_{f_{10}} & \\\\\n" \
        "A_{7} \\ar@/_3mm/[rrr]_{f_{8}} & A_{3} \\ar[r]^{f_{3}} \\ar[l]_{f_{11}} " \
        "& A_{4} \\ar[r]^{f_{11}} & A_{8} \n" \
        "}\n"
    # 绘制图表 d 和网格 grid，并检查绘制结果是否与预期的 LaTeX 字符串匹配

    # The same diagram, transposed.
    grid = DiagramGrid(d, transpose=True)
    # 使用 transpose=True 参数创建一个转置的图表网格布局对象 grid

    drawer = XypicDiagramDrawer()
    # 创建一个新的图表绘制器对象 drawer

    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "& & A_{7} \\ar@/^3mm/[ddd]^{f_{8}} \\\\\n" \
        "A_{5} \\ar[d]_{f_{5}} \\ar[rru]^{f_{6}} & A_{1} \\ar[d]^{f_{1}} " \
        "\\ar[r]^{f_{2}} \\ar[l]^{f_{9}} & A_{3} \\ar[d]_{f_{3}} " \
        "\\ar[u]^{f_{11}} \\\\\n" \
        "A_{6} \\ar[rrd]_{f_{7}} & A_{2} \\ar[r]^{f_{3}} \\ar[l]^{f_{10}} " \
        "& A_{4} \\ar[d]_{f_{11}} \\\\\n" \
        "& & A_{8} \n" \
        "}\n"
    # 绘制转置后的图表 d 和网格 grid，并检查绘制结果是否与预期的 LaTeX 字符串匹配
def test_XypicDiagramDrawer_curved_and_loops():
    # 定义四个对象 A, B, C, D
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")

    # 定义四个带有名称的态射 f, g, h, k
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(D, A, "h")
    k = NamedMorphism(D, B, "k")

    # 创建一个包含 f, g, h, k 的图表 d
    d = Diagram([f, g, h, k])

    # 根据图表 d 创建一个网格 grid
    grid = DiagramGrid(d)

    # 创建一个 XypicDiagramDrawer 对象 drawer
    drawer = XypicDiagramDrawer()

    # 断言使用 drawer 绘制图表 d 在网格 grid 中的结果
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_3mm/[ll]_{h} \\\\\n" \
        "& C & \n" \
        "}\n"

    # 将图表 d 转置后再次绘制
    grid = DiagramGrid(d, transpose=True)

    # 断言使用 drawer 绘制转置后的图表 d 在网格 grid 中的结果
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n" \
        "}\n"

    # 将图表 d 放大并旋转后再次绘制
    assert drawer.draw(d, grid, diagram_format="@+1cm@dr") == \
        "\\xymatrix@+1cm@dr{\n" \
        "A \\ar[d]^{f} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^3mm/[uu]^{h} & \n" \
        "}\n"

    # 创建一个包含三条曲线箭头的图表 d
    h1 = NamedMorphism(D, A, "h1")
    h2 = NamedMorphism(A, D, "h2")
    k = NamedMorphism(D, B, "k")
    d = Diagram([f, g, h, k, h1, h2])

    # 创建一个网格 grid
    grid = DiagramGrid(d)

    # 断言使用 drawer 绘制图表 d 在网格 grid 中的结果
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} & B \\ar[d]^{g} & D \\ar[l]^{k} " \
        "\\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\\\\n" \
        "& C & \n" \
        "}\n"

    # 将图表 d 转置后再次绘制
    grid = DiagramGrid(d, transpose=True)

    # 断言使用 drawer 绘制转置后的图表 d 在网格 grid 中的结果
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} & \n" \
        "}\n"

    # 创建一个包含 "loop" 态射的图表 d
    l_A = NamedMorphism(A, A, "l_A")
    l_D = NamedMorphism(D, D, "l_D")
    l_C = NamedMorphism(C, C, "l_C")
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C])

    # 创建一个网格 grid
    grid = DiagramGrid(d)

    # 断言使用 drawer 绘制图表 d 在网格 grid 中的结果
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} " \
        "& B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_7mm/[ll]_{h} " \
        "\\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} \\\\\n" \
        "& C \\ar@(l,d)[]^{l_{C}} & \n" \
        "}\n"

    # 将图表 d 转置后再次绘制
    grid = DiagramGrid(d, transpose=True)

    # 创建一个新的 drawer 对象
    drawer = XypicDiagramDrawer()
    # 确认使用绘图器绘制的图表与预期的 ASCII 表示是否匹配
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} " \
        "\\ar@(l,d)[]^{l_{D}} & \n" \
        "}\n"
    
    # 创建带有命名态射的图表对象，用于后续的绘制
    l_A_ = NamedMorphism(A, A, "n_A")
    l_D_ = NamedMorphism(D, D, "n_D")
    l_C_ = NamedMorphism(C, C, "n_C")
    d = Diagram([f, g, h, k, h1, h2, l_A, l_D, l_C, l_A_, l_D_, l_C_])
    
    # 创建一个以给定图表为基础的网格对象
    grid = DiagramGrid(d)
    
    # 创建 XypicDiagramDrawer 绘图器的实例
    drawer = XypicDiagramDrawer()
    
    # 确认使用绘图器绘制的图表与预期的 ASCII 表示是否匹配
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[r]_{f} \\ar@/^3mm/[rr]^{h_{2}} \\ar@(u,l)[]^{l_{A}} " \
        "\\ar@/^3mm/@(l,d)[]^{n_{A}} & B \\ar[d]^{g} & D \\ar[l]^{k} " \
        "\\ar@/_7mm/[ll]_{h} \\ar@/_11mm/[ll]_{h_{1}} \\ar@(r,u)[]^{l_{D}} " \
        "\\ar@/^3mm/@(d,r)[]^{n_{D}} \\\\\n" \
        "& C \\ar@(l,d)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} & \n" \
        "}\n"
    
    # 创建带有命名态射的图表对象，用于后续的绘制，同时转置网格
    grid = DiagramGrid(d, transpose=True)
    
    # 创建 XypicDiagramDrawer 绘图器的实例
    drawer = XypicDiagramDrawer()
    
    # 确认使用绘图器绘制的图表与预期的 ASCII 表示是否匹配
    assert drawer.draw(d, grid) == "\\xymatrix{\n" \
        "A \\ar[d]^{f} \\ar@/_3mm/[dd]_{h_{2}} \\ar@(r,u)[]^{l_{A}} " \
        "\\ar@/^3mm/@(u,l)[]^{n_{A}} & \\\\\n" \
        "B \\ar[r]^{g} & C \\ar@(r,u)[]^{l_{C}} \\ar@/^3mm/@(d,r)[]^{n_{C}} \\\\\n" \
        "D \\ar[u]_{k} \\ar@/^7mm/[uu]^{h} \\ar@/^11mm/[uu]^{h_{1}} " \
        "\\ar@(l,d)[]^{l_{D}} \\ar@/^3mm/@(d,r)[]^{n_{D}} & \n" \
        "}\n"
# 定义一个测试函数，用于测试 xypic_draw_diagram 函数的绘图功能
def test_xypic_draw_diagram():
    # 创建五个对象 A, B, C, D, E，分别代表图中的节点
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")
    E = Object("E")

    # 创建从 A 到 B, B 到 C, C 到 D, D 到 E 的命名态射（箭头），分别用 f, g, h, i 表示
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    h = NamedMorphism(C, D, "h")
    i = NamedMorphism(D, E, "i")

    # 创建一个包含这些命名态射的图表对象 d
    d = Diagram([f, g, h, i])

    # 使用布局方式 "sequential" 创建一个图表网格 grid
    grid = DiagramGrid(d, layout="sequential")

    # 创建一个 XypicDiagramDrawer 实例作为绘图器 drawer
    drawer = XypicDiagramDrawer()

    # 断言绘图器 drawer 绘制图表 d，得到的结果与 xypic_draw_diagram 函数相同
    assert drawer.draw(d, grid) == xypic_draw_diagram(d, layout="sequential")
```