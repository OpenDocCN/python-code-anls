# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_circuitutils.py`

```
# 导入必要的符号计算模块
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
# 导入符号计算工具模块
from sympy.utilities import numbered_symbols
# 导入量子计算相关模块
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
# 导入量子计算中的图搜索相关模块
from sympy.physics.quantum.identitysearch import bfs_identity_search
# 导入量子电路工具模块，包括一系列函数
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
        replace_subcircuit, convert_to_symbolic_indices,
        convert_to_real_indices, random_reduce, random_insert,
        flatten_ids)
# 导入用于测试的 pytest 的 slow 装饰器
from sympy.testing.pytest import slow


# 创建量子门序列的函数，参数为 qubit 默认为 0
def create_gate_sequence(qubit=0):
    # 定义常见的量子门序列
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


# 测试 KMP 算法生成的表是否正确
def test_kmp_table():
    # 定义不同的测试用例和预期结果
    word = ('a', 'b', 'c', 'd', 'a', 'b', 'd')
    expected_table = [-1, 0, 0, 0, 0, 1, 2]
    assert expected_table == kmp_table(word)

    word = ('P', 'A', 'R', 'T', 'I', 'C', 'I', 'P', 'A', 'T', 'E', ' ',
            'I', 'N', ' ', 'P', 'A', 'R', 'A', 'C', 'H', 'U', 'T', 'E')
    expected_table = [-1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
                      0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
    assert expected_table == kmp_table(word)

    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    word = (x, y, y, x, z)
    expected_table = [-1, 0, 0, 0, 1]
    assert expected_table == kmp_table(word)

    word = (x, x, y, h, z)
    expected_table = [-1, 0, 1, 0, 0]
    assert expected_table == kmp_table(word)


# 测试在量子电路中查找子电路的功能
def test_find_subcircuit():
    # 定义不同的量子门和符号
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    x1 = X(1)
    y1 = Y(1)

    i0 = Symbol('i0')
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)
    h_i0 = H(i0)

    circuit = (x, y, z)

    assert find_subcircuit(circuit, (x,)) == 0
    assert find_subcircuit(circuit, (x1,)) == -1
    assert find_subcircuit(circuit, (y,)) == 1
    assert find_subcircuit(circuit, (h,)) == -1
    assert find_subcircuit(circuit, Mul(x, h)) == -1
    assert find_subcircuit(circuit, Mul(x, y, z)) == 0
    assert find_subcircuit(circuit, Mul(y, z)) == 1
    assert find_subcircuit(Mul(*circuit), (x, y, z, h)) == -1
    assert find_subcircuit(Mul(*circuit), (z, y, x)) == -1
    assert find_subcircuit(circuit, (x,), start=2, end=1) == -1

    circuit = (x, y, x, y, z)
    assert find_subcircuit(Mul(*circuit), Mul(x, y, z)) == 2
    assert find_subcircuit(circuit, (x,), start=1) == 2
    assert find_subcircuit(circuit, (x, y), start=1, end=2) == -1
    assert find_subcircuit(Mul(*circuit), (x, y), start=1, end=3) == -1
    assert find_subcircuit(circuit, (x, y), start=1, end=4) == 2
    assert find_subcircuit(circuit, (x, y), start=2, end=4) == 2

    circuit = (x, y, z, x1, x, y, z, h, x, y, x1,
               x, y, z, h, y1, h)
    assert find_subcircuit(circuit, (x, y, z, h, y1)) == 11

    circuit = (x, y, x_i0, y_i0, z_i0, z)
    assert find_subcircuit(circuit, (x_i0, y_i0, z_i0)) == 2

    circuit = (x_i0, y_i0, z_i0, x_i0, y_i0, h_i0)
    subcircuit = (x_i0, y_i0, z_i0)
    result = find_subcircuit(circuit, subcircuit)
    assert result == 0
def test_replace_subcircuit():
    # 创建具体的量子门对象，并将其分配给变量
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    # 标准情况下的测试用例
    circuit = (z, y, x, x)
    remove = (z, y, x)
    # 测试替换子电路函数，验证返回结果是否符合预期
    assert replace_subcircuit(circuit, Mul(*remove)) == (x,)
    assert replace_subcircuit(circuit, remove + (x,)) == ()
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    assert replace_subcircuit(circuit, remove, pos=0) == (x,)
    assert replace_subcircuit(circuit, (x, x), pos=2) == (z, y)
    assert replace_subcircuit(circuit, (h,)) == circuit

    circuit = (x, y, x, y, z)
    remove = (x, y, z)
    assert replace_subcircuit(Mul(*circuit), Mul(*remove)) == (x, y)
    remove = (x, y, x, y)
    assert replace_subcircuit(circuit, remove) == (z,)

    circuit = (x, h, cgate_z, h, cnot)
    remove = (x, h, cgate_z)
    assert replace_subcircuit(circuit, Mul(*remove), pos=-1) == (h, cnot)
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    remove = (h, h)
    assert replace_subcircuit(circuit, remove) == circuit
    remove = (h, cgate_z, h, cnot)
    assert replace_subcircuit(circuit, remove) == (x,)

    replace = (h, x)
    # 测试在指定位置替换子电路并返回结果
    actual = replace_subcircuit(circuit, remove, replace=replace)
    assert actual == (x, h, x)

    circuit = (x, y, h, x, y, z)
    remove = (x, y)
    replace = (cnot, cgate_z)
    # 测试在指定位置替换子电路并返回结果
    actual = replace_subcircuit(circuit, remove, replace=Mul(*replace))
    assert actual == (cnot, cgate_z, h, x, y, z)

    actual = replace_subcircuit(circuit, remove, replace=replace, pos=1)
    # 测试在指定位置替换子电路并返回结果
    assert actual == (x, y, h, cnot, cgate_z, z)


def test_convert_to_symbolic_indices():
    # 创建门序列并分配给变量
    (x, y, z, h) = create_gate_sequence()

    i0 = Symbol('i0')
    exp_map = {i0: Integer(0)}
    # 测试将具体门转换为符号索引的函数，验证返回结果是否符合预期
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x,))
    assert actual == (X(i0),)
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h))
    assert actual == expected
    assert exp_map == act_map

    (x1, y1, z1, h1) = create_gate_sequence(1)
    i1 = Symbol('i1')

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0), X(i1), Y(i1), Z(i1), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h,
                                         x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x1, y1,
                                         z1, h1, x, y, z, h))
    assert actual == expected
    assert act_map == exp_map
    # 定义预期的元组，包含符号 X、Y、Z 和 H 的不同组合
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1))
    # 定义预期的映射，将符号 i0 映射到整数 0，将符号 i1 映射到整数 1
    exp_map = {i0: Integer(0), i1: Integer(1)}
    # 将 x、x1、y、y1、z、z1、h、h1 这些符号转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x, x1,
                                         y, y1, z, z1, h, h1))
    # 断言实际的返回值与预期的元组相等
    assert actual == expected
    # 断言实际的映射与预期的映射相等
    assert act_map == exp_map

    # 更新预期的映射，将符号 i0 映射到整数 1，将符号 i1 映射到整数 0
    exp_map = {i0: Integer(1), i1: Integer(0)}
    # 将 x1、x、y1、y、z1、z、h1、h 这些符号转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, x, y1, y,
                                         z1, z, h1, h))
    # 断言实际的返回值与更新后的预期元组相等
    assert actual == expected
    # 断言实际的映射与更新后的预期映射相等
    assert act_map == exp_map

    # 创建 CNOT 和 CGate 对象，用于后续的符号索引转换
    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    # 定义新的预期元组，包括符号索引和控制门的不同组合
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i1, i0), CNOT(i0, i1),
                CGate(i1, Z(i0)), CGate(i0, Z(i1)))
    # 更新预期的映射，将符号 i0 映射到整数 0，将符号 i1 映射到整数 1
    exp_map = {i0: Integer(0), i1: Integer(1)}
    # 将 x、x1、y、y1、z、z1、h、h1、cnot_10、cnot_01、cgate_z_10、cgate_z_01 这些符号转换为符号索引，并获取相关的返回值
    args = (x, x1, y, y1, z, z1, h, h1, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际的返回值与更新后的预期元组相等
    assert actual == expected
    # 断言实际的映射与更新后的预期映射相等
    assert act_map == exp_map

    # 更新参数顺序，以符合新的预期元组
    args = (x1, x, y1, y, z1, z, h1, h, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    # 更新预期元组，包括符号索引和控制门的不同组合
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i0, i1), CNOT(i1, i0),
                CGate(i0, Z(i1)), CGate(i1, Z(i0)))
    # 更新预期的映射，将符号 i0 映射到整数 1，将符号 i1 映射到整数 0
    exp_map = {i0: Integer(1), i1: Integer(0)}
    # 将更新后的参数转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际的返回值与更新后的预期元组相等
    assert actual == expected
    # 断言实际的映射与更新后的预期映射相等
    assert act_map == exp_map

    # 定义新的参数组合，包含控制门和 H 门
    args = (cnot_10, h, cgate_z_01, h)
    # 更新预期元组，包括控制门和 H 门的不同组合
    expected = (CNOT(i0, i1), H(i1), CGate(i1, Z(i0)), H(i1))
    # 更新预期的映射，将符号 i0 映射到整数 1，将符号 i1 映射到整数 0
    exp_map = {i0: Integer(1), i1: Integer(0)}
    # 将参数转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际的返回值与更新后的预期元组相等
    assert actual == expected
    # 断言实际的映射与更新后的预期映射相等
    assert act_map == exp_map

    # 定义新的参数组合，包含控制门和 H 门
    args = (cnot_01, h1, cgate_z_10, h1)
    # 更新预期的映射，将符号 i0 映射到整数 0，将符号 i1 映射到整数 1
    exp_map = {i0: Integer(0), i1: Integer(1)}
    # 将参数转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际的返回值与预期的元组相等
    assert actual == expected
    # 断言实际的映射与预期的映射相等
    assert act_map == exp_map

    # 定义新的参数组合，包含控制门和 H 门
    args = (cnot_10, h1, cgate_z_01, h1)
    # 更新预期元组，包括控制门和 H 门的不同组合
    expected = (CNOT(i0, i1), H(i0), CGate(i1, Z(i0)), H(i0))
    # 更新预期的映射，将符号 i0 映射到整数 1，将符号 i1 映射到整数 0
    exp_map = {i0: Integer(1), i1: Integer(0)}
    # 将参数转换为符号索引，并获取相关的返回值
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际的返回值与更新后的预期元组相等
    assert actual == expected
    # 断言实际的映射与更新后的预期映射相等
    assert act_map == exp_map

    # 定义新的符号 i2
    i2 = Symbol('i2')
    # 创建多重控制门对象
    ccgate_z = CGate(0, CGate(1, Z(2)))
    ccgate_x = CGate(1, CGate(2, X(0)))
    # 定义新的参数组合，包含多重控制门对象
    args = (ccgate_z, ccgate_x)
    # 更新预
    # 调用函数 convert_to_symbolic_indices，并传入参数 args、qubit_map=ndx_map、start=i0、gen=index_gen
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args,
                                         qubit_map=ndx_map,
                                         start=i0,
                                         gen=index_gen)
    # 断言实际返回值 actual 等于期望值 expected
    assert actual == expected
    # 断言实际映射 act_map 等于期望映射 exp_map
    assert act_map == exp_map

    # 创建符号变量 i3
    i3 = Symbol('i3')
    # 创建 CGate 对象 cgate_x0_c321，该对象表示控制门 (3, 2, 1) 控制第 0 个量子比特的 X 门
    cgate_x0_c321 = CGate((3, 2, 1), X(0))
    # 创建期望的映射 exp_map，将 i0 映射到整数 3，i1 映射到整数 2，i2 映射到整数 1，i3 映射到整数 0
    exp_map = {i0: Integer(3), i1: Integer(2),
               i2: Integer(1), i3: Integer(0)}
    # 创建期望的结果元组 expected，包含一个 CGate 对象，其控制位为 (i0, i1, i2)，作用门为 X(i3)
    expected = (CGate((i0, i1, i2), X(i3)),)
    # 更新参数 args，使其包含 cgate_x0_c321 对象
    args = (cgate_x0_c321,)
    # 调用函数 convert_to_symbolic_indices，并传入更新后的参数 args
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    # 断言实际返回值 actual 等于期望值 expected
    assert actual == expected
    # 断言实际映射 act_map 等于期望映射 exp_map
    assert act_map == exp_map
def test_convert_to_real_indices():
    # 创建符号变量 i0 和 i1
    i0 = Symbol('i0')
    i1 = Symbol('i1')

    # 创建门序列并解包为变量 x, y, z, h
    (x, y, z, h) = create_gate_sequence()

    # 创建针对 i0 的 X, Y, Z 门操作
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)

    # 定义 qubit_map，将 i0 映射到 0
    qubit_map = {i0: 0}
    # 定义 args 和期望结果 expected
    args = (z_i0, y_i0, x_i0)
    expected = (z, y, x)
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    # 创建不同的控制门和控制门带符号参数的操作
    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    # 创建针对 i0 和 i1 的 CNOT 和 CGate 操作
    cnot_i1_i0 = CNOT(i1, i0)
    cnot_i0_i1 = CNOT(i0, i1)
    cgate_z_i1_i0 = CGate(i1, Z(i0))

    # 设置 qubit_map，将 i0 映射到 0，i1 映射到 1
    qubit_map = {i0: 0, i1: 1}
    # 定义 args 和期望结果 expected
    args = (cnot_i1_i0,)
    expected = (cnot_10,)
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    # 重置 args 和期望结果 expected
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_10,)
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    # 重置 args 和期望结果 expected
    args = (cnot_i0_i1,)
    expected = (cnot_01,)
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    # 重置 qubit_map，将 i0 映射到 1，i1 映射到 0
    qubit_map = {i0: 1, i1: 0}
    # 重置 args 和期望结果 expected
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_01,)
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    # 创建符号变量 i2
    i2 = Symbol('i2')
    # 创建复合门操作 ccgate_z 和 ccgate_x
    ccgate_z = CGate(i0, CGate(i1, Z(i2)))
    ccgate_x = CGate(i1, CGate(i2, X(i0)))

    # 设置 qubit_map，将 i0 映射到 0，i1 映射到 1，i2 映射到 2
    qubit_map = {i0: 0, i1: 1, i2: 2}
    # 定义 args 和期望结果 expected
    args = (ccgate_z, ccgate_x)
    expected = (CGate(0, CGate(1, Z(2))), CGate(1, CGate(2, X(0))))
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(Mul(*args), qubit_map)
    assert actual == expected

    # 重置 qubit_map，将 i0 映射到 1，i1 映射到 2，i2 映射到 0
    qubit_map = {i0: 1, i2: 0, i1: 2}
    # 重置 args 和期望结果 expected
    args = (ccgate_x, ccgate_z)
    expected = (CGate(2, CGate(0, X(1))), CGate(1, CGate(2, Z(0))))
    # 调用函数 convert_to_real_indices，并进行断言
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    # 调用 random_insert 函数，在 circuit 中随机插入 choices 中的元素，使用 seed 控制随机性
    actual = random_insert(circuit, choices, seed=[loc, choice])
    # 断言检查插入结果是否符合预期
    assert actual == (x, x, y, y)

    # 设定新的 circuit 和 choices
    circuit = (x, y, z, h)
    choices = [(h, h), (x, y, z)]
    # 期望的插入结果
    expected = (x, x, y, z, y, z, h)
    # 设定 loc 和 choice 的值
    loc, choice = 1, 1
    # 调用 random_insert 函数，在 circuit 中随机插入 choices 中的元素，使用 seed 控制随机性
    actual = random_insert(circuit, choices, seed=[loc, choice])
    # 断言检查插入结果是否符合预期
    assert actual == expected

    # 生成 gate_list 列表，其中包含 x, y, z, h, cnot, cgate_z
    gate_list = [x, y, z, h, cnot, cgate_z]
    # 使用 bfs_identity_search 函数搜索长度为 2 的基于 gate_list 的 BFS 树，最大深度为 4
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))

    # 将多层列表 ids 扁平化
    eq_ids = flatten_ids(ids)

    # 设定新的 circuit
    circuit = (x, y, h, cnot, cgate_z)
    # 期望的插入结果
    expected = (x, z, x, z, x, y, h, cnot, cgate_z)
    # 设定 loc 和 choice 的值
    loc, choice = 1, 30
    # 调用 random_insert 函数，在 circuit 中随机插入 eq_ids 中的元素，使用 seed 控制随机性
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    # 断言检查插入结果是否符合预期
    assert actual == expected

    # 将 circuit 中的元素作为参数传递给 Mul 函数，并将返回值重新赋给 circuit
    circuit = Mul(*circuit)
    # 再次调用 random_insert 函数，在 circuit 中随机插入 eq_ids 中的元素，使用 seed 控制随机性
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    # 断言检查插入结果是否符合预期
    assert actual == expected
```