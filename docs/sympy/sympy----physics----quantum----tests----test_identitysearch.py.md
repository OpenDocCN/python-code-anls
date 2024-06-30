# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_identitysearch.py`

```
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块导入 Integer 类
from sympy.core.numbers import Integer
# 从 sympy.physics.quantum.dagger 模块导入 Dagger 类
from sympy.physics.quantum.dagger import Dagger
# 从 sympy.physics.quantum.gate 模块导入多个量子门类：X, Y, Z, H, CNOT,
# IdentityGate, CGate, PhaseGate, TGate
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
        IdentityGate, CGate, PhaseGate, TGate)
# 从 sympy.physics.quantum.identitysearch 模块导入多个函数和类：
# generate_gate_rules, generate_equivalent_ids, GateIdentity,
# bfs_identity_search, is_scalar_sparse_matrix,
# is_scalar_nonsparse_matrix, is_degenerate, is_reducible
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
        generate_equivalent_ids, GateIdentity, bfs_identity_search,
        is_scalar_sparse_matrix,
        is_scalar_nonsparse_matrix, is_degenerate, is_reducible)
# 从 sympy.testing.pytest 模块导入 skip 函数
from sympy.testing.pytest import skip


# 定义一个函数 create_gate_sequence，默认创建指定量子比特上的量子门序列
def create_gate_sequence(qubit=0):
    # 定义一组量子门（X, Y, Z, H）并返回
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


# 定义一个测试函数 test_generate_gate_rules_1
def test_generate_gate_rules_1():
    # 调用 create_gate_sequence 函数创建量子门序列，分别赋值给变量 x, y, z, h
    (x, y, z, h) = create_gate_sequence()
    # 创建一个 PhaseGate 对象 ph
    ph = PhaseGate(0)
    # 创建一个控制门 CGate 对象，其中的目标门是 TGate(1)
    cgate_t = CGate(0, TGate(1))

    # 断言：调用 generate_gate_rules 函数，传入参数为元组 (x,)，期望返回值为字典 {((x,), ())}
    assert generate_gate_rules((x,)) == {((x,), ())}

    # 定义期望的 gate_rules 字典，包含两个元组作为键
    gate_rules = {((x, x), ()), ((x,), (x,))}
    # 断言：调用 generate_gate_rules 函数，传入参数为元组 (x, x)，期望返回值为 gate_rules 定义的字典
    assert generate_gate_rules((x, x)) == gate_rules

    # 定义期望的 gate_rules 字典，包含六个元组作为键
    gate_rules = {((x, y, x), ()),
                      ((y, x, x), ()),
                      ((x, x, y), ()),
                      ((y, x), (x,)),
                      ((x, y), (x,)),
                      ((y,), (x, x))}
    # 断言：调用 generate_gate_rules 函数，传入参数为元组 (x, y, x)，期望返回值为 gate_rules 定义的字典
    assert generate_gate_rules((x, y, x)) == gate_rules

    # 定义期望的 gate_rules 字典，包含十二个元组作为键
    gate_rules = {((x, y, z), ()), ((y, z, x), ()), ((z, x, y), ()),
                      ((), (x, z, y)), ((), (y, x, z)), ((), (z, y, x)),
                      ((x,), (z, y)), ((y, z), (x,)), ((y,), (x, z)),
                      ((z, x), (y,)), ((z,), (y, x)), ((x, y), (z,))}
    # 调用 generate_gate_rules 函数，传入参数为元组 (x, y, z)，将返回值赋给变量 actual
    actual = generate_gate_rules((x, y, z))
    # 断言：actual 的值应与 gate_rules 相等
    assert actual == gate_rules

    # 定义期望的 gate_rules 字典，包含十八个元组作为键
    gate_rules = {
        ((), (h, z, y, x)), ((), (x, h, z, y)), ((), (y, x, h, z)),
         ((), (z, y, x, h)), ((h,), (z, y, x)), ((x,), (h, z, y)),
         ((y,), (x, h, z)), ((z,), (y, x, h)), ((h, x), (z, y)),
         ((x, y), (h, z)), ((y, z), (x, h)), ((z, h), (y, x)),
         ((h, x, y), (z,)), ((x, y, z), (h,)), ((y, z, h), (x,)),
         ((z, h, x), (y,)), ((h, x, y, z), ()), ((x, y, z, h), ()),
         ((y, z, h, x), ()), ((z, h, x, y), ())}
    # 调用 generate_gate_rules 函数，传入参数为元组 (x, y, z, h)，将返回值赋给变量 actual
    actual = generate_gate_rules((x, y, z, h))
    # 断言：actual 的值应与 gate_rules 相等
    assert actual == gate_rules

    # 定义期望的 gate_rules 字典，包含十二个元组作为键
    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                      ((), (ph**(-1), x, cgate_t**(-1))),
                      ((), (x, cgate_t**(-1), ph**(-1))),
                      ((cgate_t,), (ph**(-1), x)),
                      ((ph,), (x, cgate_t**(-1))),
                      ((x,), (cgate_t**(-1), ph**(-1))),
                      ((cgate_t, x), (ph**(-1),)),
                      ((ph, cgate_t), (x,)),
                      ((x, ph), (cgate_t**(-1),)),
                      ((cgate_t, x, ph), ()),
                      ((ph, cgate_t, x), ()),
                      ((x, ph, cgate_t), ())}
    # 调用 generate_gate_rules 函数，传入参数为元组 (x, ph, cgate_t)，将返回值赋给变量 actual
    actual = generate_gate_rules((x, ph, cgate_t))
    # 断言：actual 的值应与 gate_rules 相等
    assert actual == gate_rules
    # 定义门规则的集合，包含了不同的门序列及其对应的权重因子
    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                      (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                      (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                      (cgate_t, ph**(-1)*x),
                      (ph, x*cgate_t**(-1)),
                      (x, cgate_t**(-1)*ph**(-1)),
                      (cgate_t*x, ph**(-1)),
                      (ph*cgate_t, x),
                      (x*ph, cgate_t**(-1)),
                      (cgate_t*x*ph, Integer(1)),
                      (ph*cgate_t*x, Integer(1)),
                      (x*ph*cgate_t, Integer(1))}
    # 调用函数生成门规则的集合，以元组形式返回，并要求返回的元素为乘积形式
    actual = generate_gate_rules((x, ph, cgate_t), return_as_muls=True)
    # 断言生成的门规则集合与预定义的 gate_rules 集合相等
    assert actual == gate_rules
def test_generate_gate_rules_2():
    # Test with Muls

    # 创建门序列并分配变量 x, y, z, h
    (x, y, z, h) = create_gate_sequence()
    # 创建一个相位门对象，作用于第0个位置
    ph = PhaseGate(0)
    # 创建一个控制门对象，控制第0个位置，目标是T门对象
    cgate_t = CGate(0, TGate(1))

    # Note: 1 (type int) is not the same as 1 (type One)
    # 期望结果是 {(x, Integer(1))}
    assert generate_gate_rules((x,), return_as_muls=True) == {(x, Integer(1))}

    # 期望结果是 {(Integer(1), Integer(1))}
    assert generate_gate_rules(x*x, return_as_muls=True) == {(Integer(1), Integer(1))}

    # 期望结果是 {((), ())}
    assert generate_gate_rules(x*x, return_as_muls=False) == {((), ())}

    # 定义门规则集合 gate_rules
    gate_rules = {(x*y*x, Integer(1)),
                  (y, Integer(1)),
                  (y*x, x),
                  (x*y, x)}
    # 生成门规则并进行断言比较
    assert generate_gate_rules(x*y*x, return_as_muls=True) == gate_rules

    # 定义门规则集合 gate_rules
    gate_rules = {(x*y*z, Integer(1)),
                  (y*z*x, Integer(1)),
                  (z*x*y, Integer(1)),
                  (Integer(1), x*z*y),
                  (Integer(1), y*x*z),
                  (Integer(1), z*y*x),
                  (x, z*y),
                  (y*z, x),
                  (y, x*z),
                  (z*x, y),
                  (z, y*x),
                  (x*y, z)}
    # 生成门规则并进行断言比较
    actual = generate_gate_rules(x*y*z, return_as_muls=True)
    assert actual == gate_rules

    # 定义门规则集合 gate_rules
    gate_rules = {(Integer(1), h*z*y*x),
                  (Integer(1), x*h*z*y),
                  (Integer(1), y*x*h*z),
                  (Integer(1), z*y*x*h),
                  (h, z*y*x),
                  (x, h*z*y),
                  (y, x*h*z),
                  (z, y*x*h),
                  (h*x, z*y),
                  (z*h, y*x),
                  (x*y, h*z),
                  (y*z, x*h),
                  (h*x*y, z),
                  (x*y*z, h),
                  (y*z*h, x),
                  (z*h*x, y),
                  (h*x*y*z, Integer(1)),
                  (x*y*z*h, Integer(1)),
                  (y*z*h*x, Integer(1)),
                  (z*h*x*y, Integer(1))}
    # 生成门规则并进行断言比较
    actual = generate_gate_rules(x*y*z*h, return_as_muls=True)
    assert actual == gate_rules

    # 定义门规则集合 gate_rules
    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                  (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                  (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                  (cgate_t, ph**(-1)*x),
                  (ph, x*cgate_t**(-1)),
                  (x, cgate_t**(-1)*ph**(-1)),
                  (cgate_t*x, ph**(-1)),
                  (ph*cgate_t, x),
                  (x*ph, cgate_t**(-1)),
                  (cgate_t*x*ph, Integer(1)),
                  (ph*cgate_t*x, Integer(1)),
                  (x*ph*cgate_t, Integer(1))}
    # 生成门规则并进行断言比较
    actual = generate_gate_rules(x*ph*cgate_t, return_as_muls=True)
    assert actual == gate_rules
    # 定义门规则的集合，每个元素是一个元组，包含两个空元组和一个元组，每个元组包含两个元素
    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                  ((), (ph**(-1), x, cgate_t**(-1))),
                  ((), (x, cgate_t**(-1), ph**(-1))),
                  ((cgate_t,), (ph**(-1), x)),
                  ((ph,), (x, cgate_t**(-1))),
                  ((x,), (cgate_t**(-1), ph**(-1))),
                  ((cgate_t, x), (ph**(-1),)),
                  ((ph, cgate_t), (x,)),
                  ((x, ph), (cgate_t**(-1),)),
                  ((cgate_t, x, ph), ()),
                  ((ph, cgate_t, x), ()),
                  ((x, ph, cgate_t), ())}
    
    # 调用函数生成给定 x*ph*cgate_t 的门规则集合
    actual = generate_gate_rules(x*ph*cgate_t)
    
    # 断言生成的门规则集合与预期的 gate_rules 相等
    assert actual == gate_rules
def test_generate_equivalent_ids_1():
    # Test with tuples

    # 创建门序列并解包为元组 (x, y, z, h)
    (x, y, z, h) = create_gate_sequence()

    # 测试生成等价标识集合
    assert generate_equivalent_ids((x,)) == {(x,)}
    assert generate_equivalent_ids((x, x)) == {(x, x)}
    assert generate_equivalent_ids((x, y)) == {(x, y), (y, x)}

    # 设置门序列为 (x, y, z)
    gate_seq = (x, y, z)

    # 生成包含所有可能顺序的门组合的集合
    gate_ids = {(x, y, z), (y, z, x), (z, x, y), (z, y, x),
                    (y, x, z), (x, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    # 生成包含所有可能顺序的门组合的集合，作为 Mul 对象
    gate_ids = {Mul(x, y, z), Mul(y, z, x), Mul(z, x, y),
                    Mul(z, y, x), Mul(y, x, z), Mul(x, z, y)}
    assert generate_equivalent_ids(gate_seq, return_as_muls=True) == gate_ids

    # 设置门序列为 (x, y, z, h)
    gate_seq = (x, y, z, h)

    # 生成包含所有可能顺序的门组合的集合
    gate_ids = {(x, y, z, h), (y, z, h, x),
                    (h, x, y, z), (h, z, y, x),
                    (z, y, x, h), (y, x, h, z),
                    (z, h, x, y), (x, h, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    # 设置门序列为 (x, y, x, y)
    gate_seq = (x, y, x, y)

    # 生成包含所有可能顺序的门组合的集合
    gate_ids = {(x, y, x, y), (y, x, y, x)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    # 创建 CGate 对象 cgate_y
    cgate_y = CGate((1,), y)
    # 设置门序列为 (y, cgate_y, y, cgate_y)
    gate_seq = (y, cgate_y, y, cgate_y)

    # 生成包含所有可能顺序的门组合的集合
    gate_ids = {(y, cgate_y, y, cgate_y), (cgate_y, y, cgate_y, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    # 创建 CNOT 和 CGate 对象 cnot, cgate_z
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    # 设置门序列为 (cnot, h, cgate_z, h)
    gate_seq = (cnot, h, cgate_z, h)

    # 生成包含所有可能顺序的门组合的集合
    gate_ids = {(cnot, h, cgate_z, h), (h, cgate_z, h, cnot),
                    (h, cnot, h, cgate_z), (cgate_z, h, cnot, h)}
    assert generate_equivalent_ids(gate_seq) == gate_ids


def test_generate_equivalent_ids_2():
    # Test with Muls

    # 创建门序列并解包为元组 (x, y, z, h)
    (x, y, z, h) = create_gate_sequence()

    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids((x,), return_as_muls=True) == {x}

    # 创建 Integer(1) 对象
    gate_ids = {Integer(1)}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(x*x, return_as_muls=True) == gate_ids

    # 创建 x*y 和 y*x 对象
    gate_ids = {x*y, y*x}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(x*y, return_as_muls=True) == gate_ids

    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {(x, y), (y, x)}
    # 测试生成等价标识集合
    assert generate_equivalent_ids(x*y) == gate_ids

    # 创建 Mul 对象 circuit = x * y * z
    circuit = Mul(*(x, y, z))
    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {x*y*z, y*z*x, z*x*y, z*y*x,
                    y*x*z, x*z*y}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    # 创建 Mul 对象 circuit = x * y * z * h
    circuit = Mul(*(x, y, z, h))
    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {x*y*z*h, y*z*h*x,
                    h*x*y*z, h*z*y*x,
                    z*y*x*h, y*x*h*z,
                    z*h*x*y, x*h*z*y}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    # 创建 Mul 对象 circuit = x * y * x * y
    circuit = Mul(*(x, y, x, y))
    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {x*y*x*y, y*x*y*x}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    # 创建 CGate 对象 cgate_y
    cgate_y = CGate((1,), y)
    # 创建 Mul 对象 circuit = y * cgate_y * y * cgate_y
    circuit = Mul(*(y, cgate_y, y, cgate_y))
    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {y*cgate_y*y*cgate_y, cgate_y*y*cgate_y*y}
    # 测试生成等价标识集合，返回为 Muls 对象
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    # 创建 CNOT 和 CGate 对象 cnot, cgate_z
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    # 创建 Mul 对象 circuit = cnot * h * cgate_z * h
    circuit = Mul(*(cnot, h, cgate_z, h))
    # 创建包含所有可能顺序的门组合的集合
    gate_ids = {cnot*h*cgate_z*h, h*cgate_z*h*cnot,
                    h*cnot*h*cgate_z, cgate_z*h*cnot*h}
    # 使用断言来检查函数生成的等效 IDs 是否与给定的 gate_ids 相等
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids
# 定义一个测试函数，用于检查是否为标量非稀疏矩阵
def test_is_scalar_nonsparse_matrix():
    # 设定量子比特数为2
    numqubits = 2
    # 设定是否仅包含单位矩阵的标志为False
    id_only = False

    # 创建一个仅包含单个单位门的元组
    id_gate = (IdentityGate(1),)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(id_gate, numqubits, id_only)
    assert actual is True

    # 创建一个包含两个X门的量子电路元组
    x0 = X(0)
    xx_circuit = (x0, x0)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(xx_circuit, numqubits, id_only)
    assert actual is True

    # 创建一个包含X门和Y门的量子电路元组
    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为False
    actual = is_scalar_nonsparse_matrix(xy_circuit, numqubits, id_only)
    assert actual is False

    # 创建一个包含X门、Y门和Z门的量子电路元组
    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is True

    # 创建一个包含CNOT门的量子电路元组
    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True

    # 创建一个包含Hadamard门的量子电路元组
    h = H(0)
    hh_circuit = (h, h)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True

    # 创建一个包含X门、H门、Z门和H门的量子电路元组
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True

    # 将id_only标志设为True
    id_only = True
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为False
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is False
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True
    # 调用函数is_scalar_nonsparse_matrix，判断其返回值是否为True
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True


# 定义一个测试函数，用于检查是否为标量稀疏矩阵
def test_is_scalar_sparse_matrix():
    # 导入numpy模块，如果导入失败则跳过测试
    np = import_module('numpy')
    if not np:
        skip("numpy not installed.")

    # 导入scipy模块中的sparse部分，如果导入失败则跳过测试
    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if not scipy:
        skip("scipy not installed.")

    # 设定量子比特数为2
    numqubits = 2
    # 设定是否仅包含单位矩阵的标志为False
    id_only = False

    # 创建一个仅包含单个单位门的元组
    id_gate = (IdentityGate(1),)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(id_gate, numqubits, id_only) is True

    # 创建一个包含两个X门的量子电路元组
    x0 = X(0)
    xx_circuit = (x0, x0)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(xx_circuit, numqubits, id_only) is True

    # 创建一个包含X门和Y门的量子电路元组
    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    # 断言函数is_scalar_sparse_matrix的返回值为False
    assert is_scalar_sparse_matrix(xy_circuit, numqubits, id_only) is False

    # 创建一个包含X门、Y门和Z门的量子电路元组
    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is True

    # 创建一个包含CNOT门的量子电路元组
    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True

    # 创建一个包含Hadamard门的量子电路元组
    h = H(0)
    hh_circuit = (h, h)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True

    # 创建一个包含X门、H门、Z门和H门的量子电路元组
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True

    # 将id_only标志设为True
    id_only = True
    # 断言函数is_scalar_sparse_matrix的返回值为True
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True
    # 断言函数is_scalar_sparse_matrix的返回值为False
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is False
    # 断言：验证 cnot_circuit 是一个标量稀疏矩阵，并且 is_scalar_sparse_matrix 返回 True
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True
    
    # 断言：验证 hh_circuit 是一个标量稀疏矩阵，并且 is_scalar_sparse_matrix 返回 True
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True
# 测试是否为退化量子电路的函数
def test_is_degenerate():
    # 创建一个门序列，并解包得到 x, y, z, h 四个门操作符
    (x, y, z, h) = create_gate_sequence()

    # 创建一个 GateIdentity 对象，包含 x, y, z 三个门操作符
    gate_id = GateIdentity(x, y, z)
    # 将 gate_id 添加到集合 ids 中
    ids = {gate_id}

    # 创建另一个门标识对象，顺序为 z, y, x
    another_id = (z, y, x)
    # 断言 is_degenerate 函数对 ids 和 another_id 返回 True
    assert is_degenerate(ids, another_id) is True


# 测试是否可简化量子电路的函数
def test_is_reducible():
    # 设定量子比特数为 2
    nqubits = 2
    # 创建一个门序列，并解包得到 x, y, z, h 四个门操作符
    (x, y, z, h) = create_gate_sequence()

    # 定义电路 circuit 为 (x, y, y)
    circuit = (x, y, y)
    # 断言 is_reducible 函数对 circuit, nqubits, 1, 3 返回 True
    assert is_reducible(circuit, nqubits, 1, 3) is True

    # 依次类推，进行其他电路的简化测试
    circuit = (x, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is False

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 0, 4) is True

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is True

    circuit = (x, y, z, y, y)
    assert is_reducible(circuit, nqubits, 1, 5) is True


# 测试广度优先搜索门操作符标识的函数
def test_bfs_identity_search():
    # 断言对空 gate_list 和 1 的 bfs_identity_search 函数返回空集合
    assert bfs_identity_search([], 1) == set()

    # 创建一个门序列，并解包得到 x, y, z, h 四个门操作符
    (x, y, z, h) = create_gate_sequence()

    # gate_list 包含一个元素 x
    gate_list = [x]
    # 创建包含 GateIdentity(x, x) 的集合 id_set
    id_set = {GateIdentity(x, x)}
    # 断言 bfs_identity_search 函数对 gate_list, 1, max_depth=2 返回 id_set
    assert bfs_identity_search(gate_list, 1, max_depth=2) == id_set

    # 继续测试包含多个门操作符的 gate_list
    gate_list = [x, y, z]
    id_set = {GateIdentity(x, x),
              GateIdentity(y, y),
              GateIdentity(z, z),
              GateIdentity(x, y, z)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
              GateIdentity(y, y),
              GateIdentity(z, z),
              GateIdentity(x, y, z),
              GateIdentity(x, y, x, y),
              GateIdentity(x, z, x, z),
              GateIdentity(y, z, y, z)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    gate_list = [x, y, z, h]
    id_set = {GateIdentity(x, x),
              GateIdentity(y, y),
              GateIdentity(z, z),
              GateIdentity(h, h),
              GateIdentity(x, y, z),
              GateIdentity(x, y, x, y),
              GateIdentity(x, z, x, z),
              GateIdentity(x, h, z, h),
              GateIdentity(y, z, y, z),
              GateIdentity(y, h, y, h)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
              GateIdentity(y, y),
              GateIdentity(z, z),
              GateIdentity(h, h)}
    # 断言 bfs_identity_search 函数对 gate_list, 1, max_depth=3, identity_only=True 返回 id_set
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=3,
                                         identity_only=True)

    id_set = {GateIdentity(x, x),
              GateIdentity(y, y),
              GateIdentity(z, z),
              GateIdentity(h, h),
              GateIdentity(x, y, z),
              GateIdentity(x, y, x, y),
              GateIdentity(x, z, x, z),
              GateIdentity(x, h, z, h),
              GateIdentity(y, z, y, z),
              GateIdentity(y, h, y, h),
              GateIdentity(x, y, h, x, h),
              GateIdentity(x, z, h, y, h),
              GateIdentity(y, z, h, z, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=4)
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 5 的标识集合，并验证其与 id_set 相等
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    # 创建包含 GateIdentity 对象的集合 id_set，包括门 x, y, z, h 和它们的标识
    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, h, z, h)}
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 4 的标识集合，仅考虑标识，验证其与 id_set 相等
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=4,
                                         identity_only=True)

    # 创建 CNOT 门对象 cnot，并将其添加到 gate_list 中
    cnot = CNOT(1, 0)
    gate_list = [x, cnot]
    # 创建包含 GateIdentity 对象的集合 id_set，包括门 x, cnot 和它们的标识
    id_set = {GateIdentity(x, x),
                  GateIdentity(cnot, cnot),
                  GateIdentity(x, cnot, x, cnot)}
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 4 的标识集合，验证其与 id_set 相等
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    # 创建 CGate 对象 cgate_x，并将其添加到 gate_list 中
    cgate_x = CGate((1,), x)
    gate_list = [x, cgate_x]
    # 创建包含 GateIdentity 对象的集合 id_set，包括门 x, cgate_x 和它们的标识
    id_set = {GateIdentity(x, x),
                  GateIdentity(cgate_x, cgate_x),
                  GateIdentity(x, cgate_x, x, cgate_x)}
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 4 的标识集合，验证其与 id_set 相等
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    # 创建 CGate 对象 cgate_z，并将其添加到 gate_list 中；同时添加门 cnot 和 h 到 gate_list 中
    cgate_z = CGate((0,), Z(1))
    gate_list = [cnot, cgate_z, h]
    # 创建包含 GateIdentity 对象的集合 id_set，包括门 h, cgate_z, cnot 和它们的标识
    id_set = {GateIdentity(h, h),
                  GateIdentity(cgate_z, cgate_z),
                  GateIdentity(cnot, cnot),
                  GateIdentity(cnot, h, cgate_z, h)}
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 4 的标识集合，验证其与 id_set 相等
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    # 创建 PhaseGate 对象 s 和 TGate 对象 t，并将它们添加到 gate_list 中
    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [s, t]
    # 创建包含 GateIdentity 对象的集合 id_set，包括门 s 和它们的标识
    id_set = {GateIdentity(s, s, s, s)}
    # 断言：使用 BFS 算法在 gate_list 中搜索深度为 4 的标识集合，验证其与 id_set 相等
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set
# 定义一个测试函数，用于测试 BFS（广度优先搜索）身份搜索的特性，预期会失败（xfail）
def test_bfs_identity_search_xfail():
    # 创建一个 PhaseGate 对象，作用在量子比特 0 上
    s = PhaseGate(0)
    # 创建一个 TGate 对象，作用在量子比特 0 上
    t = TGate(0)
    # 创建一个包含 Dagger(s) 和 t 的门列表
    gate_list = [Dagger(s), t]
    # 创建一个包含 GateIdentity(Dagger(s), t, t) 的集合
    id_set = {GateIdentity(Dagger(s), t, t)}
    # 断言 BFS 身份搜索函数对于 gate_list 中的门，最大深度为 3 时，返回的结果与 id_set 相等
    assert bfs_identity_search(gate_list, 1, max_depth=3) == id_set
```