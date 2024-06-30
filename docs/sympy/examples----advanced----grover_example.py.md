# `D:\src\scipysrc\sympy\examples\advanced\grover_example.py`

```
#!/usr/bin/env python

"""Grover's quantum search algorithm example."""

# 导入需要的模块和函数
from sympy import pprint
from sympy.physics.quantum import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (OracleGate, superposition_basis,
        WGate, grover_iteration)


def demo_vgate_app(v):
    # 遍历所有可能的整数态
    for i in range(2**v.nqubits):
        # 打印调用信息
        print('qapply(v*IntQubit(%i, %r))' % (i, v.nqubits))
        # 打印并演示量子态的应用结果
        pprint(qapply(v*IntQubit(i, nqubits=v.nqubits)))
        # 再次打印调用信息
        qapply(v*IntQubit(i, nqubits=v.nqubits))


def black_box(qubits):
    # 判断量子态是否等于预定义的整数态
    return True if qubits == IntQubit(1, nqubits=qubits.nqubits) else False


def main():
    print()
    print('Demonstration of Grover\'s Algorithm')
    print('The OracleGate or V Gate carries the unknown function f(x)')
    print('> V|x> = ((-1)^f(x))|x> where f(x) = 1 when x = a (True in our case)')
    print('> and 0 (False in our case) otherwise')
    print()

    # 设置量子比特数量
    nqubits = 2
    print('nqubits = ', nqubits)

    # 创建 OracleGate 对象
    v = OracleGate(nqubits, black_box)
    print('Oracle or v = OracleGate(%r, black_box)' % nqubits)
    print()

    # 构建超position基础
    psi = superposition_basis(nqubits)
    print('psi:')
    pprint(psi)
    # 演示 V Gate 的应用
    demo_vgate_app(v)
    print('qapply(v*psi)')
    pprint(qapply(v*psi))
    print()

    # 创建 WGate 对象
    w = WGate(nqubits)
    print('WGate or w = WGate(%r)' % nqubits)
    print('On a 2 Qubit system like psi, 1 iteration is enough to yield |1>')
    print('qapply(w*v*psi)')
    pprint(qapply(w*v*psi))
    print()

    # 设置量子比特数量为3
    nqubits = 3
    print('On a 3 Qubit system, it requires 2 iterations to achieve')
    print('|1> with high enough probability')
    psi = superposition_basis(nqubits)
    print('psi:')
    pprint(psi)

    # 创建 OracleGate 对象
    v = OracleGate(nqubits, black_box)
    print('Oracle or v = OracleGate(%r, black_box)' % nqubits)
    print()

    # 执行第一次 Grover 迭代
    print('iter1 = grover.grover_iteration(psi, v)')
    iter1 = qapply(grover_iteration(psi, v))
    pprint(iter1)
    print()

    # 执行第二次 Grover 迭代
    print('iter2 = grover.grover_iteration(iter1, v)')
    iter2 = qapply(grover_iteration(iter1, v))
    pprint(iter2)
    print()

if __name__ == "__main__":
    main()
```