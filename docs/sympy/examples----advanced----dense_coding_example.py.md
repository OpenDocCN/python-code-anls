# `D:\src\scipysrc\sympy\examples\advanced\dense_coding_example.py`

```
# Python 脚本的入口点，用于演示量子密集编码。
#!/usr/bin/env python

# 导入 sympy 库中的打印函数
from sympy import pprint
# 导入 sympy 的量子物理模块
from sympy.physics.quantum import qapply
# 导入量子门：Hadamard门、Pauli-X门、Pauli-Z门和CNOT门
from sympy.physics.quantum.gate import H, X, Z, CNOT
# 导入量子搜索算法库
from sympy.physics.quantum.grover import superposition_basis

# 主函数定义
def main():
    # 创建一个包含两个量子比特的量子叠加态
    psi = superposition_basis(2)
    psi

    # 密集编码演示开始：

    # 假设 Alice 拥有量子叠加态中左边的量子比特
    print("An even superposition of 2 qubits.  Assume Alice has the left QBit.")
    pprint(psi)

    # Alice 量子比特上的门操作：
    # 单位门 (1), 反门 (X), Z 门 (Z), Z 门和反门 (ZX)
    # 然后是控制非门（以 Alice 的量子比特为控制）：CNOT(1, 0)
    # 最后是应用于 Alice 量子比特的哈达玛门：H(1)

    # 发送消息 |0>|0> 给 Bob
    print("To Send Bob the message |00>.")
    circuit = H(1)*CNOT(1, 0)
    result = qapply(circuit*psi)
    result
    pprint(result)

    # 发送消息 |0>|1> 给 Bob
    print("To Send Bob the message |01>.")
    circuit = H(1)*CNOT(1, 0)*X(1)
    result = qapply(circuit*psi)
    result
    pprint(result)

    # 发送消息 |1>|0> 给 Bob
    print("To Send Bob the message |10>.")
    circuit = H(1)*CNOT(1, 0)*Z(1)
    result = qapply(circuit*psi)
    result
    pprint(result)

    # 发送消息 |1>|1> 给 Bob
    print("To Send Bob the message |11>.")
    circuit = H(1)*CNOT(1, 0)*Z(1)*X(1)
    result = qapply(circuit*psi)
    result
    pprint(result)

# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```