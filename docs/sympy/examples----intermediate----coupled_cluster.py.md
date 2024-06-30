# `D:\src\scipysrc\sympy\examples\intermediate\coupled_cluster.py`

```
#!/usr/bin/env python

"""
Calculates the Coupled-Cluster energy- and amplitude equations
See 'An Introduction to Coupled Cluster Theory' by
T. Daniel Crawford and Henry F. Schaefer III.

Other Resource : http://vergil.chemistry.gatech.edu/notes/sahan-cc-2010.pdf
"""

# 导入必要的库和模块
from sympy.physics.secondquant import (AntiSymmetricTensor, wicks,
        F, Fd, NO, evaluate_deltas, substitute_dummies, Commutator,
        simplify_index_permutations, PermutationOperator)
from sympy import (
    symbols, Rational, latex, Dummy
)

# 定义用于美化指标的字典
pretty_dummies_dict = {
    'above': 'cdefgh',
    'below': 'klmno',
    'general': 'pqrstu'
}

# 定义获取 Coupled-Cluster 运算符的函数
def get_CC_operators():
    """
    Returns a tuple (T1,T2) of unique operators.
    """
    # 定义符号并创建对应的反对称张量和正规序算符
    i = symbols('i', below_fermi=True, cls=Dummy)
    a = symbols('a', above_fermi=True, cls=Dummy)
    t_ai = AntiSymmetricTensor('t', (a,), (i,))
    ai = NO(Fd(a)*F(i))

    # 定义更复杂的情况下的反对称张量和正规序算符
    i, j = symbols('i,j', below_fermi=True, cls=Dummy)
    a, b = symbols('a,b', above_fermi=True, cls=Dummy)
    t_abij = AntiSymmetricTensor('t', (a, b), (i, j))
    abji = NO(Fd(a)*Fd(b)*F(j)*F(i))

    # 创建并返回 T1 和 T2 运算符
    T1 = t_ai*ai
    T2 = Rational(1, 4)*t_abij*abji
    return (T1, T2)

# 主函数入口
def main():
    print()
    print("Calculates the Coupled-Cluster energy- and amplitude equations")
    print("See 'An Introduction to Coupled Cluster Theory' by")
    print("T. Daniel Crawford and Henry F. Schaefer III")
    print("Reference to a Lecture Series: http://vergil.chemistry.gatech.edu/notes/sahan-cc-2010.pdf")
    print()

    # 设置哈密顿量
    p, q, r, s = symbols('p,q,r,s', cls=Dummy)
    f = AntiSymmetricTensor('f', (p,), (q,))
    pr = NO(Fd(p)*F(q))
    v = AntiSymmetricTensor('v', (p, q), (r, s))
    pqsr = NO(Fd(p)*Fd(q)*F(s)*F(r))

    H = f*pr + Rational(1, 4)*v*pqsr
    print("Using the hamiltonian:", latex(H))

    print("Calculating 4 nested commutators")
    C = Commutator

    # 计算第一个交换子
    T1, T2 = get_CC_operators()
    T = T1 + T2
    print("commutator 1...")
    comm1 = wicks(C(H, T))
    comm1 = evaluate_deltas(comm1)
    comm1 = substitute_dummies(comm1)

    # 计算第二个交换子
    T1, T2 = get_CC_operators()
    T = T1 + T2
    print("commutator 2...")
    comm2 = wicks(C(comm1, T))
    comm2 = evaluate_deltas(comm2)
    comm2 = substitute_dummies(comm2)

    # 计算第三个交换子
    T1, T2 = get_CC_operators()
    T = T1 + T2
    print("commutator 3...")
    comm3 = wicks(C(comm2, T))
    comm3 = evaluate_deltas(comm3)
    comm3 = substitute_dummies(comm3)

    # 计算第四个交换子
    T1, T2 = get_CC_operators()
    T = T1 + T2
    print("commutator 4...")
    comm4 = wicks(C(comm3, T))
    comm4 = evaluate_deltas(comm4)
    comm4 = substitute_dummies(comm4)

    # 构建哈斯道夫展开式
    print("construct Hausdorff expansion...")
    eq = H + comm1 + comm2/2 + comm3/6 + comm4/24
    eq = eq.expand()
    eq = evaluate_deltas(eq)
    eq = substitute_dummies(eq, new_indices=True,
            pretty_indices=pretty_dummies_dict)
    print("*********************")
    print()

    # 提取从完整的 Hbar 中得到的 CC 方程
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    # 定义符号变量 a, b, c, d，并指定它们位于费米面以上
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    
    # 打印空行
    print()
    
    # 打印 "CC Energy:"，用 LaTeX 格式打印方程 eq，并简化虚拟指标
    print("CC Energy:")
    print(latex(wicks(eq, simplify_dummies=True,
        keep_only_fully_contracted=True)))
    print()
    
    # 打印 "CC T1:"，计算并用 LaTeX 格式打印经过 Wick 缩并的 T1 方程 eqT1
    eqT1 = wicks(NO(Fd(i)*F(a))*eq, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
    eqT1 = substitute_dummies(eqT1)
    print(latex(eqT1))
    print()
    
    # 打印 "CC T2:"，计算并用 LaTeX 格式打印经过 Wick 缩并的 T2 方程 eqT2
    eqT2 = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*eq, simplify_dummies=True, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    P = PermutationOperator
    # 对指标进行排列，简化等价的指标排列
    eqT2 = simplify_index_permutations(eqT2, [P(a, b), P(i, j)])
    print(latex(eqT2))
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块
    main()
```