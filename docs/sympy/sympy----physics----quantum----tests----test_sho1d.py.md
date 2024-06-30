# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_sho1d.py`

```
"""Tests for sho1d.py"""

# 导入符号计算库中所需的模块和函数
from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import Commutator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.represent import represent
from sympy.external import import_module
from sympy.testing.pytest import skip

# 导入量子力学中的一维谐振子相关类和函数
from sympy.physics.quantum.sho1d import (RaisingOp, LoweringOp,
                                        SHOKet, SHOBra,
                                        Hamiltonian, NumberOp)

# 创建升算符、降算符、态矢量、态矢共轭、哈密顿量和数算符的实例
ad = RaisingOp('a')
a = LoweringOp('a')
k = SHOKet('k')
kz = SHOKet(0)
kf = SHOKet(1)
k3 = SHOKet(3)
b = SHOBra('b')
b3 = SHOBra(3)
H = Hamiltonian('H')
N = NumberOp('N')
omega = Symbol('omega')
m = Symbol('m')
ndim = Integer(4)

# 导入 numpy 和 scipy 库，如果没有安装则跳过相关测试
np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})

# 使用符号计算库中的 represent 函数获取各种算符的表示形式
ad_rep_sympy = represent(ad, basis=N, ndim=4, format='sympy')
a_rep = represent(a, basis=N, ndim=4, format='sympy')
N_rep = represent(N, basis=N, ndim=4, format='sympy')
H_rep = represent(H, basis=N, ndim=4, format='sympy')
k3_rep = represent(k3, basis=N, ndim=4, format='sympy')
b3_rep = represent(b3, basis=N, ndim=4, format='sympy')

# 定义测试函数 test_RaisingOp，测试升算符的各种性质和方法
def test_RaisingOp():
    assert Dagger(ad) == a  # 测试升算符的共轭操作是否得到降算符
    assert Commutator(ad, a).doit() == Integer(-1)  # 测试升算符和降算符的对易子
    assert Commutator(ad, N).doit() == Integer(-1)*ad  # 测试升算符和数算符的对易子
    assert qapply(ad*k) == (sqrt(k.n + 1)*SHOKet(k.n + 1)).expand()  # 测试升算符作用在态矢上的效果
    assert qapply(ad*kz) == (sqrt(kz.n + 1)*SHOKet(kz.n + 1)).expand()  # 测试升算符作用在基态上的效果
    assert qapply(ad*kf) == (sqrt(kf.n + 1)*SHOKet(kf.n + 1)).expand()  # 测试升算符作用在第一激发态上的效果
    assert ad.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(Integer(-1)*I*Px + m*omega*X)  # 测试升算符在位置动量表象下的重写
    assert ad.hilbert_space == ComplexSpace(S.Infinity)  # 测试升算符所在的希尔伯特空间
    for i in range(ndim - 1):
        assert ad_rep_sympy[i + 1,i] == sqrt(i + 1)  # 测试升算符在指定基底下的表示形式是否正确

    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过接下来的测试

    ad_rep_numpy = represent(ad, basis=N, ndim=4, format='numpy')
    for i in range(ndim - 1):
        assert ad_rep_numpy[i + 1,i] == float(sqrt(i + 1))  # 测试升算符在 numpy 数组中的表示形式是否正确

    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过接下来的测试
    if not scipy:
        skip("scipy not installed.")  # 如果没有安装 scipy，则跳过接下来的测试

    ad_rep_scipy = represent(ad, basis=N, ndim=4, format='scipy.sparse', spmatrix='lil')
    for i in range(ndim - 1):
        assert ad_rep_scipy[i + 1,i] == float(sqrt(i + 1))  # 测试升算符在 scipy 稀疏矩阵中的表示形式是否正确

    assert ad_rep_numpy.dtype == 'float64'  # 测试 numpy 数组的数据类型是否为 float64
    assert ad_rep_scipy.dtype == 'float64'  # 测试 scipy 稀疏矩阵的数据类型是否为 float64

# 定义测试函数 test_LoweringOp，测试降算符的各种性质和方法
def test_LoweringOp():
    assert Dagger(a) == ad  # 测试降算符的共轭操作是否得到升算符
    assert Commutator(a, ad).doit() == Integer(1)  # 测试降算符和升算符的对易子
    assert Commutator(a, N).doit() == a  # 测试降算符和数算符的对易子
    # 断言，验证操作符a乘以标量k后的结果是否等于(sqrt(k.n)*SHOKet(k.n-Integer(1)))的展开形式
    assert qapply(a*k) == (sqrt(k.n)*SHOKet(k.n-Integer(1))).expand()
    
    # 断言，验证操作符a乘以Ket符号kz后的结果是否等于整数0
    assert qapply(a*kz) == Integer(0)
    
    # 断言，验证操作符a乘以标量kf后的结果是否等于(sqrt(kf.n)*SHOKet(kf.n-Integer(1)))的展开形式
    assert qapply(a*kf) == (sqrt(kf.n)*SHOKet(kf.n-Integer(1))).expand()
    
    # 断言，验证操作符a应用rewrite('xp')转换后是否等于给定表达式的结果
    assert a.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(I*Px + m*omega*X)
    
    # 遍历范围为ndim - 1的整数序列，对每个i进行断言
    for i in range(ndim - 1):
        # 断言，验证操作符a_rep[i,i + 1]是否等于sqrt(i + 1)
        assert a_rep[i,i + 1] == sqrt(i + 1)
# 定义测试函数 test_NumberOp，用于测试 NumberOp 类的功能
def test_NumberOp():
    # 断言：计算 N 和 ad 的对易子，并检查结果是否等于 ad
    assert Commutator(N, ad).doit() == ad
    # 断言：计算 N 和 a 的对易子，并检查结果是否等于 -a
    assert Commutator(N, a).doit() == Integer(-1)*a
    # 断言：计算 N 和 H 的对易子，并检查结果是否等于 0
    assert Commutator(N, H).doit() == Integer(0)
    # 断言：应用算符 N 到 k，检查结果是否等于 (k.n * k) 的展开形式
    assert qapply(N*k) == (k.n*k).expand()
    # 断言：对 N 应用 'a' 重写规则后进行求值，检查结果是否等于 ad * a
    assert N.rewrite('a').doit() == ad*a
    # 断言：对 N 应用 'xp' 重写规则后进行求值，检查结果是否符合给定表达式
    assert N.rewrite('xp').doit() == (Integer(1)/(Integer(2)*m*hbar*omega))*(
        Px**2 + (m*omega*X)**2) - Integer(1)/Integer(2)
    # 断言：对 N 应用 'H' 重写规则后进行求值，检查结果是否符合给定表达式
    assert N.rewrite('H').doit() == H/(hbar*omega) - Integer(1)/Integer(2)
    # 遍历 ndim 范围内的每一个索引 i
    for i in range(ndim):
        # 断言：检查 N_rep[i,i] 是否等于 i
        assert N_rep[i,i] == i
    # 断言：检查 N_rep 是否等于 ad_rep_sympy * a_rep

# 定义测试函数 test_Hamiltonian，用于测试 Hamiltonian 类的功能
def test_Hamiltonian():
    # 断言：计算 H 和 N 的对易子，并检查结果是否等于 0
    assert Commutator(H, N).doit() == Integer(0)
    # 断言：应用算符 H 到 k，检查结果是否符合给定表达式的展开形式
    assert qapply(H*k) == ((hbar*omega*(k.n + Integer(1)/Integer(2)))*k).expand()
    # 断言：对 H 应用 'a' 重写规则后进行求值，检查结果是否符合给定表达式
    assert H.rewrite('a').doit() == hbar*omega*(ad*a + Integer(1)/Integer(2))
    # 断言：对 H 应用 'xp' 重写规则后进行求值，检查结果是否符合给定表达式
    assert H.rewrite('xp').doit() == \
        (Integer(1)/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)
    # 断言：对 H 应用 'N' 重写规则后进行求值，检查结果是否符合给定表达式
    assert H.rewrite('N').doit() == hbar*omega*(N + Integer(1)/Integer(2))
    # 遍历 ndim 范围内的每一个索引 i
    for i in range(ndim):
        # 断言：检查 H_rep[i,i] 是否等于给定表达式
        assert H_rep[i,i] == hbar*omega*(i + Integer(1)/Integer(2))

# 定义测试函数 test_SHOKet，用于测试 SHOKet 类的功能
def test_SHOKet():
    # 断言：检查 'k' 类型的 SHOKet 实例的 dual_class 方法返回结果是否为 SHOBra 类
    assert SHOKet('k').dual_class() == SHOBra
    # 断言：检查 'b' 类型的 SHOBra 实例的 dual_class 方法返回结果是否为 SHOKet 类
    assert SHOBra('b').dual_class() == SHOKet
    # 断言：计算 b 和 k 的内积，并检查结果是否等于 KroneckerDelta(k.n, b.n)
    assert InnerProduct(b,k).doit() == KroneckerDelta(k.n, b.n)
    # 断言：检查 k 的 Hilbert 空间是否等于 ComplexSpace(S.Infinity)
    assert k.hilbert_space == ComplexSpace(S.Infinity)
    # 断言：检查 k3_rep[k3.n, 0] 是否等于 1
    assert k3_rep[k3.n, 0] == Integer(1)
    # 断言：检查 b3_rep[0, b3.n] 是否等于 1
    assert b3_rep[0, b3.n] == Integer(1)
```