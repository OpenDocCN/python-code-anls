# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_residue.py`

```
# 导入必要的模块和函数
from collections import defaultdict  # 导入 defaultdict 类来创建默认字典
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.core.singleton import S  # 导入 S 单例对象
from sympy.core.symbol import (Dummy, Symbol)  # 导入 Dummy 和 Symbol 符号类
from sympy.functions.combinatorial.numbers import totient  # 导入计算欧拉函数的函数
from sympy.ntheory import (  # 导入数论相关的函数和类
    n_order, is_primitive_root, is_quad_residue,
    legendre_symbol, jacobi_symbol, primerange, sqrt_mod,
    primitive_root, quadratic_residues, is_nthpow_residue, nthroot_mod,
    sqrt_mod_iter, mobius, discrete_log, quadratic_congruence,
    polynomial_congruence, sieve
)
from sympy.ntheory.residue_ntheory import (  # 导入更高级的数论函数和类
    _primitive_root_prime_iter,
    _primitive_root_prime_power_iter, _primitive_root_prime_power2_iter,
    _nthroot_mod_prime_power, _discrete_log_trial_mul, _discrete_log_shanks_steps,
    _discrete_log_pollard_rho, _discrete_log_index_calculus, _discrete_log_pohlig_hellman,
    _binomial_mod_prime_power, binomial_mod
)
from sympy.polys.domains import ZZ  # 导入整数环 ZZ
from sympy.testing.pytest import raises  # 导入 pytest 中的 raises 函数
from sympy.core.random import randint, choice  # 导入随机整数生成和随机选择函数


def test_residue():
    # 检查 n_order 函数的计算结果
    assert n_order(2, 13) == 12
    # 检查多个 a 值下 n_order 函数的计算结果
    assert [n_order(a, 7) for a in range(1, 7)] == \
           [1, 3, 6, 3, 6, 2]
    assert n_order(5, 17) == 16
    # 检查两个数的 n_order 是否相等
    assert n_order(17, 11) == n_order(6, 11)
    # 检查 n_order 函数对大数的计算结果
    assert n_order(101, 119) == 6
    assert n_order(11, (10**50 + 151)**2) == 10000000000000000000000000000000000000000000000030100000000000000000000000000000000000000000000022650
    # 检查 n_order 函数对非素数的抛出异常
    raises(ValueError, lambda: n_order(6, 9))

    # 检查 is_primitive_root 函数的计算结果
    assert is_primitive_root(2, 7) is False
    assert is_primitive_root(3, 8) is False
    assert is_primitive_root(11, 14) is False
    # 检查两个数的 is_primitive_root 是否相等
    assert is_primitive_root(12, 17) == is_primitive_root(29, 17)
    # 检查 is_primitive_root 函数对非素数的抛出异常
    raises(ValueError, lambda: is_primitive_root(3, 6))

    # 遍历素数范围内的每个素数 p
    for p in primerange(3, 100):
        # 获取 p 的所有原根
        li = list(_primitive_root_prime_iter(p))
        # 检查最小的原根是否为列表中的第一个元素
        assert li[0] == min(li)
        # 检查每个原根的 n_order 是否等于 p - 1
        for g in li:
            assert n_order(g, p) == p - 1
        # 检查原根的数量是否等于 totient(totient(p))
        assert len(li) == totient(totient(p))
        # 对每个指数 e 进行测试
        for e in range(1, 4):
            # 获取不同指数下的原根列表
            li_power = list(_primitive_root_prime_power_iter(p, e))
            li_power2 = list(_primitive_root_prime_power2_iter(p, e))
            # 检查两种不同迭代方法得到的原根列表长度是否相等，并且与 totient(totient(p**e)) 相等
            assert len(li_power) == len(li_power2) == totient(totient(p**e))
    # 检查给定数的原根
    assert primitive_root(97) == 5
    assert n_order(primitive_root(97, False), 97) == totient(97)
    assert primitive_root(97**2) == 5
    assert n_order(primitive_root(97**2, False), 97**2) == totient(97**2)
    assert primitive_root(40487) == 5
    assert n_order(primitive_root(40487, False), 40487) == totient(40487)
    # note that primitive_root(40487) + 40487 = 40492 is a primitive root
    # of 40487**2, but it is not the smallest
    assert primitive_root(40487**2) == 10
    assert n_order(primitive_root(40487**2, False), 40487**2) == totient(40487**2)
    assert primitive_root(82) == 7
    assert n_order(primitive_root(82, False), 82) == totient(82)
    # 检查大数的原根和 n_order 计算
    p = 10**50 + 151
    assert primitive_root(p) == 11
    assert n_order(primitive_root(p, False), p) == totient(p)
    # 断言，验证 2*p 的原根为 11
    assert primitive_root(2*p) == 11
    # 断言，验证在模 2*p 中，11 的阶数与欧拉函数 totient(2*p) 相等
    assert n_order(primitive_root(2*p, False), 2*p) == totient(2*p)
    # 断言，验证 p^2 的原根为 11
    assert primitive_root(p**2) == 11
    # 断言，验证在模 p^2 中，11 的阶数与欧拉函数 totient(p^2) 相等
    assert n_order(primitive_root(p**2, False), p**2) == totient(p**2)
    # 断言，验证 4*11 的原根和其非原根都为 None
    assert primitive_root(4 * 11) is None and primitive_root(4 * 11, False) is None
    # 断言，验证 15 的原根和其非原根都为 None
    assert primitive_root(15) is None and primitive_root(15, False) is None
    # 断言，验证对于负数的参数抛出 ValueError 异常
    raises(ValueError, lambda: primitive_root(-3))

    # 断言，验证在模 7 中，3 不是二次剩余
    assert is_quad_residue(3, 7) is False
    # 断言，验证在模 13 中，10 是二次剩余
    assert is_quad_residue(10, 13) is True
    # 断言，验证同余情况下的二次剩余性质
    assert is_quad_residue(12364, 139) == is_quad_residue(12364 % 139, 139)
    # 断言，验证在模 251 中，207 是二次剩余
    assert is_quad_residue(207, 251) is True
    # 断言，验证在模 1 中，0 是二次剩余
    assert is_quad_residue(0, 1) is True
    # 断言，验证在模 1 中，1 是二次剩余
    assert is_quad_residue(1, 1) is True
    # 断言，验证在模 2 中，0 和 1 都是二次剩余
    assert is_quad_residue(0, 2) == is_quad_residue(1, 2) is True
    # 断言，验证在模 4 中，1 是二次剩余
    assert is_quad_residue(1, 4) is True
    # 断言，验证在模 27 中，2 不是二次剩余
    assert is_quad_residue(2, 27) is False
    # 断言，验证在模 13604889600 中，13122380800 是二次剩余
    assert is_quad_residue(13122380800, 13604889600) is True
    # 断言，验证在模 14 中，所有二次剩余的列表
    assert [j for j in range(14) if is_quad_residue(j, 14)] == \
           [0, 1, 2, 4, 7, 8, 9, 11]
    # 断言，验证对于浮点数参数抛出 ValueError 异常
    raises(ValueError, lambda: is_quad_residue(1.1, 2))
    # 断言，验证对于模数为 0 的参数抛出 ValueError 异常
    raises(ValueError, lambda: is_quad_residue(2, 0))

    # 断言，验证 S.One 作为参数时的二次剩余列表
    assert quadratic_residues(S.One) == [0]
    # 断言，验证 1 作为参数时的二次剩余列表
    assert quadratic_residues(1) == [0]
    # 断言，验证 12 作为参数时的二次剩余列表
    assert quadratic_residues(12) == [0, 1, 4, 9]
    # 断言，验证 13 作为参数时的二次剩余列表
    assert quadratic_residues(13) == [0, 1, 3, 4, 9, 10, 12]
    # 断言，验证前 20 个自然数对应的二次剩余列表长度
    assert [len(quadratic_residues(i)) for i in range(1, 20)] == \
      [1, 2, 2, 2, 3, 4, 4, 3, 4, 6, 6, 4, 7, 8, 6, 4, 9, 8, 10]

    # 断言，验证 sqrt_mod_iter 函数对于参数 6 和 2 的迭代结果
    assert list(sqrt_mod_iter(6, 2)) == [0]
    # 断言，验证在模 13 中，3 的平方根为 4
    assert sqrt_mod(3, 13) == 4
    # 断言，验证在模 -13 中，3 的平方根为 4
    assert sqrt_mod(3, -13) == 4
    # 断言，验证在模 23 中，6 的平方根为 11
    assert sqrt_mod(6, 23) == 11
    # 断言，验证在模 690 中，345 的平方根为 345
    assert sqrt_mod(345, 690) == 345
    # 断言，验证在模 101 中，67 没有平方根
    assert sqrt_mod(67, 101) == None
    # 断言，验证在模 104729 中，1020 没有平方根
    assert sqrt_mod(1020, 104729) == None

    # 循环，验证在各个素数 p 下的平方根模运算特性
    for p in range(3, 100):
        # 创建一个 defaultdict，存储每个 i 的平方根集合
        d = defaultdict(list)
        for i in range(p):
            d[pow(i, 2, p)].append(i)
        # 遍历每个 i 的情况
        for i in range(1, p):
            # 获取平方根迭代器和直接求解结果
            it = sqrt_mod_iter(i, p)
            v = sqrt_mod(i, p, True)
            if v:
                # 如果有解，则排序并与 defaultdict 中的值比较
                v = sorted(v)
                assert d[i] == v
            else:
                # 如果无解，则检查 defaultdict 中是否为空
                assert not d[i]

    # 断言，验证在模 27 中，9 的平方根集合
    assert sqrt_mod(9, 27, True) == [3, 6, 12, 15, 21, 24]
    # 断言，验证在模 81 中，9 的平方根集合
    assert sqrt_mod(9, 81, True) == [3, 24, 30, 51, 57, 78]
    # 断言，验证在模 3^5 中，9 的平方根集合
    assert sqrt_mod(9, 3**5, True) == [3, 78, 84, 159, 165, 240]
    # 断言，验证在模 3^4 中，81 的平方根集合
    assert sqrt_mod(81, 3**4, True) == [0, 9, 18, 27, 36, 45, 54, 63, 72]
    # 断言，验证在模 3^5 中，81 的平方根集合
    assert sqrt_mod(81, 3**5, True) == [9, 18, 36, 45, 63, 72, 90, 99, 117,\
            126, 144, 153, 171, 180, 198, 207, 225, 234]
    # 断言，验证在模 3^6 中，81 的平方根集合
    assert sqrt_mod(81, 3**6, True) == [9, 72, 90, 153, 171, 234, 252, 315,\
            333, 396, 414, 477, 495, 558, 576, 639, 657, 720]
    # 断言，验证在模 3^7 中，81 的平方根集合
    # 计算变量 a 和 p 的值，用于测试平方根模迭代器
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+2)
    # 使用给定的 a 和 p 创建平方根模迭代器
    it = sqrt_mod_iter(a, p)
    # 对平方根模迭代器进行十次迭代，并断言每次迭代的结果的平方模 p 等于 a
    for i in range(10):
        assert pow(next(it), 2, p) == a

    # 重新计算变量 a 和 p 的值，用于第二轮测试
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+3)
    # 使用给定的 a 和 p 创建平方根模迭代器
    it = sqrt_mod_iter(a, p)
    # 对平方根模迭代器进行两次迭代，并断言每次迭代的结果的平方模 p 等于 a
    for i in range(2):
        assert pow(next(it), 2, p) == a

    # 设置变量 n 的值为 100
    n = 100
    # 重新计算变量 a 和 p 的值，用于第三轮测试
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+1)
    # 使用给定的 a 和 p 创建平方根模迭代器
    it = sqrt_mod_iter(a, p)
    # 对平方根模迭代器进行两次迭代，并断言每次迭代的结果的平方模 p 等于 a
    for i in range(2):
        assert pow(next(it), 2, p) == a

    # 断言 next(sqrt_mod_iter(9, 27)) 返回的结果类型为 int
    assert type(next(sqrt_mod_iter(9, 27))) is int
    # 断言 next(sqrt_mod_iter(9, 27, ZZ)) 返回的结果类型与 ZZ(1) 的类型相同
    assert type(next(sqrt_mod_iter(9, 27, ZZ))) is type(ZZ(1))
    # 断言 next(sqrt_mod_iter(1, 7, ZZ)) 返回的结果类型与 ZZ(1) 的类型相同
    assert type(next(sqrt_mod_iter(1, 7, ZZ))) is type(ZZ(1))

    # 断言 is_nthpow_residue(2, 1, 5) 返回 True
    assert is_nthpow_residue(2, 1, 5)

    # issue 10816 相关断言
    assert is_nthpow_residue(1, 0, 1) is False
    assert is_nthpow_residue(1, 0, 2) is True
    assert is_nthpow_residue(3, 0, 2) is True
    assert is_nthpow_residue(0, 1, 8) is True
    assert is_nthpow_residue(2, 3, 2) is True
    assert is_nthpow_residue(2, 3, 9) is False
    assert is_nthpow_residue(3, 5, 30) is True
    assert is_nthpow_residue(21, 11, 20) is True
    assert is_nthpow_residue(7, 10, 20) is False
    assert is_nthpow_residue(5, 10, 20) is True
    assert is_nthpow_residue(3, 10, 48) is False
    assert is_nthpow_residue(1, 10, 40) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(1, 10, 24) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(2, 10, 48) is False
    assert is_nthpow_residue(81, 3, 972) is False
    assert is_nthpow_residue(243, 5, 5103) is True
    assert is_nthpow_residue(243, 3, 1240029) is False
    assert is_nthpow_residue(36010, 8, 87382) is True
    assert is_nthpow_residue(28552, 6, 2218) is True
    assert is_nthpow_residue(92712, 9, 50026) is True

    # 使用集合推导式生成 a 的 56 次幂对 1024 取模的集合
    x = {pow(i, 56, 1024) for i in range(1024)}
    # 断言 is_nthpow_residue(a, 56, 1024) 为真的 a 值集合与 x 相等
    assert {a for a in range(1024) if is_nthpow_residue(a, 56, 1024)} == x

    # 使用集合推导式生成 a 的 256 次幂对 2048 取模的集合
    x = {pow(i, 256, 2048) for i in range(2048)}
    # 断言 is_nthpow_residue(a, 256, 2048) 为真的 a 值集合与 x 相等
    assert {a for a in range(2048) if is_nthpow_residue(a, 256, 2048)} == x

    # 使用集合推导式生成 a 的 11 次幂对 324000 取模的集合，并生成结果列表
    x = {pow(i, 11, 324000) for i in range(1000)}
    assert [ is_nthpow_residue(a, 11, 324000) for a in x]

    # 使用集合推导式生成 a 的 17 次幂对 22217575536 取模的集合，并生成结果列表
    x = {pow(i, 17, 22217575536) for i in range(1000)}
    assert [ is_nthpow_residue(a, 17, 22217575536) for a in x]

    # 单独的 is_nthpow_residue 断言测试
    assert is_nthpow_residue(676, 3, 5364)
    assert is_nthpow_residue(9, 12, 36)
    assert is_nthpow_residue(32, 10, 41)
    assert is_nthpow_residue(4, 2, 64)
    assert is_nthpow_residue(31, 4, 41)
    assert not is_nthpow_residue(2, 2, 5)
    assert is_nthpow_residue(8547, 12, 10007)
    assert is_nthpow_residue(Dummy(even=True) + 3, 3, 2) == True
    # 使用 primerange 函数生成素数范围 [2, 10)，对每个素数执行以下操作：
    for p in primerange(2, 10):
        # 对每个素数执行三次循环
        for a in range(3):
            # 对 n 在范围 [3, 5) 中执行循环
            for n in range(3, 5):
                # 调用 _nthroot_mod_prime_power 函数计算模 p 的第 n 次根
                ans = _nthroot_mod_prime_power(a, n, p, 1)
                # 断言返回值为列表类型
                assert isinstance(ans, list)
                # 如果返回列表为空
                if len(ans) == 0:
                    # 对所有可能的 b 在模 p 的情况下进行断言
                    for b in range(p):
                        assert pow(b, n, p) != a % p
                    # 对 k 在 [2, 10) 范围内进行断言，函数应返回空列表
                    for k in range(2, 10):
                        assert _nthroot_mod_prime_power(a, n, p, k) == []
                else:
                    # 对所有可能的 b 在模 p 的情况下进行断言
                    for b in range(p):
                        pred = pow(b, n, p) == a % p
                        assert not(pred ^ (b in ans))
                    # 对 k 在 [2, 10) 范围内进行断言，返回结果应与第一次调用一致
                    for k in range(2, 10):
                        ans = _nthroot_mod_prime_power(a, n, p, k)
                        if not ans:
                            break
                        # 对返回的每个 b 进行断言，满足模 p^k 的条件
                        for b in ans:
                            assert pow(b, n , p**k) == a

    # 断言使用 Dummy 类创建的对象调用 nthroot_mod 函数返回值为 1
    assert nthroot_mod(Dummy(odd=True), 3, 2) == 1
    # 断言调用 nthroot_mod 函数返回值为 45
    assert nthroot_mod(29, 31, 74) == 45
    # 断言调用 nthroot_mod 函数返回值为 44
    assert nthroot_mod(1801, 11, 2663) == 44

    # 对每组 (a, q, p) 进行断言，调用 nthroot_mod 函数返回值满足条件
    for a, q, p in [(51922, 2, 203017), (43, 3, 109), (1801, 11, 2663),
          (26118163, 1303, 33333347), (1499, 7, 2663), (595, 6, 2663),
          (1714, 12, 2663), (28477, 9, 33343)]:
        r = nthroot_mod(a, q, p)
        # 断言 r 的 q 次幂模 p 等于 a
        assert pow(r, q, p) == a

    # 断言调用 nthroot_mod 函数返回 None
    assert nthroot_mod(11, 3, 109) is None
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(16, 5, 36, True) == [4, 22]
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(9, 16, 36, True) == [3, 9, 15, 21, 27, 33]
    # 断言调用 nthroot_mod 函数返回 None
    assert nthroot_mod(4, 3, 3249000) is None
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(36010, 8, 87382, True) == [40208, 47174]
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(0, 12, 37, True) == [0]
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(0, 7, 100, True) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(4, 4, 27, True) == [5, 22]
    # 断言调用 nthroot_mod 函数返回列表，满足条件
    assert nthroot_mod(4, 4, 121, True) == [19, 102]
    # 断言调用 nthroot_mod 函数返回空列表
    assert nthroot_mod(2, 3, 7, True) == []

    # 对所有 p 在 [1, 20) 范围内，对每个 (a, n, p) 组合调用 nthroot_mod 函数，并进行断言
    for p in range(1, 20):
        for a in range(p):
            for n in range(1, p):
                ans = nthroot_mod(a, n, p, True)
                assert isinstance(ans, list)
                # 对返回的列表中的每个 b 进行断言，满足条件
                for b in range(p):
                    pred = pow(b, n, p) == a
                    assert not(pred ^ (b in ans))
                # 对返回空列表进行断言
                ans2 = nthroot_mod(a, n, p, False)
                if ans2 is None:
                    assert ans == []
                else:
                    assert ans2 in ans

    # 创建两个符号变量，并进行断言
    x = Symbol('x', positive=True)
    i = Symbol('i', integer=True)
    assert _discrete_log_trial_mul(587, 2**7, 2) == 7
    assert _discrete_log_trial_mul(941, 7**18, 7) == 18
    assert _discrete_log_trial_mul(389, 3**81, 3) == 81
    assert _discrete_log_trial_mul(191, 19**123, 19) == 123
    assert _discrete_log_shanks_steps(442879, 7**2, 7) == 2
    assert _discrete_log_shanks_steps(874323, 5**19, 5) == 19
    assert _discrete_log_shanks_steps(6876342, 7**71, 7) == 71
    assert _discrete_log_shanks_steps(2456747, 3**321, 3) == 321
    assert _discrete_log_pollard_rho(6013199, 2**6, 2, rseed=0) == 6
    # 使用 Pollard Rho 方法计算离散对数，确保对数为 19
    assert _discrete_log_pollard_rho(6138719, 2**19, 2, rseed=0) == 19
    # 使用 Pollard Rho 方法计算离散对数，确保对数为 40
    assert _discrete_log_pollard_rho(36721943, 2**40, 2, rseed=0) == 40
    # 使用 Pollard Rho 方法计算离散对数，确保对数为 333
    assert _discrete_log_pollard_rho(24567899, 3**333, 3, rseed=0) == 333
    # 使用 Pollard Rho 方法预期引发 ValueError，因为基数和模数没有公因数
    raises(ValueError, lambda: _discrete_log_pollard_rho(11, 7, 31, rseed=0))
    # 使用 Pollard Rho 方法预期引发 ValueError，因为基数和模数没有公因数
    raises(ValueError, lambda: _discrete_log_pollard_rho(227, 3**7, 5, rseed=0))
    # 使用 Index Calculus 方法计算离散对数，确保对数为 183
    assert _discrete_log_index_calculus(983, 948, 2, 491) == 183
    # 使用 Index Calculus 方法计算离散对数，确保对数为 68048
    assert _discrete_log_index_calculus(633383, 21794, 2, 316691) == 68048
    # 使用 Index Calculus 方法计算离散对数，确保对数为 338029275
    assert _discrete_log_index_calculus(941762639, 68822582, 2, 470881319) == 338029275
    # 使用 Index Calculus 方法计算离散对数，确保对数为 142811376514
    assert _discrete_log_index_calculus(999231337607, 888188918786, 2, 499615668803) == 142811376514
    # 使用 Index Calculus 方法计算离散对数，确保对数为 590504662
    assert _discrete_log_index_calculus(47747730623, 19410045286, 43425105668, 645239603) == 590504662
    # 使用 Pohlig-Hellman 方法计算离散对数，确保对数为 9
    assert _discrete_log_pohlig_hellman(98376431, 11**9, 11) == 9
    # 使用 Pohlig-Hellman 方法计算离散对数，确保对数为 31
    assert _discrete_log_pohlig_hellman(78723213, 11**31, 11) == 31
    # 使用 Pohlig-Hellman 方法计算离散对数，确保对数为 98
    assert _discrete_log_pohlig_hellman(32942478, 11**98, 11) == 98
    # 使用 Pohlig-Hellman 方法计算离散对数，确保对数为 444
    assert _discrete_log_pohlig_hellman(14789363, 11**444, 11) == 444
    # 使用通用方法计算离散对数，确保对数为 9
    assert discrete_log(587, 2**9, 2) == 9
    # 使用通用方法计算离散对数，确保对数为 51
    assert discrete_log(2456747, 3**51, 3) == 51
    # 使用通用方法计算离散对数，确保对数为 127
    assert discrete_log(32942478, 11**127, 11) == 127
    # 使用通用方法计算离散对数，确保对数为 324
    assert discrete_log(432751500361, 7**324, 7) == 324
    # 使用通用方法计算离散对数，确保对数为 17835221372061
    assert discrete_log(265390227570863, 184500076053622, 2) == 17835221372061
    # 使用通用方法计算离散对数，确保对数为 2068031853682195777930683306640554533145512201725884603914601918777510185469769997054750835368413389728895
    assert discrete_log(22708823198678103974314518195029102158525052496759285596453269189798311427475159776411276642277139650833937,
                        17463946429475485293747680247507700244427944625055089103624311227422110546803452417458985046168310373075327,
                        123456) == 2068031853682195777930683306640554533145512201725884603914601918777510185469769997054750835368413389728895
    # 通过参数元组来计算离散对数，确保对数为 687
    args = 5779, 3528, 6215
    assert discrete_log(*args) == 687
    # 通过参数元组的 Tuple 类型来计算离散对数，确保对数为 687
    assert discrete_log(*Tuple(*args)) == 687
    # 解决二次同余方程，确保解为 [295, 615, 935, 1255, 1575]
    assert quadratic_congruence(400, 85, 125, 1600) == [295, 615, 935, 1255, 1575]
    # 解决二次同余方程，确保解为 [3, 20]
    assert quadratic_congruence(3, 6, 5, 25) == [3, 20]
    # 解决二次同余方程，确保无解，返回空列表
    assert quadratic_congruence(120, 80, 175, 500) == []
    # 解决二次同余方程，确保解为 [1]
    assert quadratic_congruence(15, 14, 7, 2) == [1]
    # 解决二次同余方程，确保解为 [10, 28]
    assert quadratic_congruence(8, 15, 7, 29) == [10, 28]
    # 解决二次同余方程，确保解为 [144, 431]
    assert quadratic_congruence(160, 200, 300, 461) == [144, 431]
    # 解决二次同余方程，确保解为 [30417843635344493501, 36001135160550533083]
    assert quadratic_congruence(100000, 123456, 7415263, 48112959837082048697) == [30417843635344493501, 36001135160550533083]
    # 解决二次同余方程，确保解为 [249, 252]
    assert quadratic_congruence(65, 121, 72, 277) == [249, 252]
    # 解决二次同余方程，确保解为 [0]
    assert quadratic_congruence(5, 10, 14, 2) == [0]
    # 解决二次同余方程，确保解为 [1]
    assert quadratic_congruence(10, 17, 19, 2) == [1]
    # 解决二次同余方程，确保解为 [0, 1]
    assert quadratic_congruence(10, 14, 20, 2) == [0, 1]
    # 解决多项式同余方程，确保解为 [220999, 242999, 463999, 485999, 706999, 728999, 949999, 971999]
    assert polynomial_congruence(6*x**5 + 10*x**4 + 5*x**3 + x**2 + x + 1,
        972000) == [220999, 242999, 463999, 485999, 706999, 728999, 949999, 971999]
    # 解决多项式同余方程，确保解为 [30287]
    assert polynomial_congruence(x**3 - 10*x**2 + 12*x - 82, 33075) == [30287]
    # 解决多项式同余方程，确保解为 [785, 1615]
    assert polynomial_congruence(x**2 + x + 47, 2401) == [785, 1615]
    # 解决多项式同余方程，确保解为 [0, 1]
    assert polynomial_congruence(10*x**
    # 断言：对 x^3 + 3 模 16 进行多项式同余检验，预期结果是 [5]
    assert polynomial_congruence(x**3 + 3, 16) == [5]
    
    # 断言：对 65*x^2 + 121*x + 72 模 277 进行多项式同余检验，预期结果是 [249, 252]
    assert polynomial_congruence(65*x**2 + 121*x + 72, 277) == [249, 252]
    
    # 断言：对 x^4 - 4 模 27 进行多项式同余检验，预期结果是 [5, 22]
    assert polynomial_congruence(x**4 - 4, 27) == [5, 22]
    
    # 断言：对 35*x^3 - 6*x^2 - 567*x + 2308 模 148225 进行多项式同余检验，预期结果是 [86957, 111157, 122531, 146731]
    assert polynomial_congruence(35*x**3 - 6*x**2 - 567*x + 2308, 148225) == [86957, 111157, 122531, 146731]
    
    # 断言：对 x^16 - 9 模 36 进行多项式同余检验，预期结果是 [3, 9, 15, 21, 27, 33]
    assert polynomial_congruence(x**16 - 9, 36) == [3, 9, 15, 21, 27, 33]
    
    # 断言：对 x^6 - 2*x^5 - 35 模 6125 进行多项式同余检验，预期结果是 [3257]
    assert polynomial_congruence(x**6 - 2*x**5 - 35, 6125) == [3257]
    
    # 预期会引发 ValueError 异常，lambda 函数检验 x^x 的多项式同余
    raises(ValueError, lambda: polynomial_congruence(x**x, 6125))
    
    # 预期会引发 ValueError 异常，lambda 函数检验 x^i 的多项式同余
    raises(ValueError, lambda: polynomial_congruence(x**i, 6125))
    
    # 预期会引发 ValueError 异常，lambda 函数检验 0.1*x^2 + 6 的多项式同余
    raises(ValueError, lambda: polynomial_congruence(0.1*x**2 + 6, 100))
    
    # 断言：对 (-1 choose 1) 模 10 进行二项式系数取模计算，预期结果是 0
    assert binomial_mod(-1, 1, 10) == 0
    
    # 断言：对 (1 choose -1) 模 10 进行二项式系数取模计算，预期结果是 0
    assert binomial_mod(1, -1, 10) == 0
    
    # 预期会引发 ValueError 异常，二项式系数取模中 n < k
    raises(ValueError, lambda: binomial_mod(2, 1, -1))
    
    # 断言：对 51 choose 10 模 10 进行二项式系数取模计算，预期结果是 0
    assert binomial_mod(51, 10, 10) == 0
    
    # 断言：对 1000 choose 500 模 3^6 进行二项式系数取模计算，预期结果是 567
    assert binomial_mod(10**3, 500, 3**6) == 567
    
    # 断言：对 (10^18 - 1) choose 123456789 模 4 进行二项式系数取模计算，预期结果是 0
    assert binomial_mod(10**18 - 1, 123456789, 4) == 0
    
    # 断言：对 10^18 choose 10^12 模 (10^5 + 3)^2 进行二项式系数取模计算，预期结果是 3744312326
    assert binomial_mod(10**18, 10**12, (10**5 + 3)**2) == 3744312326
# 定义测试函数，计算并验证二项式系数的特定素数幂模。
def test_binomial_p_pow():
    # 初始化变量：总数、二项式系数列表、初始二项式系数
    n, binomials, binomial = 1000, [1], 1
    # 计算所有的二项式系数并存储在列表中
    for i in range(1, n + 1):
        binomial *= n - i + 1  # 计算二项式的分子部分
        binomial //= i  # 计算二项式的分母部分
        binomials.append(binomial)  # 将计算得到的二项式系数添加到列表中

    # 测试对于二的幂次方的情况，算法会稍微处理得不同
    trials_2 = 100
    for _ in range(trials_2):
        m, power = randint(0, n), randint(1, 20)
        # 断言验证特定的二项式系数模二的幂次方的结果是否正确
        assert _binomial_mod_prime_power(n, m, 2, power) == binomials[m] % 2**power

    # 测试对于其他素数幂的情况
    primes = list(sieve.primerange(2*n))  # 获取所有小于2*n的素数列表
    trials = 1000
    for _ in range(trials):
        m, prime, power = randint(0, n), choice(primes), randint(1, 10)
        # 断言验证特定的二项式系数模素数的幂次方的结果是否正确
        assert _binomial_mod_prime_power(n, m, prime, power) == binomials[m] % prime**power


# 测试已弃用的数论符号函数
def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    # 测试 mobius 函数在给出警告后是否返回预期的结果
    with warns_deprecated_sympy():
        assert mobius(3) == -1
    # 测试 legendre_symbol 函数在给出警告后是否返回预期的结果
    with warns_deprecated_sympy():
        assert legendre_symbol(2, 3) == -1
    # 测试 jacobi_symbol 函数在给出警告后是否返回预期的结果
    with warns_deprecated_sympy():
        assert jacobi_symbol(2, 3) == -1
```