# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_basis.py`

```
# 从 sympy.abc 模块中导入变量 x
# 从 sympy.core 模块中导入 S
# 从 sympy.core.numbers 模块中导入 AlgebraicNumber
# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt
# 从 sympy.polys 模块中导入 Poly 和 cyclotomic_poly
# 从 sympy.polys.domains 模块中导入 QQ
# 从 sympy.polys.matrices 模块中导入 DomainMatrix 和 DM
# 从 sympy.polys.numberfields.basis 模块中导入 round_two
# 从 sympy.testing.pytest 模块中导入 raises

# 定义一个测试函数，用于测试 round_two 函数
def test_round_two():
    # 测试在多个情况下调用 round_two 函数会引发 ValueError 异常
    # 首先测试 Poly(x ** 2 - 1) 函数调用
    raises(ValueError, lambda: round_two(Poly(x ** 2 - 1)))
    # 接着测试 Poly(x ** 2 + sqrt(2)) 函数调用
    raises(ValueError, lambda: round_two(Poly(x ** 2 + sqrt(2))))

    # 定义一个循环，遍历 cases 中的每个元素
    for f, B_exp, d_exp in cases:
        # 使用 QQ.alg_field_from_poly(f) 创建一个代数域 K
        K = QQ.alg_field_from_poly(f)
        # 获取 K 的最大秩 order，转换为 QQ_matrix 类型并赋值给 B
        B = K.maximal_order().QQ_matrix
        # 计算 K 的判别式并赋值给 d
        d = K.discriminant()
        # 断言判别式 d 等于预期值 d_exp
        assert d == d_exp
        # 断言 B 的逆乘以 B_exp 的结果的行列式的平方等于 1
        assert (B.inv()*B_exp).det()**2 == 1


# 定义一个测试函数，用于测试 AlgebraicField_integral_basis 函数
def test_AlgebraicField_integral_basis():
    # 使用 sqrt(5) 创建一个代数数 alpha，并取别名为 'alpha'
    alpha = AlgebraicNumber(sqrt(5), alias='alpha')
    # 使用 QQ.algebraic_field(alpha) 创建一个代数域 k
    k = QQ.algebraic_field(alpha)
    # 调用 k.integral_basis() 函数并赋值给 B0
    B0 = k.integral_basis()
    # 调用 k.integral_basis(fmt='sympy') 函数并赋值给 B1
    B1 = k.integral_basis(fmt='sympy')
    # 调用 k.integral_basis(fmt='alg') 函数并赋值给 B2
    B2 = k.integral_basis(fmt='alg')
    # 断言 B0 的结果等于预期值 [k([1]), k([S.Half, S.Half])]
    assert B0 == [k([1]), k([S.Half, S.Half])]
    # 断言 B1 的结果等于预期值 [1, S.Half + alpha/2]
    assert B1 == [1, S.Half + alpha/2]
    # 断言 B2 的结果等于预期值 [k.ext.field_element([1]), k.ext.field_element([S.Half, S.Half])]
    assert B2 == [k.ext.field_element([1]),
                  k.ext.field_element([S.Half, S.Half])]
```