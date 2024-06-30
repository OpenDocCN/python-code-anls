# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_multinomial.py`

```
# 从 sympy.ntheory.multinomial 模块中导入需要使用的函数和迭代器
from sympy.ntheory.multinomial import (binomial_coefficients, binomial_coefficients_list, multinomial_coefficients)
from sympy.ntheory.multinomial import multinomial_coefficients_iterator

# 测试函数，验证 binomial_coefficients_list 函数的输出是否符合预期
def test_binomial_coefficients_list():
    # 断言函数对于输入 0 的返回结果是否为 [1]
    assert binomial_coefficients_list(0) == [1]
    # 断言函数对于输入 1 的返回结果是否为 [1, 1]
    assert binomial_coefficients_list(1) == [1, 1]
    # 断言函数对于输入 2 的返回结果是否为 [1, 2, 1]
    assert binomial_coefficients_list(2) == [1, 2, 1]
    # 断言函数对于输入 3 的返回结果是否为 [1, 3, 3, 1]
    assert binomial_coefficients_list(3) == [1, 3, 3, 1]
    # 断言函数对于输入 4 的返回结果是否为 [1, 4, 6, 4, 1]
    assert binomial_coefficients_list(4) == [1, 4, 6, 4, 1]
    # 断言函数对于输入 5 的返回结果是否为 [1, 5, 10, 10, 5, 1]
    assert binomial_coefficients_list(5) == [1, 5, 10, 10, 5, 1]
    # 断言函数对于输入 6 的返回结果是否为 [1, 6, 15, 20, 15, 6, 1]
    assert binomial_coefficients_list(6) == [1, 6, 15, 20, 15, 6, 1]

# 测试函数，验证 binomial_coefficients 函数是否正确计算并返回与 binomial_coefficients_list 函数相同的结果
def test_binomial_coefficients():
    # 对于范围内的每一个 n，依次执行以下操作
    for n in range(15):
        # 调用 binomial_coefficients 函数计算结果
        c = binomial_coefficients(n)
        # 从计算结果中提取排序后的值组成列表
        l = [c[k] for k in sorted(c)]
        # 断言提取的列表与对应 n 的 binomial_coefficients_list 函数的返回结果相同
        assert l == binomial_coefficients_list(n)

# 测试函数，验证 multinomial_coefficients 函数的输出是否符合预期
def test_multinomial_coefficients():
    # 断言函数对于输入 (1, 1) 的返回结果是否为 {(1,): 1}
    assert multinomial_coefficients(1, 1) == {(1,): 1}
    # 断言函数对于输入 (1, 2) 的返回结果是否为 {(2,): 1}
    assert multinomial_coefficients(1, 2) == {(2,): 1}
    # 断言函数对于输入 (1, 3) 的返回结果是否为 {(3,): 1}
    assert multinomial_coefficients(1, 3) == {(3,): 1}
    # 断言函数对于输入 (2, 0) 的返回结果是否为 {(0, 0): 1}
    assert multinomial_coefficients(2, 0) == {(0, 0): 1}
    # 断言函数对于输入 (2, 1) 的返回结果是否为 {(0, 1): 1, (1, 0): 1}
    assert multinomial_coefficients(2, 1) == {(0, 1): 1, (1, 0): 1}
    # 断言函数对于输入 (2, 2) 的返回结果是否为 {(2, 0): 1, (0, 2): 1, (1, 1): 2}
    assert multinomial_coefficients(2, 2) == {(2, 0): 1, (0, 2): 1, (1, 1): 2}
    # 断言函数对于输入 (2, 3) 的返回结果是否为 {(3, 0): 1, (1, 2): 3, (0, 3): 1, (2, 1): 3}
    assert multinomial_coefficients(2, 3) == {(3, 0): 1, (1, 2): 3, (0, 3): 1, (2, 1): 3}
    # 断言函数对于输入 (3, 1) 的返回结果是否为 {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1}
    assert multinomial_coefficients(3, 1) == {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1}
    # 断言函数对于输入 (3, 2) 的返回结果是否为 {(0, 1, 1): 2, (0, 0, 2): 1, (1, 1, 0): 2, (0, 2, 0): 1, (1, 0, 1): 2, (2, 0, 0): 1}
    assert multinomial_coefficients(3, 2) == {(0, 1, 1): 2, (0, 0, 2): 1, (1, 1, 0): 2, (0, 2, 0): 1, (1, 0, 1): 2, (2, 0, 0): 1}
    # 调用 multinomial_coefficients 函数，将返回结果保存到变量 mc 中
    mc = multinomial_coefficients(3, 3)
    # 断言变量 mc 是否与预期的字典相等
    assert mc == {(2, 1, 0): 3, (0, 3, 0): 1, (1, 0, 2): 3, (0, 2, 1): 3, (0, 1, 2): 3, (3, 0, 0): 1, (2, 0, 1): 3, (1, 2, 0): 3, (1, 1, 1): 6, (0, 0, 3): 1}
    # 断言转换成字典后的 multinomial_coefficients_iterator 函数的输出是否与预期相等
    assert dict(multinomial_coefficients_iterator(2, 0)) == {(0, 0): 1}
    # 断言转换成字典后的 multinomial_coefficients_iterator 函数的输出是否与预期相等
    assert dict(multinomial_coefficients_iterator(2, 1)) == {(0, 1): 1, (1, 0): 1}
    # 断言转换成字典后的 multinomial_coefficients_iterator 函数的输出是否与预期相等
    assert dict(multinomial_coefficients_iterator(2, 2)) == {(2, 0): 1, (0, 2): 1, (1, 1): 2}
    # 断言转换成字典后的 multinomial_coefficients_iterator 函数的输出是否与预期相等
    assert dict(multinomial_coefficients_iterator(3, 3)) == mc
    # 创建迭代器对象 it，验证前四个结果是否与预期相等
    it = multinomial_coefficients_iterator(7, 2)
    assert [next(it) for i in range(4)] == [((2, 0, 0, 0, 0, 0, 0), 1), ((1, 1, 0, 0, 0, 0, 0), 2),
                                            ((0, 2, 0, 0, 0, 0, 0), 1), ((1, 0, 1, 0, 0, 0, 0), 2)]
```