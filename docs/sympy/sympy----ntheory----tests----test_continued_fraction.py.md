# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_continued_fraction.py`

```
# 导入所需的模块和函数
import itertools  # 导入 itertools 模块，用于处理迭代器和循环相关的功能
from sympy.core import GoldenRatio as phi  # 导入黄金比例常数 phi
from sympy.core.numbers import (Rational, pi)  # 导入有理数和圆周率常数 pi
from sympy.core.singleton import S  # 导入 SymPy 中的单例对象 S
from sympy.functions.elementary.miscellaneous import sqrt  # 导入计算平方根的函数 sqrt
from sympy.ntheory.continued_fraction import \
    (continued_fraction_periodic as cf_p,  # 导入周期连分数计算函数 cf_p 别名为 cf_p
     continued_fraction_iterator as cf_i,  # 导入迭代器连分数计算函数 cf_i 别名为 cf_i
     continued_fraction_convergents as cf_c,  # 导入收敛数连分数计算函数 cf_c 别名为 cf_c
     continued_fraction_reduce as cf_r,  # 导入简化连分数计算函数 cf_r 别名为 cf_r
     continued_fraction as cf)  # 导入基本连分数计算函数 cf 别名为 cf
from sympy.testing.pytest import raises  # 导入用于测试时的异常抛出函数 raises


# 定义测试函数 test_continued_fraction
def test_continued_fraction():
    # 断言：比较两个周期连分数函数的输出是否相等
    assert cf_p(1, 1, 10, 0) == cf_p(1, 1, 0, 1)
    # 断言：比较两个周期连分数函数的输出是否相等
    assert cf_p(1, -1, 10, 1) == cf_p(-1, 1, 10, -1)
    # 计算 sqrt(2)
    t = sqrt(2)
    # 断言：比较连分数和常数 -1 是否相等
    assert cf((1 + t)*(1 - t)) == cf(-1)
    
    # 遍历给定的数值列表，进行连分数计算和简化后与原数值的比较
    for n in [0, 2, Rational(2, 3), sqrt(2), 3*sqrt(2), 1 + 2*sqrt(3)/5,
            (2 - 3*sqrt(5))/7, 1 + sqrt(2), (-5 + sqrt(17))/4]:
        # 断言：比较简化后的连分数结果和原数值是否相等
        assert (cf_r(cf(n)) - n).expand() == 0
        # 断言：比较简化后的连分数结果和原数值是否相等
        assert (cf_r(cf(-n)) + n).expand() == 0
    
    # 检查是否能够正确抛出 ValueError 异常：sqrt(2 + sqrt(3)) 不支持
    raises(ValueError, lambda: cf(sqrt(2 + sqrt(3))))
    # 检查是否能够正确抛出 ValueError 异常：sqrt(2) + sqrt(3) 不支持
    raises(ValueError, lambda: cf(sqrt(2) + sqrt(3)))
    # 检查是否能够正确抛出 ValueError 异常：pi 不支持
    raises(ValueError, lambda: cf(pi))
    # 检查是否能够正确抛出 ValueError 异常：0.1 不支持
    raises(ValueError, lambda: cf(.1))

    # 检查是否能够正确抛出 ValueError 异常：在周期连分数计算函数 cf_p 中，第一个参数为 0 时不支持
    raises(ValueError, lambda: cf_p(1, 0, 0))
    # 检查是否能够正确抛出 ValueError 异常：在周期连分数计算函数 cf_p 中，第三个参数为负数时不支持
    raises(ValueError, lambda: cf_p(1, 1, -1))
    
    # 断言：验证周期连分数计算函数 cf_p 的输出是否正确
    assert cf_p(4, 3, 0) == [1, 3]
    assert cf_p(0, 3, 5) == [0, 1, [2, 1, 12, 1, 2, 2]]
    assert cf_p(1, 1, 0) == [1]
    assert cf_p(3, 4, 0) == [0, 1, 3]
    assert cf_p(4, 5, 0) == [0, 1, 4]
    assert cf_p(5, 6, 0) == [0, 1, 5]
    assert cf_p(11, 13, 0) == [0, 1, 5, 2]
    assert cf_p(16, 19, 0) == [0, 1, 5, 3]
    assert cf_p(27, 32, 0) == [0, 1, 5, 2, 2]
    assert cf_p(1, 2, 5) == [[1]]
    assert cf_p(0, 1, 2) == [1, [2]]
    assert cf_p(6, 7, 49) == [1, 1, 6]
    assert cf_p(3796, 1387, 0) == [2, 1, 2, 1, 4]
    assert cf_p(3245, 10000) == [0, 3, 12, 4, 13]
    assert cf_p(1932, 2568) == [0, 1, 3, 26, 2]
    assert cf_p(6589, 2569) == [2, 1, 1, 3, 2, 1, 3, 1, 23]

    # 定义函数 take，用于获取迭代器中的前 n 个元素
    def take(iterator, n=7):
        return list(itertools.islice(iterator, n))

    # 断言：验证迭代器连分数函数 cf_i 在给定黄金比例常数 phi 下的输出是否正确
    assert take(cf_i(phi)) == [1, 1, 1, 1, 1, 1, 1]
    # 断言：验证迭代器连分数函数 cf_i 在给定圆周率常数 pi 下的输出是否正确
    assert take(cf_i(pi)) == [3, 7, 15, 1, 292, 1, 1]

    # 断言：验证收敛数连分数函数 cf_c 在给定有理数 17/12 下的输出是否正确
    assert list(cf_i(Rational(17, 12))) == [1, 2, 2, 2]
    # 断言：验证收敛数连分数函数 cf_c 在给定负有理数 -17/12 下的输出是否正确
    assert list(cf_i(Rational(-17, 12))) == [-2, 1, 1, 2, 2]

    # 断言：验证收敛数连分数函数 cf_c 在给定系列 [1, 6, 1, 8] 下的输出是否正确
    assert list(cf_c([1, 6, 1, 8])) == [S.One, Rational(7, 6), Rational(8, 7), Rational(71, 62)]
    # 断言：验证收敛数连分数函数 cf_c 在给定单一系列 [2] 下的输出是否正确
    assert list(cf_c([2])) == [S(2)]
    # 断言：验证收敛数连分数函数 cf_c 在给定连分数系列 [1, 1, 1, 1, 1, 1, 1] 下的输出是否正确
    assert list(cf_c([1, 1, 1, 1, 1, 1, 1])) == [S.One, S(2), Rational(3, 2), Rational(5, 3),
                                                 Rational(8, 5), Rational(13, 8), Rational(21, 13)]
    # 断言：验证收敛数连分数函数 cf_c 在给定连分数系列 [1, 6, -1/2, 4] 下的输出是否正确
    assert list(cf_c([1, 6, Rational(-1, 2), 4])) == [S.One, Rational(7, 6), Rational(5, 4), Rational(3
    # 创建一个生成器表达式，生成一个连分数的迭代器
    # 连分数的形式如下：2, 2/3, 2, 2/3, 2, 2/3, ...
    cf_iter_e = (2 if i == 1 else i // 3 * 2 if i % 3 == 0 else 1 for i in itertools.count(1))
    
    # 使用连分数迭代器生成连分数并验证取出的前几项是否正确
    assert take(cf_c(cf_iter_e)) == [S(2), S(3), Rational(8, 3), Rational(11, 4), Rational(19, 7),
                                     Rational(87, 32), Rational(106, 39)]
    
    # 验证给定系数列表生成的连分数是否正确计算出结果
    assert cf_r([1, 6, 1, 8]) == Rational(71, 62)
    
    # 验证当只有一个元素的系数列表生成的连分数是否正确计算出结果
    assert cf_r([3]) == S(3)
    
    # 验证负数系数列表生成的连分数是否正确计算出结果
    assert cf_r([-1, 5, 1, 4]) == Rational(-24, 29)
    
    # 验证复杂的系数列表生成的连分数是否正确计算出结果，并展开后与给定表达式相等
    assert (cf_r([0, 1, 1, 7, [24, 8]]) - (sqrt(3) + 2)/7).expand() == 0
    
    # 验证给定系数列表生成的连分数是否正确计算出结果
    assert cf_r([1, 5, 9]) == Rational(55, 46)
    
    # 验证复杂的系数列表生成的连分数是否正确计算出结果，并展开后与给定表达式相等
    assert (cf_r([[1]]) - (sqrt(5) + 1)/2).expand() == 0
    
    # 验证包含嵌套列表的系数列表生成的连分数是否正确计算出结果
    assert cf_r([-3, 1, 1, [2]]) == -1 - sqrt(2)
```