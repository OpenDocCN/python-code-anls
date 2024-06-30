# `D:\src\scipysrc\sympy\sympy\polys\tests\test_subresultants_qq_zz.py`

```
# 导入必要的符号变量
from sympy.core.symbol import var
# 从 sympy.polys.polytools 模块导入多项式的商、余式、斯图姆序列和子结果集
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
# 从 sympy.matrices 模块导入矩阵类
from sympy.matrices import Matrix
# 从 sympy.polys.subresultants_qq_zz 模块导入各种子结果集函数
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
    subresultants_sylv, modified_subresultants_sylv,
    subresultants_bezout, modified_subresultants_bezout,
    backward_eye,
    sturm_pg, sturm_q, sturm_amv, euclid_pg, euclid_q,
    euclid_amv, modified_subresultants_pg, subresultants_pg,
    subresultants_amv_q, quo_z, rem_z, subresultants_amv,
    modified_subresultants_amv, subresultants_rem,
    subresultants_vv, subresultants_vv_2)

# 定义测试函数 test_sylvester
def test_sylvester():
    # 定义符号变量 x
    x = var('x')

    # 第一个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(x**3 -7, 0, x) == sylvester(x**3 -7, 0, x, 1) == Matrix([[0]])
    # 第二个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(0, x**3 -7, x) == sylvester(0, x**3 -7, x, 1) == Matrix([[0]])
    # 第三个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(x**3 -7, 0, x, 2) == Matrix([[0]])
    # 第四个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(0, x**3 -7, x, 2) == Matrix([[0]])

    # 第五个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(x**3 -7, 7, x).det() == sylvester(x**3 -7, 7, x, 1).det() == 343
    # 第六个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(7, x**3 -7, x).det() == sylvester(7, x**3 -7, x, 1).det() == 343
    # 第七个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(x**3 -7, 7, x, 2).det() == -343
    # 第八个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(7, x**3 -7, x, 2).det() == 343

    # 第九个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(3, 7, x).det() == sylvester(3, 7, x, 1).det() == sylvester(3, 7, x, 2).det() == 1

    # 第十个断言：比较 sylvester 函数的输出行列式值和预期的值
    assert sylvester(3, 0, x).det() == sylvester(3, 0, x, 1).det() == sylvester(3, 0, x, 2).det() == 1

    # 第十一个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(x - 3, x - 8, x) == sylvester(x - 3, x - 8, x, 1) == sylvester(x - 3, x - 8, x, 2) == Matrix([[1, -3], [1, -8]])

    # 第十二个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x) == sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x, 1) == Matrix([[1, 0, -7,  7,  0], [0, 1,  0, -7,  7], [3, 0, -7,  0,  0], [0, 3,  0, -7,  0], [0, 0,  3,  0, -7]])

    # 第十三个断言：比较 sylvester 函数的输出和预期的矩阵
    assert sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x, 2) == Matrix([
[1, 0, -7,  7,  0,  0], [0, 3,  0, -7,  0,  0], [0, 1,  0, -7,  7,  0], [0, 0,  3,  0, -7,  0], [0, 0,  1,  0, -7,  7], [0, 0,  0,  3,  0, -7]])

# 定义测试函数 test_subresultants_sylv
def test_subresultants_sylv():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 第一个断言：比较 subresultants_sylv 函数的输出和 subresultants 函数的输出
    assert subresultants_sylv(p, q, x) == subresultants(p, q, x)
    # 第二个断言：比较 subresultants_sylv 函数的输出的最后一个元素和 res 函数的输出
    assert subresultants_sylv(p, q, x)[-1] == res(p, q, x)
    # 第三个断言：比较 subresultants_sylv 函数的输出和 euclid_amv 函数的输出
    assert subresultants_sylv(p, q, x) != euclid_amv(p, q, x)
    # 准备 AMV 因子列表
    amv_factors = [1, 1, -1, 1, -1, 1]
    # 第四个断言：比较 subresultants_sylv 函数的输出和 modified_subresultants_amv 函数的输出
    assert subresultants_sylv(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 重新定义 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 第五个断言：比较 subresultants_sylv 函数的输出和 euclid_amv 函数的输出
    assert subresultants_sylv(p, q, x) == euclid_amv(p, q, x)

# 定义测试函数 test_modified_subresultants_sylv
def test_modified_subresultants_sylv():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 准备 AMV 因子列表
    amv_factors = [1, 1, -1, 1, -1,
    # 断言检查修改后的 Sylvester 子式结果与 Sturm 序列方法计算的结果不相等
    assert modified_subresultants_sylv(p, q, x) != sturm_amv(p, q, x)

    # 给定多项式 p 和 q 的赋值
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    
    # 断言检查修改后的 Sylvester 子式结果与 Sturm 序列方法计算的结果相等
    assert modified_subresultants_sylv(p, q, x) == sturm_amv(p, q, x)
    
    # 断言检查修改后的 Sylvester 子式结果与负多项式 p 和 q 的 Sturm 序列方法计算的结果不相等
    assert modified_subresultants_sylv(-p, q, x) != sturm_amv(-p, q, x)
def test_res():
    # 定义符号变量 x
    x = var('x')

    # 断言调用 res 函数返回值为 1
    assert res(3, 5, x) == 1

def test_res_q():
    # 定义符号变量 x
    x = var('x')

    # 断言调用 res_q 函数返回值为 1
    assert res_q(3, 5, x) == 1

def test_res_z():
    # 定义符号变量 x
    x = var('x')

    # 断言调用 res_z 函数返回值为 1
    assert res_z(3, 5, x) == 1

    # 断言 res 函数、res_q 函数、res_z 函数返回相同的结果
    assert res(3, 5, x) == res_q(3, 5, x) == res_z(3, 5, x)

def test_bezout():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = -2*x**5 + 7*x**3 + 9*x**2 - 3*x + 1
    q = -10*x**4 + 21*x**2 + 18*x - 3

    # 断言使用 'bz' 算法计算的 bezout 结果的行列式等于 sylvester 行列式
    assert bezout(p, q, x, 'bz').det() == sylvester(p, q, x, 2).det()

    # 断言使用 'bz' 算法计算的 bezout 结果的行列式不等于 sylvester 行列式（次数为 1 的情况）
    assert bezout(p, q, x, 'bz').det() != sylvester(p, q, x, 1).det()

    # 断言使用 'prs' 算法计算的 bezout 结果与 backward_eye(5) * bezout(p, q, x, 'bz') * backward_eye(5) 相等
    assert bezout(p, q, x, 'prs') == backward_eye(5) * bezout(p, q, x, 'bz') * backward_eye(5)

def test_subresultants_bezout():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言 subresultants_bezout(p, q, x) 等于 subresultants(p, q, x)
    assert subresultants_bezout(p, q, x) == subresultants(p, q, x)

    # 断言 subresultants_bezout(p, q, x) 最后一个元素等于 sylvester 行列式
    assert subresultants_bezout(p, q, x)[-1] == sylvester(p, q, x).det()

    # 断言 subresultants_bezout(p, q, x) 不等于 euclid_amv(p, q, x)
    assert subresultants_bezout(p, q, x) != euclid_amv(p, q, x)

    # 定义 amv_factors 列表
    amv_factors = [1, 1, -1, 1, -1, 1]

    # 断言 subresultants_bezout(p, q, x) 等于 amv_factors 和 modified_subresultants_amv(p, q, x) 结果的乘积
    assert subresultants_bezout(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 定义另一组多项式 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言 subresultants_bezout(p, q, x) 等于 euclid_amv(p, q, x)
    assert subresultants_bezout(p, q, x) == euclid_amv(p, q, x)

def test_modified_subresultants_bezout():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 定义 amv_factors 列表
    amv_factors = [1, 1, -1, 1, -1, 1]

    # 断言 modified_subresultants_bezout(p, q, x) 等于 amv_factors 和 subresultants_amv(p, q, x) 结果的乘积
    assert modified_subresultants_bezout(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]

    # 断言 modified_subresultants_bezout(p, q, x) 最后一个元素不等于 sylvester(p + x**8, q, x).det()
    assert modified_subresultants_bezout(p, q, x)[-1] != sylvester(p + x**8, q, x).det()

    # 断言 modified_subresultants_bezout(p, q, x) 不等于 sturm_amv(p, q, x)
    assert modified_subresultants_bezout(p, q, x) != sturm_amv(p, q, x)

    # 定义另一组多项式 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言 modified_subresultants_bezout(p, q, x) 等于 sturm_amv(p, q, x)
    assert modified_subresultants_bezout(p, q, x) == sturm_amv(p, q, x)

    # 断言 modified_subresultants_bezout(-p, q, x) 不等于 sturm_amv(-p, q, x)
    assert modified_subresultants_bezout(-p, q, x) != sturm_amv(-p, q, x)

def test_sturm_pg():
    # 定义符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言 sturm_pg(p, q, x) 最后一个元素不等于 sylvester(p, q, x, 2).det()
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()

    # 定义 sam_factors 列表
    sam_factors = [1, 1, -1, -1, 1, 1]

    # 断言 sturm_pg(p, q, x) 等于 sam_factors 和 euclid_pg(p, q, x) 结果的乘积
    assert sturm_pg(p, q, x) == [i*j for i,j in zip(sam_factors, euclid_pg(p, q, x))]

    # 定义另一组多项式 p 和 q
    p = -9*x**5 - 5*x**3 - 9
    q = -45*x**4 - 15*x**2

    # 断言 sturm_pg(p, q, x, 1) 最后一个元素等于 sylvester(p, q, x, 1).det()
    assert sturm_pg(p, q, x, 1)[-1] == sylvester(p, q, x, 1).det()

    # 断言 sturm_pg(p, q, x) 最后一个元素不等于 sylvester(p, q, x, 2).det()
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()

    # 断言 sturm_pg(-p, q, x) 最后一个元素等于 sylvester(-p, q, x, 2).det()
    assert sturm_pg(-p, q, x)[-1] == s
    # 斯托姆序列验证函数应用，验证给定多项式 p, q, x 生成的斯托姆序列与给定函数返回的列表是否相等
    assert sturm_amv(p, q, x) == [i*j for i,j in zip(sam_factors, euclid_amv(p, q, x))]
    
    # 设定多项式 p 和 q 的值
    p = -9*x**5 - 5*x**3 - 9
    q = -45*x**4 - 15*x**2
    
    # 验证使用斯托姆序列方法在偏移值为 1 的情况下，最后一个元素是否等于使用西尔维斯特矩阵计算的行列式值
    assert sturm_amv(p, q, x, 1)[-1] == sylvester(p, q, x, 1).det()
    
    # 验证使用斯托姆序列方法在默认偏移值下，最后一个元素是否不等于使用西尔维斯特矩阵计算的二阶行列式值
    assert sturm_amv(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    
    # 验证使用相反的多项式 -p 和 q，最后一个元素是否等于使用相反的多项式的西尔维斯特矩阵计算的二阶行列式值
    assert sturm_amv(-p, q, x)[-1] == sylvester(-p, q, x, 2).det()
    
    # 验证使用斯托姆序列的另一种变体函数和使用修改的子结果多项式生成函数是否相等
    assert sturm_pg(-p, q, x) == modified_subresultants_pg(-p, q, x)
def test_euclid_pg():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**6 + x**5 - x**4 - x**3 + x**2 - x + 1
    q = 6*x**5 + 5*x**4 - 4*x**3 - 3*x**2 + 2*x - 1

    # 断言欧几里得算法得到的最后一个元素等于 Sylvester 矩阵的行列式
    assert euclid_pg(p, q, x)[-1] == sylvester(p, q, x).det()

    # 断言欧几里得算法结果等于子结果子式算法的结果
    assert euclid_pg(p, q, x) == subresultants_pg(p, q, x)

    # 重新赋值 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言欧几里得算法得到的最后一个元素不等于 Sylvester 矩阵的二次形式行列式
    assert euclid_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()

    # 定义相同符号因子
    sam_factors = [1, 1, -1, -1, 1, 1]

    # 断言欧几里得算法结果等于 Sturm 算法结果和相同符号因子的乘积
    assert euclid_pg(p, q, x) == [i*j for i,j in zip(sam_factors, sturm_pg(p, q, x))]


def test_euclid_q():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言欧几里得算法 Q 形式得到的最后一个元素等于 Sturm 算法的最后一个元素的相反数
    assert euclid_q(p, q, x)[-1] == -sturm(p)[-1]


def test_euclid_amv():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言增广余子式算法得到的最后一个元素等于 Sylvester 矩阵的行列式
    assert euclid_amv(p, q, x)[-1] == sylvester(p, q, x).det()

    # 断言增广余子式算法结果等于增广余子式算法的结果
    assert euclid_amv(p, q, x) == subresultants_amv(p, q, x)

    # 重新赋值 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言增广余子式算法得到的最后一个元素不等于 Sylvester 矩阵的二次形式行列式
    assert euclid_amv(p, q, x)[-1] != sylvester(p, q, x, 2).det()

    # 定义相同符号因子
    sam_factors = [1, 1, -1, -1, 1, 1]

    # 断言增广余子式算法结果等于 Sturm 算法结果和相同符号因子的乘积
    assert euclid_amv(p, q, x) == [i*j for i,j in zip(sam_factors, sturm_amv(p, q, x))]


def test_modified_subresultants_pg():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 定义增广余子式算法的符号因子
    amv_factors = [1, 1, -1, 1, -1, 1]

    # 断言修改后的子结果子式算法结果等于增广余子式算法结果和符号因子的乘积
    assert modified_subresultants_pg(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_pg(p, q, x))]

    # 断言修改后的子结果子式算法得到的最后一个元素不等于 Sylvester 矩阵增广后的行列式
    assert modified_subresultants_pg(p, q, x)[-1] != sylvester(p + x**8, q, x).det()

    # 断言修改后的子结果子式算法结果不等于 Sturm 算法结果
    assert modified_subresultants_pg(p, q, x) != sturm_pg(p, q, x)

    # 重新赋值 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言修改后的子结果子式算法结果等于 Sturm 算法结果
    assert modified_subresultants_pg(p, q, x) == sturm_pg(p, q, x)

    # 断言修改后的子结果子式算法结果不等于 Sturm 算法对 -p, q 的结果
    assert modified_subresultants_pg(-p, q, x) != sturm_pg(-p, q, x)


def test_subresultants_pg():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言子结果子式算法结果等于子结果算法结果
    assert subresultants_pg(p, q, x) == subresultants(p, q, x)

    # 断言子结果子式算法结果的最后一个元素等于 Sylvester 矩阵的行列式
    assert subresultants_pg(p, q, x)[-1] == sylvester(p, q, x).det()

    # 断言子结果子式算法结果不等于欧几里得算法结果
    assert subresultants_pg(p, q, x) != euclid_pg(p, q, x)

    # 定义增广余子式算法的符号因子
    amv_factors = [1, 1, -1, 1, -1, 1]

    # 断言子结果子式算法结果等于符号因子和修改后的增广余子式算法结果的乘积
    assert subresultants_pg(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 重新赋值 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7

    # 断言子结果子式算法结果等于欧几里得算法结果
    assert subresultants_pg(p, q, x) == euclid_pg(p, q, x)


def test_subresultants_amv_q():
    # 定义变量 x
    x = var('x')

    # 初始化多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言增广余子式算法 Q 形式结果等于子结果算法结果
    assert subresultants_amv_q(p, q,
    # 断言：验证 subresultants_amv 函数返回的结果与 euclid_amv 函数返回的结果是否相等
    assert subresultants_amv(p, q, x) == euclid_amv(p, q, x)
def test_rem_z():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 断言 rem_z(p, -q, x) 不等于 prem(p, -q, x)
    assert rem_z(p, -q, x) != prem(p, -q, x)

def test_quo_z():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 断言 quo_z(p, -q, x) 不等于 pquo(p, -q, x)
    assert quo_z(p, -q, x) != pquo(p, -q, x)

    # 创建符号变量 y
    y = var('y')
    q = 3*x**6 + 5*y**4 - 4*x**2 - 9*x + 21
    # 断言 quo_z(p, -q, x) 等于 pquo(p, -q, x)
    assert quo_z(p, -q, x) == pquo(p, -q, x)

def test_subresultants_amv():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 断言 subresultants_amv(p, q, x) 等于 subresultants(p, q, x)
    assert subresultants_amv(p, q, x) == subresultants(p, q, x)
    # 断言 subresultants_amv(p, q, x)[-1] 等于 sylvester(p, q, x).det()
    assert subresultants_amv(p, q, x)[-1] == sylvester(p, q, x).det()
    # 断言 subresultants_amv(p, q, x) 不等于 euclid_amv(p, q, x)
    assert subresultants_amv(p, q, x) != euclid_amv(p, q, x)
    # 设置 AMV 系数因子
    amv_factors = [1, 1, -1, 1, -1, 1]
    # 断言 subresultants_amv(p, q, x) 等于 [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]
    assert subresultants_amv(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 重新定义 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    # 断言 subresultants_amv(p, q, x) 等于 euclid_amv(p, q, x)
    assert subresultants_amv(p, q, x) == euclid_amv(p, q, x)


def test_modified_subresultants_amv():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 设置 AMV 系数因子
    amv_factors = [1, 1, -1, 1, -1, 1]
    # 断言 modified_subresultants_amv(p, q, x) 等于 [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    assert modified_subresultants_amv(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    # 断言 modified_subresultants_amv(p, q, x)[-1] 不等于 sylvester(p + x**8, q, x).det()
    assert modified_subresultants_amv(p, q, x)[-1] != sylvester(p + x**8, q, x).det()
    # 断言 modified_subresultants_amv(p, q, x) 不等于 sturm_amv(p, q, x)
    assert modified_subresultants_amv(p, q, x) != sturm_amv(p, q, x)

    # 重新定义 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    # 断言 modified_subresultants_amv(p, q, x) 等于 sturm_amv(p, q, x)
    assert modified_subresultants_amv(p, q, x) == sturm_amv(p, q, x)
    # 断言 modified_subresultants_amv(-p, q, x) 不等于 sturm_amv(-p, q, x)
    assert modified_subresultants_amv(-p, q, x) != sturm_amv(-p, q, x)


def test_subresultants_rem():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 断言 subresultants_rem(p, q, x) 等于 subresultants(p, q, x)
    assert subresultants_rem(p, q, x) == subresultants(p, q, x)
    # 断言 subresultants_rem(p, q, x)[-1] 等于 sylvester(p, q, x).det()
    assert subresultants_rem(p, q, x)[-1] == sylvester(p, q, x).det()
    # 断言 subresultants_rem(p, q, x) 不等于 euclid_amv(p, q, x)
    assert subresultants_rem(p, q, x) != euclid_amv(p, q, x)
    # 设置 AMV 系数因子
    amv_factors = [1, 1, -1, 1, -1, 1]
    # 断言 subresultants_rem(p, q, x) 等于 [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]
    assert subresultants_rem(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 重新定义 p 和 q
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    # 断言 subresultants_rem(p, q, x) 等于 euclid_amv(p, q, x)
    assert subresultants_rem(p, q, x) == euclid_amv(p, q, x)


def test_subresultants_vv():
    # 创建符号变量 x
    x = var('x')

    # 定义多项式 p 和 q
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 断言 subresultants_vv(p, q, x) 等于 subresultants(p, q, x)
    assert subresultants_vv(p, q, x) == subresultants(p, q, x)
    # 断言 subresultants_vv(p, q, x)[-1] 等于 sylvester(p, q, x).det()
    assert subresultants_vv(p, q, x)[-1] == sylvester(p, q, x).det()
    # 断言 subresultants_vv(p, q, x) 不等于 euclid_amv(p,
    # 计算多项式 p(x) = x^8 + x^6 - 3*x^4 - 3*x^3 + 8*x^2 + 2*x - 5
    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    # 计算多项式 q(x) = 3*x^6 + 5*x^4 - 4*x^2 - 9*x + 21
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    # 使用函数 subresultants_vv_2 计算 p(x) 和 q(x) 的副结果
    assert subresultants_vv_2(p, q, x) == subresultants(p, q, x)
    # 断言最后一个副结果与 Sylvester 矩阵的行列式相等
    assert subresultants_vv_2(p, q, x)[-1] == sylvester(p, q, x).det()
    # 断言 subresultants_vv_2(p, q, x) 和 euclid_amv(p, q, x) 不相等
    assert subresultants_vv_2(p, q, x) != euclid_amv(p, q, x)
    # 定义 AMV 因子列表
    amv_factors = [1, 1, -1, 1, -1, 1]
    # 断言 subresultants_vv_2(p, q, x) 与修改后的 AMV 副结果列表的点积相等
    assert subresultants_vv_2(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    # 计算另一组多项式
    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    # 断言 subresultants_vv_2(p, q, x) 和 euclid_amv(p, q, x) 相等
    assert subresultants_vv_2(p, q, x) == euclid_amv(p, q, x)
```