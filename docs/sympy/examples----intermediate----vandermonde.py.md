# `D:\src\scipysrc\sympy\examples\intermediate\vandermonde.py`

```
#!/usr/bin/env python

"""Vandermonde matrix example

Demonstrates matrix computations using the Vandermonde matrix.
  * https://en.wikipedia.org/wiki/Vandermonde_matrix
"""

from sympy import Matrix, pprint, Rational, symbols, Symbol, zeros


def symbol_gen(sym_str):
    """Symbol generator

    Generates sym_str_n where n is the number of times the generator
    has been called.
    """
    n = 0
    while True:
        yield Symbol("%s_%d" % (sym_str, n))
        n += 1


def comb_w_rep(n, k):
    """Combinations with repetition

    Returns the list of k combinations with repetition from n objects.
    """
    if k == 0:
        return [[]]
    combs = [[i] for i in range(n)]
    for i in range(k - 1):
        curr = []
        for p in combs:
            for m in range(p[-1], n):
                curr.append(p + [m])
        combs = curr
    return combs


def vandermonde(order, dim=1, syms='a b c d'):
    """Computes a Vandermonde matrix of given order and dimension.

    Define syms to give beginning strings for temporary variables.

    Returns the Matrix, the temporary variables, and the terms for the
    polynomials.
    """
    syms = syms.split()  # 将字符串 syms 拆分成符号列表
    n = len(syms)  # 获取符号的数量
    if n < dim:  # 如果符号数量小于维度
        new_syms = []
        for i in range(dim - n):
            j, rem = divmod(i, n)
            new_syms.append(syms[rem] + str(j))  # 生成新的符号
        syms.extend(new_syms)  # 扩展符号列表以包含新生成的符号
    terms = []
    for i in range(order + 1):
        terms.extend(comb_w_rep(dim, i))  # 获取组合列表
    rank = len(terms)  # 获取组合的数量
    V = zeros(rank)  # 创建一个零矩阵
    generators = [symbol_gen(syms[i]) for i in range(dim)]  # 生成符号生成器列表
    all_syms = []
    for i in range(rank):
        row_syms = [next(g) for g in generators]  # 获取下一行的符号
        all_syms.append(row_syms)  # 将符号添加到 all_syms 列表中
        for j, term in enumerate(terms):
            v_entry = 1
            for k in term:
                v_entry *= row_syms[k]  # 计算 Vandermonde 矩阵的每个元素
            V[i*rank + j] = v_entry  # 将计算结果放入 Vandermonde 矩阵中
    return V, all_syms, terms  # 返回 Vandermonde 矩阵、所有生成的符号和组合列表


def gen_poly(points, order, syms):
    """Generates a polynomial using a Vandermonde system"""
    num_pts = len(points)  # 获取点的数量
    if num_pts == 0:  # 如果点的数量为零，抛出异常
        raise ValueError("Must provide points")
    dim = len(points[0]) - 1  # 获取维度
    if dim > len(syms):  # 如果维度大于符号的数量，抛出异常
        raise ValueError("Must provide at least %d symbols for the polynomial" % dim)
    V, tmp_syms, terms = vandermonde(order, dim)  # 计算 Vandermonde 矩阵
    if num_pts < V.shape[0]:  # 如果点的数量小于 Vandermonde 矩阵的行数，抛出异常
        raise ValueError(
            "Must provide %d points for order %d, dimension "
            "%d polynomial, given %d points" %
            (V.shape[0], order, dim, num_pts))
    elif num_pts > V.shape[0]:  # 如果点的数量大于 Vandermonde 矩阵的行数，输出警告
        print("gen_poly given %d points but only requires %d, "\
            "continuing using the first %d points" % \
            (num_pts, V.shape[0], V.shape[0]))
        num_pts = V.shape[0]  # 更新点的数量为 Vandermonde 矩阵的行数

    subs_dict = {}
    for j in range(dim):
        for i in range(num_pts):
            subs_dict[tmp_syms[i][j]] = points[i][j]  # 构建符号替换字典
    V_pts = V.subs(subs_dict)  # 使用符号替换字典替换 Vandermonde 矩阵中的符号
    V_inv = V_pts.inv()  # 求 Vandermonde 矩阵的逆矩阵

    coeffs = V_inv.multiply(Matrix([points[i][-1] for i in range(num_pts)]))  # 计算多项式系数

    f = 0  # 初始化多项式
    # 对于 terms 列表中的每个项，使用 enumerate 获取索引 j 和项 term
    for j, term in enumerate(terms):
        # 初始化 t 为 1，用于存储每个 term 的乘积结果
        t = 1
        # 遍历 term 中的每个元素 k
        for k in term:
            # 将 syms 字典中 k 对应的值乘到 t 上
            t *= syms[k]
        # 将 coeffs[j] 乘以 t 的结果加到 f 上，更新多项式的值
        f += coeffs[j]*t
    # 返回计算后的多项式 f
    return f
# 定义程序的主函数
def main():
    # 设置 Vandermonde 矩阵的阶数为 2
    order = 2
    # 调用 vandermonde 函数生成 Vandermonde 矩阵 V，同时获取临时符号列表 tmp_syms 和空列表
    V, tmp_syms, _ = vandermonde(order)
    # 打印 Vandermonde 矩阵的信息
    print("Vandermonde matrix of order 2 in 1 dimension")
    # 使用 pprint 函数打印 Vandermonde 矩阵 V
    pprint(V)

    # 打印分隔线
    print('-'*79)
    # 打印计算行列式并与 \sum_{0<i<j<=3}(a_j - a_i) 进行比较的信息
    print(r"Computing the determinant and comparing to \sum_{0<i<j<=3}(a_j - a_i)")

    # 初始化行列式和
    det_sum = 1
    # 计算行列式和
    for j in range(order + 1):
        for i in range(j):
            # 计算行列式和乘积 (a_j - a_i)
            det_sum *= (tmp_syms[j][0] - tmp_syms[i][0])

    # 打印行列式的值 det(V)，行列式和的值 det_sum，以及展开后的行列式和值
    print(r"""
    det(V) = {det}
    \sum   = {sum}
           = {sum_expand}
    """.format(det=V.det(),
               sum=det_sum,
               sum_expand=det_sum.expand(),
              ))

    # 打印分隔线
    print('-'*79)
    # 打印使用 Vandermonde 矩阵进行多项式拟合的信息
    print("Polynomial fitting with a Vandermonde Matrix:")
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')

    # 定义多项式拟合的数据点列表 points
    points = [(0, 3), (1, 2), (2, 3)]
    # 打印拟合二次函数的信息，包括数据点和生成的多项式
    print("""
    Quadratic function, represented by 3 points:
       points = {pts}
       f = {f}
    """.format(pts=points,
               f=gen_poly(points, 2, [x]),
              ))

    # 更新数据点列表 points，拟合二维二次函数
    points = [(0, 1, 1), (1, 0, 0), (1, 1, 0), (Rational(1, 2), 0, 0),
              (0, Rational(1, 2), 0), (Rational(1, 2), Rational(1, 2), 0)]
    # 打印拟合二维二次函数的信息，包括数据点和生成的多项式
    print("""
    2D Quadratic function, represented by 6 points:
       points = {pts}
       f = {f}
    """.format(pts=points,
               f=gen_poly(points, 2, [x, y]),
              ))

    # 更新数据点列表 points，拟合三维线性函数
    points = [(0, 1, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)]
    # 打印拟合三维线性函数的信息，包括数据点和生成的多项式
    print("""
    3D linear function, represented by 4 points:
       points = {pts}
       f = {f}
    """.format(pts=points,
               f=gen_poly(points, 1, [x, y, z]),
              ))


# 如果该脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```