# `D:\src\scipysrc\sympy\sympy\polys\multivariate_resultants.py`

```
"""
This module contains functions for two multivariate resultants. These
are:

- Dixon's resultant.
- Macaulay's resultant.

Multivariate resultants are used to identify whether a multivariate
system has common roots. That is when the resultant is equal to zero.
"""
from math import prod  # 导入 math 模块中的 prod 函数，用于计算列表元素的乘积

from sympy.core.mul import Mul  # 导入 sympy.core.mul 模块中的 Mul 类
from sympy.matrices.dense import (Matrix, diag)  # 导入 sympy.matrices.dense 模块中的 Matrix 和 diag 函数
from sympy.polys.polytools import (Poly, degree_list, rem)  # 导入 sympy.polys.polytools 模块中的 Poly、degree_list 和 rem 函数
from sympy.simplify.simplify import simplify  # 导入 sympy.simplify.simplify 模块中的 simplify 函数
from sympy.tensor.indexed import IndexedBase  # 导入 sympy.tensor.indexed 模块中的 IndexedBase 类
from sympy.polys.monomials import itermonomials, monomial_deg  # 导入 sympy.polys.monomials 模块中的 itermonomials 和 monomial_deg 函数
from sympy.polys.orderings import monomial_key  # 导入 sympy.polys.orderings 模块中的 monomial_key 函数
from sympy.polys.polytools import poly_from_expr, total_degree  # 导入 sympy.polys.polytools 模块中的 poly_from_expr 和 total_degree 函数
from sympy.functions.combinatorial.factorials import binomial  # 导入 sympy.functions.combinatorial.factorials 模块中的 binomial 函数
from itertools import combinations_with_replacement  # 导入 itertools 模块中的 combinations_with_replacement 函数
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 sympy.utilities.exceptions 模块中的 sympy_deprecation_warning 函数

class DixonResultant():
    """
    A class for retrieving the Dixon's resultant of a multivariate
    system.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import DixonResultant
    >>> x, y = symbols('x, y')

    >>> p = x + y
    >>> q = x ** 2 + y ** 3
    >>> h = x ** 2 + y

    >>> dixon = DixonResultant(variables=[x, y], polynomials=[p, q, h])
    >>> poly = dixon.get_dixon_polynomial()
    >>> matrix = dixon.get_dixon_matrix(polynomial=poly)
    >>> matrix
    Matrix([
    [ 0,  0, -1,  0, -1],
    [ 0, -1,  0, -1,  0],
    [-1,  0,  1,  0,  0],
    [ 0, -1,  0,  0,  1],
    [-1,  0,  0,  1,  0]])
    >>> matrix.det()
    0

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Kapur1994]_
    .. [2] [Palancz08]_

    """

    def __init__(self, polynomials, variables):
        """
        A class that takes two lists, a list of polynomials and list of
        variables. Returns the Dixon matrix of the multivariate system.

        Parameters
        ----------
        polynomials : list of polynomials
            A list of m n-degree polynomials
        variables: list
            A list of all n variables
        """
        self.polynomials = polynomials  # 初始化多项式列表
        self.variables = variables  # 初始化变量列表

        self.n = len(self.variables)  # 计算变量的数量
        self.m = len(self.polynomials)  # 计算多项式的数量

        a = IndexedBase("alpha")  # 创建一个 IndexedBase 对象 'alpha'
        # 生成 n 个 alpha 变量的列表（替代变量）
        self.dummy_variables = [a[i] for i in range(self.n)]

        # 计算每个变量的最高次数的列表
        self._max_degrees = [max(degree_list(poly)[i] for poly in self.polynomials)
            for i in range(self.n)]

    @property
    def max_degrees(self):
        sympy_deprecation_warning(
            """
            The max_degrees property of DixonResultant is deprecated.
            """,
            deprecated_since_version="1.5",
            active_deprecations_target="deprecated-dixonresultant-properties",
        )
        return self._max_degrees
    def get_dixon_polynomial(self):
        r"""
        Returns
        =======

        dixon_polynomial: polynomial
            Dixon's polynomial is calculated as:

            delta = Delta(A) / ((x_1 - a_1) ... (x_n - a_n)) where,

            A =  |p_1(x_1,... x_n), ..., p_n(x_1,... x_n)|
                 |p_1(a_1,... x_n), ..., p_n(a_1,... x_n)|
                 |...             , ...,              ...|
                 |p_1(a_1,... a_n), ..., p_n(a_1,... a_n)|
        """
        # 检查是否满足方法的有效性条件
        if self.m != (self.n + 1):
            raise ValueError('Method invalid for given combination.')

        # 初始化行列表，第一行为原始多项式列表
        rows = [self.polynomials]

        # 复制变量列表到临时列表中
        temp = list(self.variables)

        # 生成每个变量的替换字典，并添加对应的多项式值到行列表中
        for idx in range(self.n):
            temp[idx] = self.dummy_variables[idx]
            substitution = dict(zip(self.variables, temp))
            rows.append([f.subs(substitution) for f in self.polynomials])

        # 构建矩阵 A
        A = Matrix(rows)

        # 计算变量与虚拟变量的差的乘积
        terms = zip(self.variables, self.dummy_variables)
        product_of_differences = Mul(*[a - b for a, b in terms])

        # 计算 Dixon 多项式并进行因式分解
        dixon_polynomial = (A.det() / product_of_differences).factor()

        # 将结果转换为多项式并返回其中的第一个多项式
        return poly_from_expr(dixon_polynomial, self.dummy_variables)[0]

    def get_upper_degree(self):
        sympy_deprecation_warning(
            """
            The get_upper_degree() method of DixonResultant is deprecated. Use
            get_max_degrees() instead.
            """,
            deprecated_since_version="1.5",
            active_deprecations_target="deprecated-dixonresultant-properties"
        )
        # 构建每个变量的幂次方列表，并计算它们的乘积
        list_of_products = [self.variables[i] ** self._max_degrees[i]
                            for i in range(self.n)]
        product = prod(list_of_products)

        # 提取乘积的所有单项式，并计算其最大次数
        product = Poly(product).monoms()
        return monomial_deg(*product)

    def get_max_degrees(self, polynomial):
        r"""
        Returns a list of the maximum degree of each variable appearing
        in the coefficients of the Dixon polynomial. The coefficients are
        viewed as polys in $x_1, x_2, \dots, x_n$.
        """
        # 对多项式的每个系数计算其变量的最大次数列表
        deg_lists = [degree_list(Poly(poly, self.variables))
                     for poly in polynomial.coeffs()]

        # 找出每个变量的最大次数
        max_degrees = [max(degs) for degs in zip(*deg_lists)]

        return max_degrees
    def get_dixon_matrix(self, polynomial):
        r"""
        Construct the Dixon matrix from the coefficients of polynomial
        \alpha. Each coefficient is viewed as a polynomial of x_1, ...,
        x_n.
        """

        # 获取多项式中各项的最高次数
        max_degrees = self.get_max_degrees(polynomial)

        # 生成 Dixon 矩阵的列标题列表
        monomials = itermonomials(self.variables, max_degrees)
        monomials = sorted(monomials, reverse=True,
                           key=monomial_key('lex', self.variables))

        # 构建 Dixon 矩阵
        dixon_matrix = Matrix([[Poly(c, *self.variables).coeff_monomial(m)
                                for m in monomials]
                                for c in polynomial.coeffs()])

        # 如果行数不等于列数，则可能需要移除部分列
        if dixon_matrix.shape[0] != dixon_matrix.shape[1]:
            # 确定需要保留的列索引列表
            keep = [column for column in range(dixon_matrix.shape[-1])
                    if any(element != 0 for element
                        in dixon_matrix[:, column])]

            dixon_matrix = dixon_matrix[:, keep]

        return dixon_matrix

    def KSY_precondition(self, matrix):
        """
        Test for the validity of the Kapur-Saxena-Yang precondition.

        The precondition requires that the column corresponding to the
        monomial 1 = x_1 ^ 0 * x_2 ^ 0 * ... * x_n ^ 0 is not a linear
        combination of the remaining ones. In SymPy notation this is
        the last column. For the precondition to hold the last non-zero
        row of the rref matrix should be of the form [0, 0, ..., 1].
        """
        # 如果输入矩阵为零矩阵，则直接返回 False
        if matrix.is_zero_matrix:
            return False

        m, n = matrix.shape

        # 简化矩阵并且保留非零行
        matrix = simplify(matrix.rref()[0])
        rows = [i for i in range(m) if any(matrix[i, j] != 0 for j in range(n))]
        matrix = matrix[rows,:]

        # 定义预期的条件矩阵（最后一行应为 [0, 0, ..., 1]）
        condition = Matrix([[0]*(n-1) + [1]])

        # 检查矩阵的最后一行是否满足预期条件
        if matrix[-1,:] == condition:
            return True
        else:
            return False

    def delete_zero_rows_and_columns(self, matrix):
        """Remove the zero rows and columns of the matrix."""
        # 找出所有非零行的索引
        rows = [
            i for i in range(matrix.rows) if not matrix.row(i).is_zero_matrix]
        # 找出所有非零列的索引
        cols = [
            j for j in range(matrix.cols) if not matrix.col(j).is_zero_matrix]

        # 返回仅包含非零行和列的子矩阵
        return matrix[rows, cols]

    def product_leading_entries(self, matrix):
        """Calculate the product of the leading entries of the matrix."""
        # 计算矩阵每行的首个非零元素的乘积
        res = 1
        for row in range(matrix.rows):
            for el in matrix.row(row):
                if el != 0:
                    res = res * el
                    break
        return res
    def get_KSY_Dixon_resultant(self, matrix):
        """Calculate the Kapur-Saxena-Yang approach to the Dixon Resultant."""
        # 删除矩阵中全零的行和列
        matrix = self.delete_zero_rows_and_columns(matrix)
        # 对矩阵进行 LU 分解，获取上三角矩阵 U
        _, U, _ = matrix.LUdecomposition()
        # 简化上三角矩阵 U 并删除其中全零的行和列
        matrix = self.delete_zero_rows_and_columns(simplify(U))
        # 返回简化后矩阵中主元素的乘积
        return self.product_leading_entries(matrix)
    """
    A class for calculating the Macaulay resultant. Note that the
    polynomials must be homogenized and their coefficients must be
    given as symbols.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import MacaulayResultant
    >>> x, y, z = symbols('x, y, z')

    >>> a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    >>> b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    >>> c_0, c_1, c_2, c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    >>> f = a_0 * y -  a_1 * x + a_2 * z
    >>> g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    >>> h = c_0 * y * z ** 2 - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + c_4 * z ** 3

    >>> mac = MacaulayResultant(polynomials=[f, g, h], variables=[x, y, z])
    >>> mac.monomial_set
    [x**4, x**3*y, x**3*z, x**2*y**2, x**2*y*z, x**2*z**2, x*y**3,
    x*y**2*z, x*y*z**2, x*z**3, y**4, y**3*z, y**2*z**2, y*z**3, z**4]
    >>> matrix = mac.get_matrix()
    >>> submatrix = mac.get_submatrix(matrix)
    >>> submatrix
    Matrix([
    [-a_1,  a_0,  a_2,    0],
    [   0, -a_1,    0,    0],
    [   0,    0, -a_1,    0],
    [   0,    0,    0, -a_1]])

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Bruce97]_
    .. [2] [Stiller96]_

    """
    def __init__(self, polynomials, variables):
        """
        Parameters
        ==========

        variables: list
            A list of all n variables
        polynomials : list of SymPy polynomials
            A list of m n-degree polynomials
        """
        self.polynomials = polynomials  # 存储输入的多项式列表
        self.variables = variables  # 存储输入的变量列表
        self.n = len(variables)  # 记录变量的数量

        # 计算每个多项式的最高次数，并存储在列表中
        self.degrees = [total_degree(poly, *self.variables) for poly
                        in self.polynomials]

        # 计算 degree_m 的值
        self.degree_m = self._get_degree_m()

        # 计算 monomials_size，即集合 T 的大小
        self.monomials_size = self.get_size()

        # 获取所有给定次数 degree_m 的单项式集合 T
        self.monomial_set = self.get_monomials_of_certain_degree(self.degree_m)

    def _get_degree_m(self):
        r"""
        Returns
        =======

        degree_m: int
            The degree_m is calculated as  1 + \sum_1 ^ n (d_i - 1),
            where d_i is the degree of the i polynomial
        """
        return 1 + sum(d - 1 for d in self.degrees)

    def get_size(self):
        r"""
        Returns
        =======

        size: int
            The size of set T. Set T is the set of all possible
            monomials of the n variables for degree equal to the
            degree_m
        """
        return binomial(self.degree_m + self.n - 1, self.n - 1)
    def get_monomials_of_certain_degree(self, degree):
        """
        Returns
        =======

        monomials: list
            A list of monomials of a certain degree.
        """
        # Generate monomials of the specified degree using combinations with replacement of variables
        monomials = [Mul(*monomial) for monomial
                     in combinations_with_replacement(self.variables,
                                                      degree)]

        return sorted(monomials, reverse=True,
                      key=monomial_key('lex', self.variables))

    def get_row_coefficients(self):
        """
        Returns
        =======

        row_coefficients: list
            The row coefficients of Macaulay's matrix
        """
        row_coefficients = []
        divisible = []
        # Iterate over each variable index up to n
        for i in range(self.n):
            if i == 0:
                # Compute the degree and obtain monomials for the first row
                degree = self.degree_m - self.degrees[i]
                monomial = self.get_monomials_of_certain_degree(degree)
                row_coefficients.append(monomial)
            else:
                # Prepare divisible terms up to the current index
                divisible.append(self.variables[i - 1] **
                                 self.degrees[i - 1])
                # Compute the degree and initial possible rows for the current index
                degree = self.degree_m - self.degrees[i]
                poss_rows = self.get_monomials_of_certain_degree(degree)
                # Iterate through divisible terms and filter possible rows
                for div in divisible:
                    for p in poss_rows:
                        if rem(p, div) == 0:
                            poss_rows = [item for item in poss_rows
                                         if item != p]
                row_coefficients.append(poss_rows)
        return row_coefficients

    def get_matrix(self):
        """
        Returns
        =======

        macaulay_matrix: Matrix
            The Macaulay numerator matrix
        """
        rows = []
        row_coefficients = self.get_row_coefficients()
        # Iterate over each variable index up to n
        for i in range(self.n):
            # Iterate through each multiplier in row coefficients
            for multiplier in row_coefficients[i]:
                coefficients = []
                # Create polynomial by multiplying current polynomial with multiplier
                poly = Poly(self.polynomials[i] * multiplier,
                            *self.variables)

                # Compute coefficients for each monomial in the monomial set
                for mono in self.monomial_set:
                    coefficients.append(poly.coeff_monomial(mono))
                rows.append(coefficients)

        # Construct the Macaulay matrix using computed rows
        macaulay_matrix = Matrix(rows)
        return macaulay_matrix
    def get_reduced_nonreduced(self):
        r"""
        Returns
        =======

        reduced: list
            A list of the reduced monomials
        non_reduced: list
            A list of the monomials that are not reduced

        Definition
        ==========

        A polynomial is said to be reduced in x_i, if its degree (the
        maximum degree of its monomials) in x_i is less than d_i. A
        polynomial that is reduced in all variables but one is said
        simply to be reduced.
        """
        # Initialize an empty list to store divisibility conditions
        divisible = []
        # Iterate over each monomial in the monomial set
        for m in self.monomial_set:
            temp = []
            # Check each variable in the polynomial
            for i, v in enumerate(self.variables):
                # Append True if the total degree of the monomial in variable v is >= d_i
                temp.append(bool(total_degree(m, v) >= self.degrees[i]))
            # Append the list of divisibility conditions for the current monomial m
            divisible.append(temp)
        
        # Identify indices of reduced and non-reduced polynomials based on divisibility conditions
        reduced = [i for i, r in enumerate(divisible)
                   if sum(r) < self.n - 1]
        non_reduced = [i for i, r in enumerate(divisible)
                       if sum(r) >= self.n - 1]

        # Return lists of indices indicating reduced and non-reduced polynomials
        return reduced, non_reduced

    def get_submatrix(self, matrix):
        r"""
        Returns
        =======

        macaulay_submatrix: Matrix
            The Macaulay denominator matrix. Columns that are non reduced are kept.
            The row which contains one of the a_{i}s is dropped. a_{i}s
            are the coefficients of x_i ^ {d_i}.
        """
        # Obtain lists of reduced and non-reduced polynomials
        reduced, non_reduced = self.get_reduced_nonreduced()

        # if reduced == [], then det(matrix) should be 1
        if reduced == []:
            # Return a diagonal matrix with 1 as the only element
            return diag([1])

        # Create reduction_set with powers of variables raised to their respective degrees
        reduction_set = [v ** self.degrees[i] for i, v
                         in enumerate(self.variables)]

        # Obtain coefficients a_i for each variable x_i
        ais = [self.polynomials[i].coeff(reduction_set[i])
               for i in range(self.n)]

        # Extract columns from 'matrix' corresponding to reduced polynomials
        reduced_matrix = matrix[:, reduced]
        
        # Initialize a list to store indices of rows to keep in the final submatrix
        keep = []
        # Iterate over each row in the reduced_matrix
        for row in range(reduced_matrix.rows):
            # Check if none of the ais are present in the current row of reduced_matrix
            check = [ai in reduced_matrix[row, :] for ai in ais]
            # If none of the ais are found in the current row, add the row index to 'keep'
            if True not in check:
                keep.append(row)

        # Return submatrix consisting of rows indicated by 'keep' and columns indicated by 'non_reduced'
        return matrix[keep, non_reduced]
```