# `D:\src\scipysrc\sympy\sympy\liealgebras\type_a.py`

```
from sympy.liealgebras.cartan_type import Standard_Cartan  # 导入Standard_Cartan类
from sympy.core.backend import eye  # 导入eye函数


class TypeA(Standard_Cartan):
    """
    This class contains the information about
    the A series of simple Lie algebras.
    ====
    """

    def __new__(cls, n):
        if n < 1:
            raise ValueError("n cannot be less than 1")  # 如果n小于1，引发数值错误异常
        return Standard_Cartan.__new__(cls, "A", n)  # 调用父类Standard_Cartan的构造方法


    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A4")
        >>> c.dimension()
        5
        """
        return self.n+1  # 返回向量空间维数


    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a 1 iin the ith position and a -1
        in the jth position.

        """
        n = self.n  # 获取参数n
        root = [0]*(n+1)  # 创建长度为n+1的零列表
        root[i] = 1  # 在第i个位置置1
        root[j] = -1  # 在第j个位置置-1
        return root  # 返回生成的根


    def simple_root(self, i):
        """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In A_n the ith simple root is the root which has a 1
        in the ith position, a -1 in the (i+1)th position,
        and zeroes elsewhere.

        This method returns the ith simple root for the A series.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A4")
        >>> c.simple_root(1)
        [1, -1, 0, 0, 0]

        """
        return self.basic_root(i-1, i)  # 调用basic_root方法返回第i个简单根


    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of A_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """

        n = self.n  # 获取参数n
        posroots = {}  # 创建空字典用于存储正根
        k = 0  # 初始化计数器k
        for i in range(0, n):  # 遍历i从0到n-1
            for j in range(i+1, n+1):  # 遍历j从i+1到n
               k += 1  # 计数器加1
               posroots[k] = self.basic_root(i, j)  # 将生成的基本根添加到正根字典中
        return posroots  # 返回正根字典


    def highest_root(self):
        """
        Returns the highest weight root for A_n
        """

        return self.basic_root(0, self.n)  # 返回最高权重根


    def roots(self):
        """
        Returns the total number of roots for A_n
        """
        n = self.n  # 获取参数n
        return n*(n+1)  # 返回总根数
    def cartan_matrix(self):
        """
        返回 A_n 的 Cartan 矩阵。
        Lie 代数的 Cartan 矩阵是通过对简单根 (alpha[1], ...., alpha[l]) 进行排序生成的。
        然后 Cartan 矩阵的第 (i, j) 项是 (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])

        """

        n = self.n  # 获取实例变量 self.n 的值
        m = 2 * eye(n)  # 创建一个 n x n 的单位矩阵，并乘以 2
        for i in range(1, n - 1):
            m[i, i+1] = -1  # 设置矩阵 m 的第 (i, i+1) 位置为 -1
            m[i, i-1] = -1  # 设置矩阵 m 的第 (i, i-1) 位置为 -1
        m[0,1] = -1  # 设置矩阵 m 的第 (0, 1) 位置为 -1
        m[n-1, n-2] = -1  # 设置矩阵 m 的第 (n-1, n-2) 位置为 -1
        return m  # 返回生成的 Cartan 矩阵 m

    def basis(self):
        """
        返回 A_n 的独立生成器的数量
        """
        n = self.n  # 获取实例变量 self.n 的值
        return n**2 - 1  # 返回 A_n 的独立生成器数量的计算结果

    def lie_algebra(self):
        """
        返回与 A_n 相关的 Lie 代数
        """
        n = self.n  # 获取实例变量 self.n 的值
        return "su(" + str(n + 1) + ")"  # 返回形如 "su(n+1)" 的字符串

    def dynkin_diagram(self):
        n = self.n  # 获取实例变量 self.n 的值
        diag = "---".join("0" for i in range(1, n+1)) + "\n"  # 创建动金图的表示字符串
        diag += "   ".join(str(i) for i in range(1, n+1))  # 将数字序列添加到图中
        return diag  # 返回动金图的字符串表示
```