# `D:\src\scipysrc\sympy\sympy\liealgebras\weyl_group.py`

```
# -*- coding: utf-8 -*-  # 指定源文件编码为UTF-8格式

# 导入CartanType类，位于当前包的cartan_type模块中
from .cartan_type import CartanType
# 导入mpmath库中的fac函数
from mpmath import fac
# 导入sympy.core.backend模块中的Matrix, eye, Rational, igcd函数
from sympy.core.backend import Matrix, eye, Rational, igcd
# 导入sympy.core.basic模块中的Atom类
from sympy.core.basic import Atom

# 定义WeylGroup类，继承自Atom类
class WeylGroup(Atom):

    """
    对于每个半单Lie群，我们有一个Weyl群。它是根系的等距群的子群。具体来说，它是通过沿着根正交的超平面的反射生成的子群。
    因此，Weyl群是反射群，所以Weyl群是有限的Coxeter群。
    """

    def __new__(cls, cartantype):
        # 创建一个新的实例对象
        obj = Atom.__new__(cls)
        # 初始化Weyl群的Cartan类型
        obj.cartan_type = CartanType(cartantype)
        return obj

    def generators(self):
        """
        这个方法为给定的Lie代数创建Weyl群的生成反射。对于秩为n的Lie代数，有n个不同的生成反射。
        此函数将它们作为列表返回。

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("F4")
        >>> c.generators()
        ['r1', 'r2', 'r3', 'r4']
        """
        # 获取Weyl群的秩
        n = self.cartan_type.rank()
        # 初始化生成器列表
        generators = []
        # 生成n个不同的生成反射并添加到列表中
        for i in range(1, n+1):
            reflection = "r"+str(i)
            generators.append(reflection)
        # 返回生成器列表
        return generators

    def group_order(self):
        """
        这个方法返回Weyl群的阶数。
        对于A、B、C、D和E类型，阶数取决于Lie代数的秩。
        对于F和G类型，阶数是固定的。

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("D4")
        >>> c.group_order()
        192.0
        """
        # 获取Weyl群的秩
        n = self.cartan_type.rank()
        # 根据不同的系列计算并返回阶数
        if self.cartan_type.series == "A":
            return fac(n+1)

        if self.cartan_type.series in ("B", "C"):
            return fac(n)*(2**n)

        if self.cartan_type.series == "D":
            return fac(n)*(2**(n-1))

        if self.cartan_type.series == "E":
            if n == 6:
                return 51840
            if n == 7:
                return 2903040
            if n == 8:
                return 696729600
        if self.cartan_type.series == "F":
            return 1152

        if self.cartan_type.series == "G":
            return 12
    def group_name(self):
        """
        This method returns some general information about the Weyl group for
        a given Lie algebra.  It returns the name of the group and the elements
        it acts on, if relevant.
        """
        # 获取 Cartan 类型的秩
        n = self.cartan_type.rank()
        
        # 如果 Cartan 类型系列为 "A"
        if self.cartan_type.series == "A":
            # 返回对称群 S(n+1)，作用在 n+1 个元素上的描述
            return "S"+str(n+1) + ": the symmetric group acting on " + str(n+1) + " elements."

        # 如果 Cartan 类型系列为 "B" 或 "C"
        if self.cartan_type.series in ("B", "C"):
            # 返回双超立方体群，作用在 2n 个元素上的描述
            return "The hyperoctahedral group acting on " + str(2*n) + " elements."

        # 如果 Cartan 类型系列为 "D"
        if self.cartan_type.series == "D":
            # 返回 demihypercube 的对称群，作用在 n 维空间上的描述
            return "The symmetry group of the " + str(n) + "-dimensional demihypercube."

        # 如果 Cartan 类型系列为 "E"
        if self.cartan_type.series == "E":
            # 对于不同的秩 n，返回不同的多胞形的对称群描述
            if n == 6:
                return "The symmetry group of the 6-polytope."
            elif n == 7:
                return "The symmetry group of the 7-polytope."
            elif n == 8:
                return "The symmetry group of the 8-polytope."

        # 如果 Cartan 类型系列为 "F"
        if self.cartan_type.series == "F":
            # 返回 24-胞体或 icositetrachoron 的对称群描述
            return "The symmetry group of the 24-cell, or icositetrachoron."

        # 如果 Cartan 类型系列为 "G"
        if self.cartan_type.series == "G":
            # 返回 D6，即阶数为 12 的二面角群，以及六边形的对称群描述
            return "D6, the dihedral group of order 12, and symmetry group of the hexagon."
    def element_order(self, weylelt):
        """
        This method returns the order of a given Weyl group element, which should
        be specified by the user in the form of products of the generating
        reflections, i.e. of the form r1*r2 etc.

        For types A-F, this method currently works by taking the matrix form of
        the specified element, and then finding what power of the matrix is the
        identity. It then returns this power.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> b = WeylGroup("B4")
        >>> b.element_order('r1*r4*r2')
        4
        """
        # 获取 Cartan 类型的秩
        n = self.cartan_type.rank()

        # 对于类型 A，计算元素的阶
        if self.cartan_type.series == "A":
            # 将指定元素转换为矩阵形式
            a = self.matrix_form(weylelt)
            order = 1
            # 循环直到矩阵等于单位矩阵
            while a != eye(n+1):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        # 对于类型 D，计算元素的阶
        if self.cartan_type.series == "D":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        # 对于类型 E，计算元素的阶
        if self.cartan_type.series == "E":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(8):  # 8 是 E 系列的特定阶数
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        # 对于类型 G，计算元素的阶
        if self.cartan_type.series == "G":
            elts = list(weylelt)
            reflections = elts[1::3]
            m = self.delete_doubles(reflections)
            while self.delete_doubles(m) != m:
                m = self.delete_doubles(m)
                reflections = m
            if len(reflections) % 2 == 1:
                return 2
            elif len(reflections) == 0:
                return 1
            else:
                if len(reflections) == 1:
                    return 2
                else:
                    m = len(reflections) // 2
                    lcm = (6 * m) / igcd(m, 6)
                order = lcm / m
                return order

        # 对于类型 F，计算元素的阶
        if self.cartan_type.series == 'F':
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(4):  # 4 是 F 系列的特定阶数
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        # 对于类型 B 和 C，计算元素的阶
        if self.cartan_type.series in ("B", "C"):
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order
    def delete_doubles(self, reflections):
        """
        This is a helper method for determining the order of an element in the
        Weyl group of G2.  It takes a Weyl element and if repeated simple reflections
        in it, it deletes them.
        """
        # 初始化计数器
        counter = 0
        # 创建副本以遍历
        copy = list(reflections)
        # 遍历副本列表
        for elt in copy:
            # 检查是否未到列表末尾
            if counter < len(copy)-1:
                # 检查相邻元素是否相同
                if copy[counter + 1] == elt:
                    # 若相同，删除相邻的两个元素
                    del copy[counter]
                    del copy[counter]
            # 增加计数器
            counter += 1

        # 返回处理后的副本列表
        return copy


    def coxeter_diagram(self):
        """
        This method returns the Coxeter diagram corresponding to a Weyl group.
        The Coxeter diagram can be obtained from a Lie algebra's Dynkin diagram
        by deleting all arrows; the Coxeter diagram is the undirected graph.
        The vertices of the Coxeter diagram represent the generating reflections
        of the Weyl group, $s_i$.  An edge is drawn between $s_i$ and $s_j$ if the order
        $m(i, j)$ of $s_is_j$ is greater than two.  If there is one edge, the order
        $m(i, j)$ is 3.  If there are two edges, the order $m(i, j)$ is 4, and if there
        are three edges, the order $m(i, j)$ is 6.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("B3")
        >>> print(c.coxeter_diagram())
        0---0===0
        1   2   3
        """
        # 获取 Cartan 类型的秩
        n = self.cartan_type.rank()
        # 根据不同的 Cartan 系列返回对应的 Coxeter 图
        if self.cartan_type.series in ("A", "D", "E"):
            return self.cartan_type.dynkin_diagram()

        if self.cartan_type.series in ("B", "C"):
            # 构建 Coxeter 图的字符串表示形式
            diag = "---".join("0" for i in range(1, n)) + "===0\n"
            diag += "   ".join(str(i) for i in range(1, n+1))
            return diag

        if self.cartan_type.series == "F":
            diag = "0---0===0---0\n"
            diag += "   ".join(str(i) for i in range(1, 5))
            return diag

        if self.cartan_type.series == "G":
            diag = "0≡≡≡0\n1   2"
            return diag
```