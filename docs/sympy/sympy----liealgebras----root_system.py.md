# `D:\src\scipysrc\sympy\sympy\liealgebras\root_system.py`

```
from .cartan_type import CartanType  # 导入CartanType类，用于处理Cartan类型的相关操作
from sympy.core.basic import Atom  # 导入Atom类，这是Sympy库中的基本类

class RootSystem(Atom):
    """Represent the root system of a simple Lie algebra

    Every simple Lie algebra has a unique root system.  To find the root
    system, we first consider the Cartan subalgebra of g, which is the maximal
    abelian subalgebra, and consider the adjoint action of g on this
    subalgebra.  There is a root system associated with this action. Now, a
    root system over a vector space V is a set of finite vectors Phi (called
    roots), which satisfy:

    1.  The roots span V
    2.  The only scalar multiples of x in Phi are x and -x
    3.  For every x in Phi, the set Phi is closed under reflection
        through the hyperplane perpendicular to x.
    4.  If x and y are roots in Phi, then the projection of y onto
        the line through x is a half-integral multiple of x.

    Now, there is a subset of Phi, which we will call Delta, such that:
    1.  Delta is a basis of V
    2.  Each root x in Phi can be written x = sum k_y y for y in Delta

    The elements of Delta are called the simple roots.
    Therefore, we see that the simple roots span the root space of a given
    simple Lie algebra.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Root_system
    .. [2] Lie Algebras and Representation Theory - Humphreys

    """

    def __new__(cls, cartantype):
        """Create a new RootSystem object

        This method assigns an attribute called cartan_type to each instance of
        a RootSystem object.  When an instance of RootSystem is called, it
        needs an argument, which should be an instance of a simple Lie algebra.
        We then take the CartanType of this argument and set it as the
        cartan_type attribute of the RootSystem instance.

        """
        obj = Atom.__new__(cls)  # 调用父类Atom的构造方法创建新的RootSystem对象
        obj.cartan_type = CartanType(cartantype)  # 使用给定的cartantype参数创建一个CartanType对象，并将其赋给cartan_type属性
        return obj

    def simple_roots(self):
        """Generate the simple roots of the Lie algebra

        The rank of the Lie algebra determines the number of simple roots that
        it has.  This method obtains the rank of the Lie algebra, and then uses
        the simple_root method from the Lie algebra classes to generate all the
        simple roots.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> roots = c.simple_roots()
        >>> roots
        {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}

        """
        n = self.cartan_type.rank()  # 获取Lie代数的秩，确定简单根的数量
        roots = {i: self.cartan_type.simple_root(i) for i in range(1, n+1)}  # 使用CartanType对象的simple_root方法生成所有简单根
        return roots  # 返回生成的简单根字典
    def all_roots(self):
        """Generate all the roots of a given root system
        
        The result is a dictionary where the keys are integer numbers.  It
        generates the roots by getting the dictionary of all positive roots
        from the bases classes, and then taking each root, and multiplying it
        by -1 and adding it to the dictionary.  In this way all the negative
        roots are generated.
        """
        # 获取所有正根的字典
        alpha = self.cartan_type.positive_roots()
        # 获取字典中所有键值作为列表
        keys = list(alpha.keys())
        # 计算键的最大值
        k = max(keys)
        # 遍历所有键值
        for val in keys:
            # 递增键的计数
            k += 1
            # 获取当前键对应的根
            root = alpha[val]
            # 创建当前根的负根
            newroot = [-x for x in root]
            # 将负根添加到字典中
            alpha[k] = newroot
        # 返回包含所有根的字典
        return alpha

    def root_space(self):
        """Return the span of the simple roots
        
        The root space is the vector space spanned by the simple roots, i.e. it
        is a vector space with a distinguished basis, the simple roots.  This
        method returns a string that represents the root space as the span of
        the simple roots, alpha[1],...., alpha[n].
        
        Examples
        ========
        
        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.root_space()
        'alpha[1] + alpha[2] + alpha[3]'
        """
        # 获取 Cartan 类型的秩
        n = self.cartan_type.rank()
        # 生成表示简单根空间的字符串
        rs = " + ".join("alpha["+str(i) +"]" for i in range(1, n+1))
        # 返回结果字符串
        return rs

    def add_simple_roots(self, root1, root2):
        """Add two simple roots together
        
        The function takes as input two integers, root1 and root2.  It then
        uses these integers as keys in the dictionary of simple roots, and gets
        the corresponding simple roots, and then adds them together.
        
        Examples
        ========
        
        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> newroot = c.add_simple_roots(1, 2)
        >>> newroot
        [1, 0, -1, 0]
        """
        # 获取所有简单根的字典
        alpha = self.simple_roots()
        # 检查输入的根是否有效
        if root1 > len(alpha) or root2 > len(alpha):
            raise ValueError("You've used a root that doesn't exist!")
        # 获取对应于输入根的简单根
        a1 = alpha[root1]
        a2 = alpha[root2]
        # 将两个简单根相加
        newroot = [_a1 + _a2 for _a1, _a2 in zip(a1, a2)]
        # 返回新的根
        return newroot
    def add_as_roots(self, root1, root2):
        """Add two roots together if and only if their sum is also a root

        It takes as input two vectors which should be roots.  It then computes
        their sum and checks if it is in the list of all possible roots.  If it
        is, it returns the sum.  Otherwise it returns a string saying that the
        sum is not a root.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1])
        [1, 0, 0, -1]
        >>> c.add_as_roots([1, -1, 0, 0], [0, 0, -1, 1])
        'The sum of these two roots is not a root'

        """
        alpha = self.all_roots()  # 获取所有可能的根
        newroot = [r1 + r2 for r1, r2 in zip(root1, root2)]  # 计算两个根的和
        if newroot in alpha.values():  # 检查新的根是否在所有可能的根中
            return newroot  # 如果是根，则返回新的根
        else:
            return "The sum of these two roots is not a root"  # 如果不是根，则返回提示信息


    def cartan_matrix(self):
        """Cartan matrix of Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.cartan_matrix()
        Matrix([
            [ 2, -1,  0],
            [-1,  2, -1],
            [ 0, -1,  2]])
        """
        return self.cartan_type.cartan_matrix()  # 返回与该根系统相关联的李代数的 Cartan 矩阵


    def dynkin_diagram(self):
        """Dynkin diagram of the Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> print(c.dynkin_diagram())
        0---0---0
        1   2   3
        """
        return self.cartan_type.dynkin_diagram()  # 返回与该根系统相关联的李代数的 Dynkin 图
```