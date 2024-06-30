# `D:\src\scipysrc\sympy\sympy\tensor\array\array_comprehension.py`

```
    @property
    def function(self):
        """
        返回被应用于限制条件中的函数。

        示例
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.function
        10*i + j
        """
        return self._args[0]

    @property
    def limits(self):
        """
        返回将在扩展数组时应用的限制条件列表。

        示例
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.limits
        ((i, 1, 4), (j, 1, 3))
        """
        return self._limits


这段代码定义了一个名为 `ArrayComprehension` 的类，用于生成列表推导式。它包含了两个属性：`function` 和 `limits`。`function` 属性返回被应用于限制条件中的函数，`limits` 属性返回将在扩展数组时应用的限制条件列表。每个属性都有相应的示例来说明其用法和返回值。
    def free_symbols(self):
        """
        返回数组中的自由符号集合。
        在边界中出现的变量应该从自由符号集合中排除。

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.free_symbols
        set()
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.free_symbols
        {k}
        """
        expr_free_sym = self.function.free_symbols
        # 遍历限制条件，排除变量，并将每个限制的自由符号并入集合
        for var, inf, sup in self._limits:
            expr_free_sym.discard(var)
            curr_free_syms = inf.free_symbols.union(sup.free_symbols)
            expr_free_sym = expr_free_sym.union(curr_free_syms)
        return expr_free_sym

    @property
    def variables(self):
        """
        返回限制条件中的变量元组。

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.variables
        [i, j]
        """
        return [l[0] for l in self._limits]

    @property
    def bound_symbols(self):
        """
        返回所有虚拟变量的列表。

        Note
        ====

        注意，所有变量都是虚拟变量，因为没有下限或上限的限制是不被接受的。
        """
        return [l[0] for l in self._limits if len(l) != 1]

    @property
    def shape(self):
        """
        返回扩展数组的形状，其中可能包含符号。

        Note
        ====

        在计算形状时，包括下限和上限。

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.shape
        (4, 3)
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.shape
        (4, k + 3)
        """
        return self._shape

    @property
    def is_shape_numeric(self):
        """
        检查数组是否为形状数值化，即没有符号维度。

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.is_shape_numeric
        True
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.is_shape_numeric
        False
        """
        for _, inf, sup in self._limits:
            if Basic(inf, sup).atoms(Symbol):
                return False
        return True
    def rank(self):
        """
        The rank of the expanded array.

        Returns
        =======
        int
            The rank of the array.

        Examples
        ========
        
        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.rank()
        2
        """
        # 返回存储在 self._rank 中的数组秩
        return self._rank

    def __len__(self):
        """
        The length of the expanded array which means the number
        of elements in the array.

        Raises
        ======
        ValueError
            When the length of the array is symbolic.

        Returns
        =======
        int
            The length of the array.

        Examples
        ========
        
        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> len(a)
        12
        """
        # 如果数组的长度是符号化的，抛出 ValueError 异常
        if self._loop_size.free_symbols:
            raise ValueError('Symbolic length is not supported')
        # 返回存储在 self._loop_size 中的数组长度
        return self._loop_size

    @classmethod
    def _check_limits_validity(cls, function, limits):
        """
        Check the validity of the limits for array comprehension.

        Parameters
        ==========
        function : callable
            The function defining the array elements.
        limits : iterable of tuples
            Tuples defining the variables, lower and upper bounds.

        Returns
        =======
        list of tuples
            Validated and processed limits for array comprehension.

        Raises
        ======
        TypeError
            If bounds are not expressions or not a combination of Integer and Symbol.
        ValueError
            If lower bound is not less than upper bound or variable is part of its own bounds.
        """
        # 将限制条件进行符号化处理
        new_limits = []
        for var, inf, sup in limits:
            var = _sympify(var)
            inf = _sympify(inf)
            # 如果上限是列表形式，则转换为元组
            if isinstance(sup, list):
                sup = Tuple(*sup)
            else:
                sup = _sympify(sup)
            new_limits.append(Tuple(var, inf, sup))
            # 检查界限是否为表达式，且由整数和符号组合而成
            if any((not isinstance(i, Expr)) or i.atoms(Symbol, Integer) != i.atoms()
                                                                for i in [inf, sup]):
                raise TypeError('Bounds should be an Expression(combination of Integer and Symbol)')
            # 检查下限是否小于上限
            if (inf > sup) == True:
                raise ValueError('Lower bound should be inferior to upper bound')
            # 检查变量是否在其边界中
            if var in inf.free_symbols or var in sup.free_symbols:
                raise ValueError('Variable should not be part of its bounds')
        # 返回处理后的限制条件列表
        return new_limits

    @classmethod
    def _calculate_shape_from_limits(cls, limits):
        """
        Calculate the shape of the array from its limits.

        Parameters
        ==========
        limits : iterable of tuples
            Tuples defining the variables, lower and upper bounds.

        Returns
        =======
        tuple
            Shape of the array derived from the limits.
        """
        # 根据限制条件计算数组的形状（维度）
        return tuple([sup - inf + 1 for _, inf, sup in limits])

    @classmethod
    def _calculate_loop_size(cls, shape):
        """
        Calculate the total number of elements in the array given its shape.

        Parameters
        ==========
        shape : tuple
            Shape of the array.

        Returns
        =======
        int
            Total number of elements in the array.
        """
        # 如果形状为空，返回 0
        if not shape:
            return 0
        # 计算数组的总元素数
        loop_size = 1
        for l in shape:
            loop_size = loop_size * l
        return loop_size

    def doit(self, **hints):
        """
        Execute the computation if shape is numeric; otherwise, return self.

        Parameters
        ==========
        **hints : keyword arguments
            Additional hints for computation.

        Returns
        =======
        ArrayComprehension or ImmutableDenseNDimArray
            Result of the computation or the array itself if shape is not numeric.
        """
        # 如果形状不是数值型，直接返回 self
        if not self.is_shape_numeric:
            return self
        # 执行数组扩展操作并返回结果
        return self._expand_array()

    def _expand_array(self):
        """
        Expand the array based on its limits and return an ImmutableDenseNDimArray.

        Returns
        =======
        ImmutableDenseNDimArray
            Expanded array.
        """
        # 使用 itertools.product 生成所有可能的索引组合
        res = []
        for values in itertools.product(*[range(inf, sup+1)
                                        for var, inf, sup
                                        in self._limits]):
            # 获取每个索引组合对应的数组元素并加入结果列表中
            res.append(self._get_element(values))
        # 返回不可变的多维数组对象，作为数组的扩展结果
        return ImmutableDenseNDimArray(res, self.shape)
    def _get_element(self, values):
        # 从 self.function 开始，依次用 values 中的值替换 self.variables 中的变量，并逐步求值
        temp = self.function
        for var, val in zip(self.variables, values):
            temp = temp.subs(var, val)
        # 返回替换完成后的表达式结果
        return temp

    def tolist(self):
        """Transform the expanded array to a list.

        Raises
        ======

        ValueError : When there is a symbolic dimension
            如果数组中存在符号维度，则抛出 ValueError 异常

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tolist()
        [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
        """
        if self.is_shape_numeric:
            # 如果数组的形状是数值型，则展开数组并转换为列表返回
            return self._expand_array().tolist()

        # 如果数组的形状不是数值型，则抛出 ValueError 异常
        raise ValueError("A symbolic array cannot be expanded to a list")

    def tomatrix(self):
        """Transform the expanded array to a matrix.

        Raises
        ======

        ValueError : When there is a symbolic dimension
            如果数组中存在符号维度，则抛出 ValueError 异常
        ValueError : When the rank of the expanded array is not equal to 2
            如果数组的秩不等于 2，则抛出 ValueError 异常

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tomatrix()
        Matrix([
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33],
        [41, 42, 43]])
        """
        from sympy.matrices import Matrix

        if not self.is_shape_numeric:
            # 如果数组的形状不是数值型，则抛出 ValueError 异常
            raise ValueError("A symbolic array cannot be expanded to a matrix")
        if self._rank != 2:
            # 如果数组的秩不等于 2，则抛出 ValueError 异常
            raise ValueError('Dimensions must be of size of 2')

        # 展开数组并转换为矩阵类型返回
        return Matrix(self._expand_array().tomatrix())
# 定义一个函数用于判断给定的变量是否为 lambda 函数
def isLambda(v):
    # 创建一个简单的 lambda 函数并判断类型是否相同，并且名称是否相同
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

# 定义一个名为 ArrayComprehensionMap 的类，继承自 ArrayComprehension
class ArrayComprehensionMap(ArrayComprehension):
    '''
    A subclass of ArrayComprehension dedicated to map external function lambda.

    Notes
    =====

    Only the lambda function is considered.
    At most one argument in lambda function is accepted in order to avoid ambiguity
    in value assignment.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehensionMap
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehensionMap(lambda: 1, (i, 1, 4))
    >>> a.doit()
    [1, 1, 1, 1]
    >>> b = ArrayComprehensionMap(lambda a: a+1, (j, 1, 4))
    >>> b.doit()
    [2, 3, 4, 5]

    '''
    
    # 定义 __new__ 方法，用于创建类的新实例
    def __new__(cls, function, *symbols, **assumptions):
        # 检查 symbols 中每个元组的长度是否为3
        if any(len(l) != 3 or None for l in symbols):
            raise ValueError('ArrayComprehension requires values lower and upper bound'
                              ' for the expression')

        # 检查给定的 function 是否为 lambda 函数
        if not isLambda(function):
            raise ValueError('Data type not supported')

        # 检查并生成参数列表
        arglist = cls._check_limits_validity(function, symbols)
        # 使用 Basic 类的 __new__ 方法创建一个新的实例
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args  # 设置实例的 _limits 属性
        obj._shape = cls._calculate_shape_from_limits(obj._limits)  # 计算实例的形状
        obj._rank = len(obj._shape)  # 计算实例的秩
        obj._loop_size = cls._calculate_loop_size(obj._shape)  # 计算循环大小
        obj._lambda = function  # 设置实例的 _lambda 属性
        return obj

    @property
    def func(self):
        # 定义一个匿名类，并返回一个新的 ArrayComprehensionMap 实例
        class _(ArrayComprehensionMap):
            def __new__(cls, *args, **kwargs):
                return ArrayComprehensionMap(self._lambda, *args, **kwargs)
        return _

    # 定义一个私有方法 _get_element，用于获取元素
    def _get_element(self, values):
        temp = self._lambda  # 将实例的 _lambda 属性赋值给 temp
        # 根据 _lambda 的参数个数进行不同的处理
        if self._lambda.__code__.co_argcount == 0:
            temp = temp()  # 若参数个数为 0，则调用 _lambda 并赋值给 temp
        elif self._lambda.__code__.co_argcount == 1:
            # 若参数个数为 1，则使用 functools.reduce 计算 values 的乘积，并将结果传递给 _lambda 并赋值给 temp
            temp = temp(functools.reduce(lambda a, b: a*b, values))
        return temp  # 返回 temp
```