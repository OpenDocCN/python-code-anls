# `D:\src\scipysrc\sympy\sympy\tensor\toperators.py`

```
# 导入 sympy 库中的相关模块和函数
from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul

# 定义一个类 PartialDerivative，继承自 TensExpr
class PartialDerivative(TensExpr):
    """
    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    """
    # 定义一个特殊方法 __new__，用于创建新的对象实例
    def __new__(cls, expr, *variables):

        # 如果表达式是 PartialDerivative 的实例，则扁平化处理
        if isinstance(expr, PartialDerivative):
            # 将变量列表扩展为表达式对象中的变量加上传入的额外变量
            variables = expr.variables + variables
            # 更新表达式为表达式对象中的表达式
            expr = expr.expr

        # 调用 _contract_indices_for_derivative 静态方法处理张量的指标收缩
        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables)

        # 使用 TensExpr 类的 __new__ 方法创建新的张量表达式对象
        obj = TensExpr.__new__(cls, *args)

        # 设置对象的索引、自由指标和虚指标属性
        obj._indices = indices
        obj._free = free
        obj._dum = dum
        return obj

    # coeff 属性的 getter 方法，返回 S.One
    @property
    def coeff(self):
        return S.One

    # nocoeff 属性的 getter 方法，返回自身
    @property
    def nocoeff(self):
        return self

    # 类方法 _contract_indices_for_derivative，处理求导时的指标收缩
    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables):
        # 初始化变量列表，用于存储反向价值的变量
        variables_opposite_valence = []

        # 遍历传入的变量列表
        for i in variables:
            # 如果变量是 Tensor 类型
            if isinstance(i, Tensor):
                # 获取变量的自由指标
                i_free_indices = i.get_free_indices()
                # 将变量替换为自由指标取反后的新变量，加入反向价值的变量列表
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            # 如果变量是 Symbol 类型
            elif isinstance(i, Symbol):
                # 直接加入反向价值的变量列表
                variables_opposite_valence.append(i)

        # 调用 TensMul 类的 _tensMul_contract_indices 方法，处理张量乘积的指标收缩
        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence, replace_indices=True)

        # 遍历处理后的参数列表
        for i in range(1, len(args)):
            # 如果参数是 Tensor 类型
            args_i = args[i]
            if isinstance(args_i, Tensor):
                # 获取参数的自由指标
                i_indices = args[i].get_free_indices()
                # 将参数替换为自由指标取反后的新参数
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        # 返回处理后的参数列表、指标、自由指标和虚指标
        return args, indices, free, dum

    # doit 方法，执行指定的求值操作
    def doit(self, **hints):
        # 调用 _contract_indices_for_derivative 方法处理表达式和变量的指标收缩
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        # 使用对象的构造函数创建新的对象实例
        obj = self.func(*args)
        # 设置对象的索引、自由指标和虚指标属性
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        # 返回处理后的对象实例
        return obj
    # 扩展偏导数表达式，处理多个变量的偏导数
    def _expand_partial_derivative(self):
        # 调用_contract_indices_for_derivative方法，获取表达式的参数、指标、自由指标和哑指标
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        # 使用参数构造当前类的实例对象
        obj = self.func(*args)
        # 设置实例对象的指标、自由指标和哑指标属性
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        # 将结果初始化为当前对象
        result = obj

        # 如果第一个参数不含自由符号，则返回零
        if not args[0].free_symbols:
            return S.Zero
        # 如果表达式是TensAdd类型，则处理多个偏导数之和
        elif isinstance(obj.expr, TensAdd):
            # 对TensAdd类型的每个参数递归调用_expand_partial_derivative方法
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        # 如果表达式是TensMul类型，则处理多个偏导数之积
        elif isinstance(obj.expr, TensMul):
            # 如果变量列表长度为1，则处理单变量的偏导数
            if len(obj.variables) == 1:
                terms = []
                mulargs = list(obj.expr.args)
                # 遍历乘积的每个参数
                for ind in range(len(mulargs)):
                    # 如果参数不是数字，则处理其偏导数
                    if not isinstance(sympify(mulargs[ind]), Number):
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        # 构造新的TensMul对象，将偏导数替换到相应位置
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                # 从terms列表创建TensAdd对象作为结果
                result = TensAdd.fromiter(terms)
            else:
                # 如果变量列表长度大于1，则处理多变量的偏导数
                result = obj.expr  # 将结果初始化为原始表达式
                for v in obj.variables:
                    # 逐个对每个变量进行偏导数扩展操作
                    result = self.func(result, v)._expand_partial_derivative()

        return result

    # 执行偏导数操作，逐个变量进行求导
    def _perform_derivative(self):
        # 将结果初始化为表达式本身
        result = self.expr
        # 遍历每个变量
        for v in self.variables:
            # 如果结果是TensExpr类型，则调用_eval_partial_derivative方法求偏导数
            if isinstance(result, TensExpr):
                result = result._eval_partial_derivative(v)
            else:
                # 否则，根据变量的_diff_wrt属性判断是否可以进行求导
                if v._diff_wrt:
                    result = result._eval_derivative(v)
                else:
                    # 如果不可求导，则结果为零
                    result = S.Zero
        return result

    # 返回对象的_indices属性，即索引列表
    def get_indices(self):
        return self._indices

    # 返回对象的_free属性排序后的自由指标列表
    def get_free_indices(self):
        free = sorted(self._free, key=lambda x: x[1])
        return [i[0] for i in free]

    # 替换表达式中的指标，返回替换后的新对象
    def _replace_indices(self, repl):
        # 使用xreplace方法替换表达式中的指标
        expr = self.expr.xreplace(repl)
        # 使用mirrored字典替换变量中的指标
        mirrored = {-k: -v for k, v in repl.items()}
        variables = [i.xreplace(mirrored) for i in self.variables]
        # 返回替换后的新对象
        return self.func(expr, *variables)

    # 表达式属性，返回对象的第一个参数，即表达式本身
    @property
    def expr(self):
        return self.args[0]

    # 变量属性，返回对象的除第一个参数外的所有参数，即变量列表
    @property
    def variables(self):
        return self.args[1:]
    # 定义一个方法用于从表达式中提取数据，根据给定的替换字典
    def _extract_data(self, replacement_dict):
        # 导入必要的模块和函数：从.array模块中导入derive_by_array和tensorcontraction函数
        from .array import derive_by_array, tensorcontraction
        # 调用表达式对象的_extract_data方法，获取其中的索引和数组数据
        indices, array = self.expr._extract_data(replacement_dict)
        
        # 遍历每个变量
        for variable in self.variables:
            # 对每个变量调用_extract_data方法，获取变量的索引和数组数据
            var_indices, var_array = variable._extract_data(replacement_dict)
            # 将变量索引中的每个索引取负值
            var_indices = [-i for i in var_indices]
            
            # 将每个变量数组中的每个元素按照乘积形式分离为系数和数组
            coeff_array, var_array = zip(*[i.as_coeff_Mul() for i in var_array])
            
            # 记录操作前数组的维度
            dim_before = len(array.shape)
            # 使用derive_by_array函数对数组进行求导操作
            array = derive_by_array(array, var_array)
            # 记录操作后数组的维度
            dim_after = len(array.shape)
            # 计算维度增加量
            dim_increase = dim_after - dim_before
            # 重新排列数组的维度顺序，以便与原先的维度顺序相匹配
            array = permutedims(array, [i + dim_increase for i in range(dim_before)] + list(range(dim_increase)))
            # 将数组转换为可变形式
            array = array.as_mutable()
            
            # 获取变量的第一个索引
            varindex = var_indices[0]
            
            # 移除基向量的系数
            coeff_index = [0] + [slice(None) for i in range(len(indices))]
            for i, coeff in enumerate(coeff_array):
                coeff_index[0] = i
                array[tuple(coeff_index)] /= coeff
            
            # 如果变量的负索引在indices中存在
            if -varindex in indices:
                pos = indices.index(-varindex)
                # 对数组进行张量收缩操作
                array = tensorcontraction(array, (0, pos+1))
                # 从indices中移除该索引
                indices.pop(pos)
            else:
                # 否则将变量的索引添加到indices中
                indices.append(varindex)
        
        # 返回更新后的索引和数组数据
        return indices, array
```