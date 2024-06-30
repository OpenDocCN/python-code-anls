# `D:\src\scipysrc\sympy\sympy\polys\constructor.py`

```
"""Tools for constructing domains for expressions. """
从 math 模块导入 prod 函数，用于计算列表的乘积

从 sympy.core 模块导入 sympify 函数，用于将输入转换为 SymPy 对象
从 sympy.core.evalf 模块导入 pure_complex 函数，用于检查一个对象是否是纯复数
从 sympy.core.sorting 模块导入 ordered 函数，用于对对象进行排序
从 sympy.polys.domains 模块导入 ZZ, QQ, ZZ_I, QQ_I, EX 对象，分别表示整数环、有理数环、整数代数扩展、有理数代数扩展、表达式环
从 sympy.polys.domains.complexfield 模块导入 ComplexField 类，表示复数域
从 sympy.polys.domains.realfield 模块导入 RealField 类，表示实数域
从 sympy.polys.polyoptions 模块导入 build_options 函数，用于构建多项式选项
从 sympy.polys.polyutils 模块导入 parallel_dict_from_basic 函数，用于从基本对象创建并行字典
从 sympy.utilities 模块导入 public 装饰器，用于将函数标记为公共 API

定义 _construct_simple 函数，接受 coeffs 和 opt 两个参数，处理简单的域，如整数、有理数、实数和代数域

# Handle simple domains, e.g.: ZZ, QQ, RR and algebraic domains.
def _construct_simple(coeffs, opt):
    rationals = floats = complexes = algebraics = False  # 初始化布尔变量，用于标记是否存在有理数、浮点数、复数和代数数
    float_numbers = []  # 初始化空列表，用于存储浮点数系数

    if opt.extension is True:
        is_algebraic = lambda coeff: coeff.is_number and coeff.is_algebraic
    else:
        is_algebraic = lambda coeff: False

    for coeff in coeffs:
        if coeff.is_Rational:  # 如果系数是有理数
            if not coeff.is_Integer:  # 如果不是整数有理数
                rationals = True  # 存在有理数
        elif coeff.is_Float:  # 如果系数是浮点数
            if algebraics:  # 如果已经存在代数数
                # there are both reals and algebraics -> EX
                return False  # 存在实数和代数数的情况，返回 False
            else:
                floats = True  # 存在浮点数
                float_numbers.append(coeff)  # 将浮点数系数添加到列表中
        else:
            is_complex = pure_complex(coeff)  # 检查是否是纯复数
            if is_complex:  # 如果是纯复数
                complexes = True  # 存在复数
                x, y = is_complex
                if x.is_Rational and y.is_Rational:  # 如果实部和虚部都是有理数
                    if not (x.is_Integer and y.is_Integer):  # 如果不是整数有理数
                        rationals = True  # 存在有理数
                    continue
                else:
                    floats = True  # 存在浮点数
                    if x.is_Float:
                        float_numbers.append(x)  # 将实部添加到浮点数列表中
                    if y.is_Float:
                        float_numbers.append(y)  # 将虚部添加到浮点数列表中
            elif is_algebraic(coeff):  # 如果是代数数
                if floats:  # 如果已经存在浮点数
                    # there are both algebraics and reals -> EX
                    return False  # 存在代数数和实数的情况，返回 False
                algebraics = True  # 存在代数数
            else:
                # this is a composite domain, e.g. ZZ[X], EX
                return None  # 这是一个复合域，例如 ZZ[X]、EX

    # Use the maximum precision of all coefficients for the RR or CC
    # precision
    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53  # 计算所有浮点数系数的最大精度

    if algebraics:  # 如果存在代数数
        domain, result = _construct_algebraic(coeffs, opt)  # 构建代数扩展域
    else:
        if floats and complexes:  # 如果存在浮点数和复数
            domain = ComplexField(prec=max_prec)  # 使用最大精度构建复数域
        elif floats:  # 如果只存在浮点数
            domain = RealField(prec=max_prec)  # 使用最大精度构建实数域
        elif rationals or opt.field:  # 如果存在有理数或者指定了域选项
            domain = QQ_I if complexes else QQ  # 如果存在复数则选择有理数代数扩展，否则选择有理数环
        else:
            domain = ZZ_I if complexes else ZZ  # 如果存在复数则选择整数代数扩展，否则选择整数环

        result = [domain.from_sympy(coeff) for coeff in coeffs]  # 将系数转换为对应域中的对象

    return domain, result  # 返回所选域和结果列表


def _construct_algebraic(coeffs, opt):
    """We know that coefficients are algebraic so construct the extension. """
    from sympy.polys.numberfields import primitive_element

    exts = set()

    # Placeholder for constructing algebraic extensions
    # 定义函数 build_trees，用于根据参数构建树形结构
    def build_trees(args):
        # 初始化空列表用于存储树结构
        trees = []
        # 遍历参数列表
        for a in args:
            # 如果参数 a 是有理数
            if a.is_Rational:
                # 创建一个有理数节点 ('Q', QQ.from_sympy(a))
                tree = ('Q', QQ.from_sympy(a))
            # 如果参数 a 是加法表达式
            elif a.is_Add:
                # 创建一个加法节点 ('+', build_trees(a.args))
                tree = ('+', build_trees(a.args))
            # 如果参数 a 是乘法表达式
            elif a.is_Mul:
                # 创建一个乘法节点 ('*', build_trees(a.args))
                tree = ('*', build_trees(a.args))
            # 如果参数 a 不属于以上任何类型
            else:
                # 创建一个特殊节点 ('e', a)，并将 a 加入到 exts 集合中
                tree = ('e', a)
                exts.add(a)
            # 将创建的节点添加到 trees 列表中
            trees.append(tree)
        # 返回构建好的树结构列表
        return trees

    # 调用 build_trees 函数构建树形结构列表
    trees = build_trees(coeffs)
    # 将 exts 集合转换为有序列表
    exts = list(ordered(exts))

    # 调用 primitive_element 函数计算原始元素 g、生成空间 span 和多项式列表 H
    g, span, H = primitive_element(exts, ex=True, polys=True)
    # 计算根 root，即 span 中各元素与 exts 中对应元素的线性组合
    root = sum(s*ext for s, ext in zip(span, exts))

    # 使用 QQ.algebraic_field 构建代数域 domain，g.rep.to_list() 返回 g 的系数列表
    domain, g = QQ.algebraic_field((g, root)), g.rep.to_list()

    # 使用 domain.dtype.from_list 为 H 中每个多项式创建扩展域元素的列表 exts_dom
    exts_dom = [domain.dtype.from_list(h, g, QQ) for h in H]
    # 使用 zip 将 exts 和 exts_dom 组合成字典 exts_map
    exts_map = dict(zip(exts, exts_dom))

    # 定义函数 convert_tree，用于将树形结构转换为相应的代数域元素
    def convert_tree(tree):
        op, args = tree
        # 如果操作符为 'Q'，返回 domain.dtype.from_list([args], g, QQ)
        if op == 'Q':
            return domain.dtype.from_list([args], g, QQ)
        # 如果操作符为 '+'，返回 sum((convert_tree(a) for a in args), domain.zero)
        elif op == '+':
            return sum((convert_tree(a) for a in args), domain.zero)
        # 如果操作符为 '*'，返回 prod(convert_tree(a) for a in args)
        elif op == '*':
            return prod(convert_tree(a) for a in args)
        # 如果操作符为 'e'，返回 exts_map[args]
        elif op == 'e':
            return exts_map[args]
        # 如果操作符未知，抛出 RuntimeError 异常
        else:
            raise RuntimeError

    # 对每个树形结构调用 convert_tree 函数，将其转换为代数域元素，存储在 result 列表中
    result = [convert_tree(tree) for tree in trees]

    # 返回最终的代数域 domain 和转换后的结果列表 result
    return domain, result
`
def _construct_composite(coeffs, opt):
    """Handle composite domains, e.g.: ZZ[X], QQ[X], ZZ(X), QQ(X). """
    # 初始化 numers 和 denoms 列表，用于存储分子和分母
    numers, denoms = [], []

    # 遍历 coeffs 列表，将每个系数分解为分子和分母
    for coeff in coeffs:
        numer, denom = coeff.as_numer_denom()

        numers.append(numer)
        denoms.append(denom)

    # 使用 parallel_dict_from_basic 函数创建多项式和生成元字典
    polys, gens = parallel_dict_from_basic(numers + denoms)  # XXX: sorting
    # 如果没有生成元，返回 None
    if not gens:
        return None

    # 如果 opt.composite 为 None，进行一些额外的检查
    if opt.composite is None:
        if any(gen.is_number and gen.is_algebraic for gen in gens):
            return None  # 生成元是数字型的，使用 EX 更合适

        all_symbols = set()

        # 检查生成元之间是否存在代数关系
        for gen in gens:
            symbols = gen.free_symbols

            if all_symbols & symbols:
                return None  # 存在代数关系
            else:
                all_symbols |= symbols

    # 获取生成元数量和多项式数量的一半
    n = len(gens)
    k = len(polys)//2

    # 将多项式分为分子和分母
    numers = polys[:k]
    denoms = polys[k:]

    # 如果 opt.field 为 True，设定分数标志为 True，否则检查是否为分数域
    if opt.field:
        fractions = True
    else:
        fractions, zeros = False, (0,)*n

        # 检查分母是否为单位元或包含零元
        for denom in denoms:
            if len(denom) > 1 or zeros not in denom:
                fractions = True
                break

    # 初始化系数集合
    coeffs = set()

    # 如果不是分数，进行分子除以分母的计算
    if not fractions:
        for numer, denom in zip(numers, denoms):
            denom = denom[zeros]

            for monom, coeff in numer.items():
                coeff /= denom
                coeffs.add(coeff)
                numer[monom] = coeff
    else:
        # 将所有系数加入 coeffs 集合中
        for numer, denom in zip(numers, denoms):
            coeffs.update(list(numer.values()))
            coeffs.update(list(denom.values()))

    # 初始化浮点数和复数标志
    rationals = floats = complexes = False
    float_numbers = []

    # 遍历所有系数，确定其类型
    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            floats = True
            float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex is not None:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)

    # 确定浮点数的精度，默认为 53
    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53

    # 根据系数类型选择合适的底域
    if floats and complexes:
        ground = ComplexField(prec=max_prec)
    elif floats:
        ground = RealField(prec=max_prec)
    elif complexes:
        if rationals:
            ground = QQ_I
        else:
            ground = ZZ_I
    elif rationals:
        ground = QQ
    else:
        ground = ZZ

    # 初始化结果列表
    result = []
    # 如果没有传入 fractions 参数，则执行以下逻辑
    if not fractions:
        # 根据生成器列表 gens 创建一个多项式环 domain
        domain = ground.poly_ring(*gens)
    
        # 遍历 numers 列表中的每个字典 numer
        for numer in numers:
            # 遍历 numer 字典中的每个单项式 monom 和系数 coeff
            for monom, coeff in numer.items():
                # 将 coeff 从 SymPy 类型转换为 ground 类型
                numer[monom] = ground.from_sympy(coeff)
    
            # 将处理过的 numer 添加到 result 列表中，作为 domain 对象的一个实例
            result.append(domain(numer))
    else:
        # 如果传入了 fractions 参数，则执行以下逻辑
        # 根据生成器列表 gens 创建一个分式域 domain
        domain = ground.frac_field(*gens)
    
        # 同时遍历 numers 和 denoms 列表中的每对字典 numer 和 denom
        for numer, denom in zip(numers, denoms):
            # 遍历 numer 字典中的每个单项式 monom 和系数 coeff，并转换系数类型
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)
    
            # 遍历 denom 字典中的每个单项式 monom 和系数 coeff，并转换系数类型
            for monom, coeff in denom.items():
                denom[monom] = ground.from_sympy(coeff)
    
            # 将处理过的 (numer, denom) 元组添加到 result 列表中，作为 domain 对象的一个实例
            result.append(domain((numer, denom)))
    
    # 返回生成的 domain 对象以及结果列表 result
    return domain, result
def _construct_expression(coeffs, opt):
    """Constructs domain elements from coefficients.

    Parameters
    ==========
    coeffs: list
        List of coefficients to convert into domain elements.
    opt: options
        Options affecting the conversion process.

    Returns
    =======
    domain: Domain
        Minimal domain that can represent the coefficients.
    result: list
        List of domain elements corresponding to the coefficients.
    """
    # Default domain for expressions
    domain, result = EX, []

    # Convert each coefficient to domain element
    for coeff in coeffs:
        result.append(domain.from_sympy(coeff))

    return domain, result


@public
def construct_domain(obj, **args):
    """Construct a minimal domain for a list of expressions.

    Explanation
    ===========
    Given a list of normal SymPy expressions (of type :py:class:`~.Expr`)
    ``construct_domain`` will find a minimal :py:class:`~.Domain` that can
    represent those expressions. The expressions will be converted to elements
    of the domain and both the domain and the domain elements are returned.

    Parameters
    ==========
    obj: list or dict
        The expressions to build a domain for.

    **args: keyword arguments
        Options that affect the choice of domain.

    Returns
    =======
    (K, elements): tuple
        Domain K that can represent the expressions and the list or dict
        of domain elements representing the same expressions as elements of K.

    Examples
    ========
    See docstring for examples of usage.
    """
    # Build options based on keyword arguments
    opt = build_options(args)

    # Check if obj is iterable
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            # If obj is a dictionary, unpack keys and values into monoms and coeffs
            if not obj:
                monoms, coeffs = [], []
            else:
                monoms, coeffs = list(zip(*list(obj.items())))
        else:
            # If obj is a list, treat it as coefficients
            coeffs = obj
    else:
        # If obj is not iterable, treat it as a single coefficient
        coeffs = [obj]
    # 将 coeffs 列表中的每个元素转换为 sympy 的表达式对象
    coeffs = list(map(sympify, coeffs))
    # 调用 _construct_simple 函数处理 coeffs 和 opt，返回处理结果
    result = _construct_simple(coeffs, opt)

    # 如果结果不为 None
    if result is not None:
        # 如果结果不是 False，则解构结果为 domain 和 coeffs
        if result is not False:
            domain, coeffs = result
        # 如果结果是 False，则调用 _construct_expression 函数重新构造 coeffs 和 opt
        else:
            domain, coeffs = _construct_expression(coeffs, opt)
    # 如果结果为 None
    else:
        # 如果 opt.composite 为 False，则结果设为 None
        if opt.composite is False:
            result = None
        # 否则，调用 _construct_composite 函数处理 coeffs 和 opt
        else:
            result = _construct_composite(coeffs, opt)

        # 如果处理结果不为 None，则解构结果为 domain 和 coeffs
        if result is not None:
            domain, coeffs = result
        # 如果处理结果为 None，则调用 _construct_expression 函数重新构造 coeffs 和 opt
        else:
            domain, coeffs = _construct_expression(coeffs, opt)

    # 如果 obj 具有 '__iter__' 属性（即 obj 是可迭代的）
    if hasattr(obj, '__iter__'):
        # 如果 obj 是字典类型，则返回 domain 和由 monoms 和 coeffs 组成的字典
        if isinstance(obj, dict):
            return domain, dict(list(zip(monoms, coeffs)))
        # 否则，返回 domain 和 coeffs
        else:
            return domain, coeffs
    # 如果 obj 不是可迭代的，则返回 domain 和 coeffs 的第一个元素
    else:
        return domain, coeffs[0]
```