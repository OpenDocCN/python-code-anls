# `D:\src\scipysrc\sympy\sympy\printing\precedence.py`

```
# 默认操作符优先级字典，指定了一些基本类型的默认优先级
PRECEDENCE = {
    "Lambda": 1,           # Lambda 表达式的优先级
    "Xor": 10,             # 异或操作的优先级
    "Or": 20,              # 逻辑或操作的优先级
    "And": 30,             # 逻辑与操作的优先级
    "Relational": 35,      # 关系运算符的优先级
    "Add": 40,             # 加法操作的优先级
    "Mul": 50,             # 乘法操作的优先级
    "Pow": 60,             # 幂操作的优先级
    "Func": 70,            # 函数调用的优先级
    "Not": 100,            # 逻辑非操作的优先级
    "Atom": 1000,          # 原子操作的优先级
    "BitwiseOr": 36,       # 按位或操作的优先级
    "BitwiseXor": 37,      # 按位异或操作的优先级
    "BitwiseAnd": 38       # 按位与操作的优先级
}

# 指定特定类别的操作符优先级，这些优先级被视为继承的关系，不需要每个类别都列出来
# 请勿与除 StrPrinter 外的其他打印机一起使用
PRECEDENCE_VALUES = {
    "Equivalent": PRECEDENCE["Xor"],         # 等价操作符的优先级
    "Xor": PRECEDENCE["Xor"],                # 异或操作符的优先级
    "Implies": PRECEDENCE["Xor"],            # 蕴含操作符的优先级
    "Or": PRECEDENCE["Or"],                  # 逻辑或操作符的优先级
    "And": PRECEDENCE["And"],                # 逻辑与操作符的优先级
    "Add": PRECEDENCE["Add"],                # 加法操作符的优先级
    "Pow": PRECEDENCE["Pow"],                # 幂操作符的优先级
    "Relational": PRECEDENCE["Relational"],  # 关系运算符的优先级
    "Sub": PRECEDENCE["Add"],                # 减法操作符的优先级（与加法相同）
    "Not": PRECEDENCE["Not"],                # 逻辑非操作符的优先级
    "Function": PRECEDENCE["Func"],          # 函数调用操作符的优先级
    "NegativeInfinity": PRECEDENCE["Add"],   # 负无穷操作符的优先级（与加法相同）
    "MatAdd": PRECEDENCE["Add"],             # 矩阵加法操作符的优先级（与加法相同）
    "MatPow": PRECEDENCE["Pow"],             # 矩阵幂操作符的优先级（与幂相同）
    "MatrixSolve": PRECEDENCE["Mul"],        # 矩阵求解操作符的优先级（与乘法相同）
    "Mod": PRECEDENCE["Mul"],                # 取模操作符的优先级（与乘法相同）
    "TensAdd": PRECEDENCE["Add"],            # 张量加法操作符的优先级（与加法相同）
    "TensMul": PRECEDENCE["Mul"],            # 张量乘法操作符的优先级（与乘法相同）
    "HadamardProduct": PRECEDENCE["Mul"],    # 哈达玛积操作符的优先级（与乘法相同）
    "HadamardPower": PRECEDENCE["Pow"],      # 哈达玛幂操作符的优先级（与幂相同）
    "KroneckerProduct": PRECEDENCE["Mul"],   # 克罗内克积操作符的优先级（与乘法相同）
    "Equality": PRECEDENCE["Mul"],           # 相等操作符的优先级（与乘法相同）
    "Unequality": PRECEDENCE["Mul"],         # 不相等操作符的优先级（与乘法相同）
}

# 有时仅仅为一个类别分配固定的优先级值是不够的。这时可以在该字典中插入一个函数，
# 函数接受该类别的实例作为参数，并返回适当的优先级值。

# 优先级函数

def precedence_Mul(item):
    if item.could_extract_minus_sign():
        return PRECEDENCE["Add"]  # 如果可以提取负号，则返回加法操作符的优先级
    return PRECEDENCE["Mul"]      # 否则返回乘法操作符的优先级

def precedence_Rational(item):
    if item.p < 0:
        return PRECEDENCE["Add"]  # 如果分数为负数，则返回加法操作符的优先级
    return PRECEDENCE["Mul"]      # 否则返回乘法操作符的优先级

def precedence_Integer(item):
    if item.p < 0:
        return PRECEDENCE["Add"]  # 如果整数为负数，则返回加法操作符的优先级
    return PRECEDENCE["Atom"]     # 否则返回原子操作符的优先级

def precedence_Float(item):
    if item < 0:
        return PRECEDENCE["Add"]  # 如果浮点数为负数，则返回加法操作符的优先级
    return PRECEDENCE["Atom"]     # 否则返回原子操作符的优先级

def precedence_PolyElement(item):
    if item.is_generator:
        return PRECEDENCE["Atom"] # 如果是生成器，则返回原子操作符的优先级
    elif item.is_ground:
        return precedence(item.coeff(1))  # 如果是常数项，则返回常数项的优先级
    elif item.is_term:
        return PRECEDENCE["Mul"]   # 如果是项，则返回乘法操作符的优先级
    else:
        return PRECEDENCE["Add"]   # 否则返回加法操作符的优先级

def precedence_FracElement(item):
    if item.denom == 1:
        return precedence_PolyElement(item.numer)  # 如果分数的分母为1，则返回分子的优先级
    else:
        return PRECEDENCE["Mul"]   # 否则返回乘法操作符的优先级

def precedence_UnevaluatedExpr(item):
    return precedence(item.args[0]) - 0.5  # 返回未评估表达式的优先级减去0.5

PRECEDENCE_FUNCTIONS = {
    "Integer": precedence_Integer,
    "Mul": precedence_Mul,
    "Rational": precedence_Rational,
    "Float": precedence_Float,
    "PolyElement": precedence_PolyElement,
    "FracElement": precedence_FracElement,
    "UnevaluatedExpr": precedence_UnevaluatedExpr,
}
}

def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    # 如果对象具有 precedence 属性，则返回其定义的优先级
    if hasattr(item, "precedence"):
        return item.precedence
    # 如果对象不是类型，遍历其类型的方法解析顺序
    if not isinstance(item, type):
        for i in type(item).mro():
            n = i.__name__
            # 如果类型名在 PRECEDENCE_FUNCTIONS 中，返回其函数计算结果
            if n in PRECEDENCE_FUNCTIONS:
                return PRECEDENCE_FUNCTIONS[n](item)
            # 如果类型名在 PRECEDENCE_VALUES 中，返回其预设的优先级值
            elif n in PRECEDENCE_VALUES:
                return PRECEDENCE_VALUES[n]
    # 如果未匹配到特定类型，默认返回 Atom 的优先级
    return PRECEDENCE["Atom"]

# 复制 PRECEDENCE 字典并进行部分修改，形成 TRADITIONAL 风格的优先级定义
PRECEDENCE_TRADITIONAL = PRECEDENCE.copy()
PRECEDENCE_TRADITIONAL['Integral'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Sum'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Product'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Limit'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Derivative'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['TensorProduct'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Transpose'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Adjoint'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Dot'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Cross'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Gradient'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Divergence'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Curl'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Laplacian'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Union'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Intersection'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Complement'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['SymmetricDifference'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['ProductSet'] = PRECEDENCE['Xor']

def precedence_traditional(item):
    """Returns the precedence of a given object according to the
    traditional rules of mathematics.

    This is the precedence for the LaTeX and pretty printer.
    """
    # 如果对象是 UnevaluatedExpr 类型，则递归计算其参数的优先级
    from sympy.core.expr import UnevaluatedExpr
    if isinstance(item, UnevaluatedExpr):
        return precedence_traditional(item.args[0])

    n = item.__class__.__name__
    # 如果对象类名在 PRECEDENCE_TRADITIONAL 中，返回相应的优先级
    if n in PRECEDENCE_TRADITIONAL:
        return PRECEDENCE_TRADITIONAL[n]

    # 否则返回基本 precedence 函数计算的优先级
    return precedence(item)
```