# `D:\src\scipysrc\pandas\pandas\_libs\ops_dispatch.pyx`

```
DISPATCHED_UFUNCS = {
    "add",           # 被分派到对应 dunder 方法的二元 ufuncs 集合，如加法
    "sub",           # 减法
    "mul",           # 乘法
    "pow",           # 指数运算
    "mod",           # 取模
    "floordiv",      # 地板除法
    "truediv",       # 真除法
    "divmod",        # 返回除法的商和余数
    "eq",            # 等于比较
    "ne",            # 不等于比较
    "lt",            # 小于比较
    "gt",            # 大于比较
    "le",            # 小于等于比较
    "ge",            # 大于等于比较
    "remainder",     # 求余
    "matmul",        # 矩阵乘法
    "or",            # 按位或
    "xor",           # 按位异或
    "and",           # 按位与
    "neg",           # 负数运算
    "pos",           # 正数运算
    "abs",           # 绝对值
}
UNARY_UFUNCS = {
    "neg",           # 一元 ufuncs 集合，如负数运算
    "pos",           # 正数运算
    "abs",           # 绝对值
}
UFUNC_ALIASES = {
    "subtract": "sub",         # 别名映射，例如 subtract 到 sub
    "multiply": "mul",         # multiply 到 mul
    "floor_divide": "floordiv",# floor_divide 到 floordiv
    "true_divide": "truediv",  # true_divide 到 truediv
    "power": "pow",            # power 到 pow
    "remainder": "mod",        # remainder 到 mod
    "divide": "truediv",       # divide 到 truediv
    "equal": "eq",             # equal 到 eq
    "not_equal": "ne",         # not_equal 到 ne
    "less": "lt",              # less 到 lt
    "less_equal": "le",        # less_equal 到 le
    "greater": "gt",           # greater 到 gt
    "greater_equal": "ge",     # greater_equal 到 ge
    "bitwise_or": "or",        # bitwise_or 到 or
    "bitwise_and": "and",      # bitwise_and 到 and
    "bitwise_xor": "xor",      # bitwise_xor 到 xor
    "negative": "neg",         # negative 到 neg
    "absolute": "abs",         # absolute 到 abs
    "positive": "pos",         # positive 到 pos
}

# For op(., Array) -> Array.__r{op}__
REVERSED_NAMES = {
    "lt": "__gt__",            # 小于变成大于的反向方法名
    "le": "__ge__",            # 小于等于变成大于等于的反向方法名
    "gt": "__lt__",            # 大于变成小于的反向方法名
    "ge": "__le__",            # 大于等于变成小于等于的反向方法名
    "eq": "__eq__",            # 等于变成等于的反向方法名
    "ne": "__ne__",            # 不等于变成不等于的反向方法名
}


def maybe_dispatch_ufunc_to_dunder_op(
    object self, object ufunc, str method, *inputs, **kwargs
):
    """
    Dispatch a ufunc to the equivalent dunder method.

    Parameters
    ----------
    self : ArrayLike
        The array whose dunder method we dispatch to
    ufunc : Callable
        A NumPy ufunc
    method : {'reduce', 'accumulate', 'reduceat', 'outer', 'at', '__call__'}
        The method of ufunc invocation
    inputs : ArrayLike
        The input arrays.
    kwargs : Any
        The additional keyword arguments, e.g. ``out``.

    Returns
    -------
    result : Any
        The result of applying the ufunc
    """
    # 检查被调用的 ufunc 的名称，并根据别名映射进行规范化
    op_name = ufunc.__name__
    op_name = UFUNC_ALIASES.get(op_name, op_name)

    def not_implemented(*args, **kwargs):
        return NotImplemented

    # 如果有关键字参数或者 ufunc 的输入数量大于两个，则返回 Not Implemented
    if kwargs or ufunc.nin > 2:
        return NotImplemented

    # 如果方法为 "__call__" 并且 op_name 在需要分派的 ufuncs 集合中
    if method == "__call__" and op_name in DISPATCHED_UFUNCS:

        # 如果 inputs 的第一个元素是 self，表示操作在左侧
        if inputs[0] is self:
            name = f"__{op_name}__"
            meth = getattr(self, name, not_implemented)

            # 如果是一元操作，则只有一个输入
            if op_name in UNARY_UFUNCS:
                assert len(inputs) == 1
                return meth()

            # 否则，调用对应的 dunder 方法并传入第二个输入
            return meth(inputs[1])

        # 如果 inputs 的第二个元素是 self，表示操作在右侧
        elif inputs[1] is self:
            name = REVERSED_NAMES.get(op_name, f"__r{op_name}__")

            # 获取对应的反向 dunder 方法并调用
            meth = getattr(self, name, not_implemented)
            result = meth(inputs[0])
            return result

        else:
            # 应该不会到达这里，为了完备性进行覆盖
            return NotImplemented

    else:
        return NotImplemented
```