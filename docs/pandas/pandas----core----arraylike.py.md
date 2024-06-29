# `D:\src\scipysrc\pandas\pandas\core\arraylike.py`

```
    # 算术方法，处理加法操作
    def __add__(self, other):
        return self._arith_method(other, operator.add)

    # 算术方法，处理减法操作
    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    # 算术方法，处理乘法操作
    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    # 算术方法，处理真除法操作
    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    # 算术方法，处理地板除法操作
    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    # 算术方法，处理取模操作
    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    # 算术方法，处理幂次方操作
    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    # 算术方法，处理左位移操作
    @unpack_zerodim_and_defer("__lshift__")
    def __lshift__(self, other):
        return self._arith_method(other, operator.lshift)

    # 算术方法，处理右位移操作
    @unpack_zerodim_and_defer("__rshift__")
    def __rshift__(self, other):
        return self._arith_method(other, operator.rshift)
    def __add__(self, other):
        """
        获取 DataFrame 和另一个对象按列相加的结果。

        相当于 ``DataFrame.add(other)``。

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            要添加到 DataFrame 的对象。

        Returns
        -------
        DataFrame
            将 ``other`` 添加到 DataFrame 后的结果。

        See Also
        --------
        DataFrame.add : 添加 DataFrame 和另一个对象，可选择按索引或列进行添加。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"height": [1.5, 2.6], "weight": [500, 800]}, index=["elk", "moose"]
        ... )
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        添加标量会影响所有行和列。

        >>> df[["height", "weight"]] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        将列表的每个元素按顺序添加到 DataFrame 的列中。

        >>> df[["height", "weight"]] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        字典的键根据列名与 DataFrame 对齐；字典中的每个值都会添加到相应的列中。

        >>> df[["height", "weight"]] + {"height": 0.5, "weight": 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        当 `other` 是 :class:`Series` 时，其索引与 DataFrame 的列对齐。

        >>> s1 = pd.Series([0.5, 1.5], index=["weight", "height"])
        >>> df[["height", "weight"]] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        即使 `other` 的索引与 DataFrame 的索引相同，:class:`Series` 也不会被重新定位。
        如果需要按索引对齐，应该使用 :meth:`DataFrame.add` 并指定 `axis='index'`。

        >>> s2 = pd.Series([0.5, 1.5], index=["elk", "moose"])
        >>> df[["height", "weight"]] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[["height", "weight"]].add(s2, axis="index")
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        当 `other` 是 :class:`DataFrame` 时，列名和索引都会进行对齐。

        >>> other = pd.DataFrame(
        ...     {"height": [0.2, 0.4, 0.6]}, index=["elk", "moose", "deer"]
        ... )
        >>> df[["height", "weight"]] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)
    # 定义一个装饰器函数，用于解包零维数据并延迟执行指定操作符的方法
    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右加）
        return self._arith_method(other, roperator.radd)
    
    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为减法）
        return self._arith_method(other, operator.sub)
    
    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右减）
        return self._arith_method(other, roperator.rsub)
    
    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为乘法）
        return self._arith_method(other, operator.mul)
    
    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右乘）
        return self._arith_method(other, roperator.rmul)
    
    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为真除法）
        return self._arith_method(other, operator.truediv)
    
    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右真除法）
        return self._arith_method(other, roperator.rtruediv)
    
    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为整除）
        return self._arith_method(other, operator.floordiv)
    
    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右整除）
        return self._arith_method(other, roperator.rfloordiv)
    
    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为取模）
        return self._arith_method(other, operator.mod)
    
    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右取模）
        return self._arith_method(other, roperator.rmod)
    
    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为取商和余数）
        return self._arith_method(other, divmod)
    
    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右取商和余数）
        return self._arith_method(other, roperator.rdivmod)
    
    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为乘方）
        return self._arith_method(other, operator.pow)
    
    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other):
        # 调用内部的 _arith_method 方法，执行特定的算术操作（此处为右乘方）
        return self._arith_method(other, roperator.rpow)
# -----------------------------------------------------------------------------
# Helpers to implement __array_ufunc__


def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    # Import necessary modules and classes from pandas
    from pandas.core.frame import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.internals import BlockManager

    # Get the class type of the current instance
    cls = type(self)

    # Standardize the keyword arguments for output
    kwargs = _standardize_out_kwarg(**kwargs)

    # for binary ops, use our custom dunder methods
    # Attempt to dispatch the ufunc operation to custom dunder methods
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # Determine if we should defer.
    no_defer = (
        np.ndarray.__array_ufunc__,
        cls.__array_ufunc__,
    )

    # Check each input for conditions that would prevent deferring to __array_ufunc__
    for item in inputs:
        # Check if the input has higher priority than self
        higher_priority = (
            hasattr(item, "__array_priority__")
            and item.__array_priority__ > self.__array_priority__
        )
        # Check if the input has its own __array_ufunc__ method and can't defer
        has_array_ufunc = (
            hasattr(item, "__array_ufunc__")
            and type(item).__array_ufunc__ not in no_defer
            and not isinstance(item, self._HANDLED_TYPES)
        )
        # If either condition is met, return NotImplemented
        if higher_priority or has_array_ufunc:
            return NotImplemented

    # align all the inputs.
    types = tuple(type(x) for x in inputs)
    # Collect inputs that are subclasses of NDFrame for alignment
    alignable = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]

    # If there are multiple alignable inputs, trigger alignment
    if len(alignable) > 1:
        # Handle special cases where ufunc operates on mixed DataFrame and Series
        set_types = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            raise NotImplementedError(
                f"Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs."
            )
        # Align axes of all alignable inputs
        axes = self.axes
        for obj in alignable[1:]:
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)

        # Reconstruct axes after alignment
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes))
        # Reindex inputs based on reconstructed axes if they are NDFrame instances
        inputs = tuple(
            x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x
            for x, t in zip(inputs, types)
        )
    else:
        # If only one alignable input, reconstruct axes from self
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))

    # If self is 1-dimensional, determine the name of the resulting object
    if self.ndim == 1:
        names = {getattr(x, "name") for x in inputs if hasattr(x, "name")}
        name = names.pop() if len(names) == 1 else None
        reconstruct_kwargs = {"name": name}
    else:
        # 如果没有设置 `reconstruct_kwargs`，则创建一个空字典
        reconstruct_kwargs = {}

    def reconstruct(result):
        # 如果 ufunc 的输出大于 1，例如 np.modf, np.frexp, np.divmod
        # 返回一个包含每个元素 `_reconstruct` 处理结果的元组
        if ufunc.nout > 1:
            return tuple(_reconstruct(x) for x in result)

        # 否则，直接返回 `_reconstruct` 处理的结果
        return _reconstruct(result)

    def _reconstruct(result):
        # 如果 `result` 是标量，则直接返回
        if lib.is_scalar(result):
            return result

        # 如果 `result` 的维度与 `self` 的维度不同
        if result.ndim != self.ndim:
            # 如果 `method` 是 "outer"，则抛出 NotImplementedError
            if method == "outer":
                raise NotImplementedError
            # 否则直接返回 `result`
            return result

        # 如果 `result` 是 BlockManager 类型
        if isinstance(result, BlockManager):
            # 通过 `_constructor_from_mgr` 方法使用 `result` 重新构造对象
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            # 否则，通过 `_constructor` 方法使用 `result` 和重建的轴参数构造对象
            result = self._constructor(
                result, **reconstruct_axes, **reconstruct_kwargs, copy=False
            )

        # TODO: 当我们支持多个值在 `__finalize__` 方法中时，
        # 应该传递 `alignable` 到 `__finalize__` 方法而不是 `self`。
        # 这样，当 `np.add(a, b)` 中 `a` 和 `b` 是 NDFrames 时，会考虑到两者的属性。
        if len(alignable) == 1:
            # 如果 `alignable` 的长度为 1，将 `self` 的属性传递给 `result`
            result = result.__finalize__(self)

        # 返回处理后的 `result`
        return result

    if "out" in kwargs:
        # 如果 `kwargs` 中包含 "out" 参数
        # 使用 `dispatch_ufunc_with_out` 方法调度 ufunc 操作，并返回结果
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        # 对结果进行重建处理，并返回
        return reconstruct(result)

    if method == "reduce":
        # 如果 `method` 是 "reduce"
        # 使用 `dispatch_reduction_ufunc` 方法调度 ufunc 函数，并返回结果
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        # 如果返回的结果不是 NotImplemented，则返回结果
        if result is not NotImplemented:
            return result

    # 当 `self` 的维度大于 1 并且输入数量大于 1 或者 ufunc 输出大于 1 时
    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        # 仅在复杂情况下放弃保持类型
        # 理论上，我们可以保持它们。
        # * 如果 nout>1，则当 BlockManager.apply 接受 nout 并返回 Tuple[BlockManager] 时可行。
        # * 如果 len(inputs) > 1，则当我们知道已经对齐块 / 数据类型时可行。

        # 将每个输入数组转换为 numpy 数组
        inputs = tuple(np.asarray(x) for x in inputs)
        # 注意：这里不能使用 default_array_ufunc，因为重新索引意味着 `self` 可能不在 `inputs` 中
        # 使用 ufunc 对象的 `method` 方法对输入进行操作，并传递其他参数
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        # 如果 `self` 的维度是 1
        # 对每个输入提取其数组部分，并使用 `ufunc` 对象的 `method` 方法对输入进行操作
        inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    else:
        # 如果不是第一个分支，执行以下代码
        # ufunc(dataframe)
        # 如果方法是 "__call__" 并且没有关键字参数 kwargs
        if method == "__call__" and not kwargs:
            # 对于 np.<ufunc>(..) 调用
            # 由于关键字参数不能逐块处理，因此只在没有 kwargs 的情况下采用这条路径
            # 获取第一个输入的数据管理器
            mgr = inputs[0]._mgr
            # 应用 ufunc 对象的 "__call__" 方法到数据管理器上
            result = mgr.apply(getattr(ufunc, method))
        else:
            # 否则针对具体的 ufunc 方法（例如 np.<ufunc>.accumulate(..)）
            # 这些方法可能有 axis 关键字参数，因此不能逐块调用
            # 调用默认的数组 ufunc 方法处理函数
            result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)
            # 例如 np.negative（只有一个被调用），kwargs 中可能包含 "where" 和 "out"

    # 对结果进行重构处理
    result = reconstruct(result)
    # 返回处理后的结果
    return result
# 标准化输出关键字参数，将 "out1" 和 "out2" 替换为元组 "out"
def _standardize_out_kwarg(**kwargs) -> dict:
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    # 如果参数中没有 "out"，但有 "out1" 和 "out2"
    if "out" not in kwargs and "out1" in kwargs and "out2" in kwargs:
        # 弹出并获取 "out1" 和 "out2" 的值
        out1 = kwargs.pop("out1")
        out2 = kwargs.pop("out2")
        # 组成元组 "out"
        out = (out1, out2)
        # 将 "out" 添加回关键字参数中
        kwargs["out"] = out
    return kwargs


# 调度带有 "out" 关键字的通用函数调用
def dispatch_ufunc_with_out(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """
    # 注意：假设 _standardize_out_kwarg 已经被调用过

    # 弹出 "out" 和 "where" 关键字参数
    out = kwargs.pop("out")
    where = kwargs.pop("where", None)

    # 调用 ufunc 的指定方法，传入输入和处理后的关键字参数
    result = getattr(ufunc, method)(*inputs, **kwargs)

    # 如果结果为 NotImplemented，则返回 NotImplemented
    if result is NotImplemented:
        return NotImplemented

    # 如果结果是一个元组
    if isinstance(result, tuple):
        # 例如 np.divmod, np.modf, np.frexp
        if not isinstance(out, tuple) or len(out) != len(result):
            # 如果 "out" 不是元组或长度与结果不匹配，则抛出 NotImplementedError
            raise NotImplementedError

        # 遍历 "out" 和 "result"，将结果赋值给对应的数组
        for arr, res in zip(out, result):
            _assign_where(arr, res, where)

        # 返回 "out" 元组
        return out

    # 如果 "out" 是一个元组
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            # 如果 "out" 的长度不为 1，则抛出 NotImplementedError
            raise NotImplementedError

    # 将结果赋值给 "out"，考虑可能的 "where" 参数
    _assign_where(out, result, where)
    return out


# 将 ufunc 的结果设置到 "out" 中，可能使用 "where" 参数进行掩码
def _assign_where(out, result, where) -> None:
    """
    Set a ufunc result into 'out', masking with a 'where' argument if necessary.
    """
    if where is None:
        # 如果没有传递 "where" 参数给 ufunc，则直接将结果赋值给 "out"
        out[:] = result
    else:
        # 否则，根据 "where" 参数进行掩码赋值
        np.putmask(out, where, result)


# 默认的数组通用函数行为，若未定义 __array_ufunc__ 则使用该行为
def default_array_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Fallback to the behavior we would get if we did not define __array_ufunc__.

    Notes
    -----
    We are assuming that `self` is among `inputs`.
    """
    # 如果 `self` 不在输入中，则抛出 NotImplementedError
    if not any(x is self for x in inputs):
        raise NotImplementedError

    # 将 `self` 替换为其对应的数组表示
    new_inputs = [x if x is not self else np.asarray(x) for x in inputs]

    # 调用 ufunc 的指定方法，传入新的输入和给定的关键字参数
    return getattr(ufunc, method)(*new_inputs, **kwargs)


# 将 ufunc 的归约操作分派到 self 的归约方法中
def dispatch_reduction_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    # 确保方法为 "reduce"
    assert method == "reduce"

    # 如果输入数量不为 1，或者第一个输入不是 self，则返回 NotImplemented
    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented

    # 如果 ufunc 的名称不在 REDUCTION_ALIASES 中，则返回 NotImplemented
    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented

    # 获取对应的方法名称
    method_name = REDUCTION_ALIASES[ufunc.__name__]

    # 注意：假设 min/max 表示最小/最大方法，这对于例如 Timestamp.min 将不正确
    # 如果 self 没有定义相应的方法，则返回 NotImplemented
    if not hasattr(self, method_name):
        return NotImplemented
    # 如果数组的维度大于1
    if self.ndim > 1:
        # 如果self是ABCNDFrame的实例
        if isinstance(self, ABCNDFrame):
            # TODO: 测试这种情况是否成立，例如2D的DTA/TDA
            # 设置关键字参数"numeric_only"为False
            kwargs["numeric_only"] = False

        # 如果关键字参数中没有包含"axis"
        if "axis" not in kwargs:
            # 对于DataFrame的减少操作，我们不希望默认axis=0
            # 注意：np.min不是一个ufunc，而是使用array_function_dispatch，
            # 因此调用DataFrame.min（而不会到达这里），np.min的默认axis=None，
            # DataFrame.min捕获并将其更改为axis=0。
            # np.minimum.reduce(df)会到达这里，因为kwargs中没有axis，
            # 因此我们设置axis=0以匹配np.minimum.reduce(df.values)的行为。
            kwargs["axis"] = 0

    # 默认情况下，numpy的减少操作不会跳过NaN值，因此我们必须传递skipna=False
    # 调用self对象的method_name方法，传递skipna=False以及其他关键字参数kwargs
    return getattr(self, method_name)(skipna=False, **kwargs)
```