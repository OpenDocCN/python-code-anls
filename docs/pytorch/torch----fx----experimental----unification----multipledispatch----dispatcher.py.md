# `.\pytorch\torch\fx\experimental\unification\multipledispatch\dispatcher.py`

```
# mypy: allow-untyped-defs
# 导入警告模块中的warn函数
from warnings import warn
# 导入inspect模块，用于获取对象信息
import inspect
# 导入typing_extensions中的deprecated装饰器
from typing_extensions import deprecated
# 导入当前包中的conflict模块中的若干符号
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
# 导入当前包中的utils模块中的expand_tuples函数
from .utils import expand_tuples
# 导入当前包中的variadic模块中的Variadic类和isvariadic函数
from .variadic import Variadic, isvariadic
# 导入标准库中的itertools模块并命名为itl
import itertools as itl

# 定义公开的接口列表
__all__ = ["MDNotImplementedError", "ambiguity_warn", "halt_ordering", "restart_ordering", "variadic_signature_matches_iter",
           "variadic_signature_matches", "Dispatcher", "source", "MethodDispatcher", "str_signature", "warning_text"]

# 自定义的NotImplementedError类，用于多重分派中的未实现错误
class MDNotImplementedError(NotImplementedError):
    """ A NotImplementedError for multiple dispatch """


# 当检测到模糊性时发出警告的函数
def ambiguity_warn(dispatcher, ambiguities):
    """ Raise warning when ambiguity is detected
    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher
    See Also:
        Dispatcher.add
        warning_text
    """
    # 调用warn函数发出模糊性警告，警告文本由warning_text函数生成
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


# 被弃用的函数，用于临时禁用排序
@deprecated(
    "`halt_ordering` is deprecated, you can safely remove this call.",
    category=FutureWarning,
)
def halt_ordering():
    """Deprecated interface to temporarily disable ordering."""


# 被弃用的函数，用于临时恢复排序
@deprecated(
    "`restart_ordering` is deprecated, if you would like to eagerly order the dispatchers, "
    "you should call the `reorder()` method on each dispatcher.",
    category=FutureWarning,
)
def restart_ordering(on_ambiguity=ambiguity_warn):
    """Deprecated interface to temporarily resume ordering."""


# 迭代器版本的函数，检查一组输入类型是否匹配可变签名
def variadic_signature_matches_iter(types, full_signature):
    """Check if a set of input types matches a variadic signature.
    Notes
    -----
    The algorithm is as follows:
    Initialize the current signature to the first in the sequence
    For each type in `types`:
        If the current signature is variadic
            If the type matches the signature
                yield True
            Else
                Try to get the next signature
                If no signatures are left we can't possibly have a match
                    so yield False
        Else
            yield True if the type matches the current signature
            Get the next signature
    """
    # 使用迭代器遍历完整签名序列
    sigiter = iter(full_signature)
    # 初始化当前签名为序列中的第一个签名
    sig = next(sigiter)
    for typ in types:
        # 检查类型typ是否是当前签名sig的子类
        matches = issubclass(typ, sig)
        # 生成匹配结果
        yield matches
        # 如果当前签名不是可变参数签名，则移动到下一个签名
        if not isvariadic(sig):
            sig = next(sigiter)
    else:
        try:
            sig = next(sigiter)
        except StopIteration:
            # 如果没有剩余的签名项，且当前签名是可变参数签名，则生成True
            assert isvariadic(sig)
            yield True
        else:
            # 如果还有剩余的签名项，则生成False，表示所有参数都未匹配
            yield False


# 检查一组输入类型是否匹配可变签名的函数
def variadic_signature_matches(types, full_signature):
    # 断言全签名参数非空，确保全签名有内容传入
    assert full_signature
    # 返回一个布尔值，表示是否所有的可变签名都匹配给定的类型和完整签名
    return all(variadic_signature_matches_iter(types, full_signature))
class Dispatcher:
    """ Dispatch methods based on type signature
    Use ``dispatch`` to add implementations
    Examples
    --------
    >>> # xdoctest: +SKIP("bad import name")
    >>> from multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1
    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1
    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """

    __slots__ = '__name__', 'name', 'funcs', '_ordering', '_cache', 'doc'

    def __init__(self, name, doc=None):
        # 设置实例变量名称和文档字符串
        self.name = self.__name__ = name
        # 初始化一个空的函数字典
        self.funcs = {}
        # 设置实例的文档字符串
        self.doc = doc
        # 初始化缓存字典
        self._cache = {}

    def register(self, *types, **kwargs):
        """ register dispatcher with new implementation
        >>> # xdoctest: +SKIP
        >>> f = Dispatcher('f')
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1
        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1
        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]
        >>> f(1)
        2
        >>> f(1.0)
        0.0
        >>> f([1, 2, 3])
        [3, 2, 1]
        """
        def _df(func):
            # 调用父类方法 add 来添加类型及其对应的函数实现
            self.add(types, func, **kwargs)   # type: ignore[call-arg]
            return func
        return _df

    @classmethod
    def get_func_params(cls, func):
        # 获取函数的参数列表
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """ get annotations of function positional parameters
        """
        # 获取函数位置参数的注解信息
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            # 仅选择位置参数或者位置或关键字参数
            params = (param for param in params
                      if param.kind in
                      (Parameter.POSITIONAL_ONLY,
                       Parameter.POSITIONAL_OR_KEYWORD))

            # 收集参数的注解
            annotations = tuple(
                param.annotation
                for param in params)

            # 如果所有的注解都不是空的 Parameter.empty，则返回注解元组
            if all(ann is not Parameter.empty for ann in annotations):
                return annotations
    def add(self, signature, func):
        """ Add new types/method pair to dispatcher

        >>> # xdoctest: +SKIP
        >>> D = Dispatcher('add')
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)
        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>
        >>> # When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        >>> # with a dispatcher/itself, and a set of ambiguous type signature pairs
        >>> # as inputs.  See ``ambiguity_warn`` for an example.
        """

        # Handle annotations
        # 如果没有显式提供签名，则尝试从函数注解中获取
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        # 处理联合类型的签名，展开元组并递归调用add方法
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func)
            return

        new_signature = []

        # Iterate through each type in the signature
        # 遍历签名中的每一个类型
        for index, typ in enumerate(signature, start=1):
            if not isinstance(typ, (type, list)):
                # 如果类型不是类或列表，则引发TypeError
                str_sig = ', '.join(c.__name__ if isinstance(c, type)
                                    else str(c) for c in signature)
                raise TypeError(f"Tried to dispatch on non-type: {typ}\n"
                                f"In signature: <{str_sig}>\n"
                                f"In function: {self.name}")

            # Handle variadic signatures
            # 处理可变长度签名
            if isinstance(typ, list):
                if index != len(signature):
                    raise TypeError(
                        'Variadic signature must be the last element'
                    )

                if len(typ) != 1:
                    raise TypeError(
                        'Variadic signature must contain exactly one element. '
                        'To use a variadic union type place the desired types '
                        'inside of a tuple, e.g., [(int, str)]'
                    )
                new_signature.append(Variadic[typ[0]])
            else:
                new_signature.append(typ)

        # Store the function under the tuple of normalized signature types
        # 将函数存储在标准化签名类型的元组下
        self.funcs[tuple(new_signature)] = func

        # Clear cache to ensure accurate function dispatch
        # 清除缓存，以确保函数分发的准确性
        self._cache.clear()

        try:
            # Attempt to delete ordering attribute to trigger recalculation
            # 尝试删除ordering属性以触发重新计算
            del self._ordering
        except AttributeError:
            pass

    @property
    def ordering(self):
        # Property method to get ordering; calculates if not already present
        # 获取排序属性的方法；如果不存在则计算
        try:
            return self._ordering
        except AttributeError:
            return self.reorder()

    def reorder(self, on_ambiguity=ambiguity_warn):
        # Reorder the functions based on precedence rules
        # 根据优先级规则重新排序函数

        # Calculate ordering dictionary
        # 计算排序字典
        self._ordering = od = ordering(self.funcs)

        # Detect and handle ambiguities if present
        # 如果存在歧义，则检测并处理
        amb = ambiguities(self.funcs)
        if amb:
            on_ambiguity(self, amb)

        return od
    def __call__(self, *args, **kwargs):
        # 确定参数类型的元组
        types = tuple([type(arg) for arg in args])
        try:
            # 尝试从缓存中获取相应类型的函数
            func = self._cache[types]
        except KeyError as e:
            # 若缓存中不存在，则通过 dispatch 方法确定适当的实现
            func = self.dispatch(*types)
            if not func:
                # 如果找不到对应的函数签名，则抛出 NotImplementedError
                raise NotImplementedError(
                    f'Could not find signature for {self.name}: <{str_signature(types)}>') from e
            # 将找到的函数缓存起来
            self._cache[types] = func
        try:
            # 调用找到的函数并返回结果
            return func(*args, **kwargs)

        except MDNotImplementedError as e:
            # 如果函数抛出 MDNotImplementedError，则继续尝试下一个适合的函数
            funcs = self.dispatch_iter(*types)
            next(funcs)  # 跳过第一个
            for func in funcs:
                try:
                    # 调用下一个函数并返回结果
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass

            # 若所有函数均未成功执行，则抛出 NotImplementedError
            raise NotImplementedError(
                "Matching functions for "
                f"{self.name}: <{str_signature(types)}> found, but none completed successfully",) from e

    def __str__(self):
        # 返回对象的字符串表示形式
        return f"<dispatched {self.name}>"
    __repr__ = __str__

    def dispatch(self, *types):
        """Determine appropriate implementation for this type signature
        This method is internal.  Users should call this object as a function.
        >>> # xdoctest: +SKIP
        >>> from multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1
        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4
        >>> print(inc.dispatch(float))
        None
        See Also:
          ``multipledispatch.conflict`` - module to determine resolution order
        """
        
        # 检查是否存在与给定类型签名相匹配的函数
        if types in self.funcs:
            return self.funcs[types]

        try:
            # 返回第一个匹配的函数
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(self, *types):
        # 迭代器，返回所有与给定类型签名相匹配的函数

        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    yield result

    @deprecated("`resolve()` is deprecated, use `dispatch(*types)`", category=FutureWarning)
    def resolve(self, types):
        """ Determine appropriate implementation for this type signature
        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        # 确定给定类型签名的适当实现（已弃用）
        return self.dispatch(*types)

    def __getstate__(self):
        # 获取对象的序列化状态
        return {'name': self.name,
                'funcs': self.funcs}

    def __setstate__(self, d):
        # 设置对象的反序列化状态
        self.name = d['name']
        self.funcs = d['funcs']
        self._ordering = ordering(self.funcs)
        self._cache = {}

    @property
    # 属性装饰器
    def __doc__(self):
        # 创建文档字符串列表，起始为多路分派方法的说明
        docs = [f"Multiply dispatched method: {self.name}"]

        # 如果存在方法的文档字符串，添加到文档列表中
        if self.doc:
            docs.append(self.doc)

        # 初始化另一个空列表，用于存储未包含文档字符串的签名
        other = []

        # 倒序遍历排序后的方法签名列表
        for sig in self.ordering[::-1]:
            # 获取当前签名对应的函数
            func = self.funcs[sig]

            # 如果函数有文档字符串，格式化并添加到文档列表中
            if func.__doc__:
                s = f'Inputs: <{str_signature(sig)}>\n'
                s += '-' * len(s) + '\n'
                s += func.__doc__.strip()
                docs.append(s)
            else:
                # 否则将签名格式化后添加到另一个列表中
                other.append(str_signature(sig))

        # 如果存在未添加到文档列表的签名，添加到文档列表的末尾
        if other:
            docs.append('Other signatures:\n    ' + '\n    '.join(other))

        # 返回最终的文档字符串，每个部分用两个换行符分隔
        return '\n\n'.join(docs)

    def _help(self, *args):
        # 调用多路分派方法，根据参数类型获取对应函数的文档字符串
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args, **kwargs):
        """ 打印与输入对应函数的文档字符串 """
        # 调用 _help 方法并打印结果
        print(self._help(*args))

    def _source(self, *args):
        # 调用多路分派方法，根据参数类型获取对应函数的源代码
        func = self.dispatch(*map(type, args))
        
        # 如果未找到对应的函数，抛出类型错误异常
        if not func:
            raise TypeError("No function found")
        
        # 返回获取到的函数的源代码
        return source(func)

    def source(self, *args, **kwargs):
        """ 打印与输入对应函数的源代码 """
        # 调用 _source 方法并打印结果
        print(self._source(*args))
# 定义一个函数，返回指定函数的源文件路径和源代码的字符串
def source(func):
    s = f'File: {inspect.getsourcefile(func)}\n\n'
    s = s + inspect.getsource(func)
    return s


# 定义一个类 MethodDispatcher，继承自 Dispatcher 类
class MethodDispatcher(Dispatcher):
    """ Dispatch methods based on type signature
    See Also:
        Dispatcher
    """
    # 限定类的属性只能有 'obj' 和 'cls'
    __slots__ = ('obj', 'cls')

    @classmethod
    # 类方法，获取函数的参数列表
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    # 实例方法，将实例对象和类对象绑定到当前实例
    def __get__(self, instance, owner):
        self.obj = instance
        self.cls = owner
        return self

    # 实例可调用对象，根据参数类型调度对应的方法执行
    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        func = self.dispatch(*types)
        if not func:
            # 如果找不到匹配的方法签名，抛出未实现的异常
            raise NotImplementedError(f'Could not find signature for {self.name}: <{str_signature(types)}>')
        # 调用匹配的方法，并传入参数执行
        return func(self.obj, *args, **kwargs)


# 函数，返回给定类型签名的字符串表示形式
def str_signature(sig):
    """ String representation of type signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ', '.join(cls.__name__ for cls in sig)


# 函数，返回函数调度时的歧义警告文本
def warning_text(name, amb):
    """ The text for ambiguity warnings """
    text = f"\nAmbiguities exist in dispatched function {name}\n\n"
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + \
                ', '.join('[' + str_signature(s) + ']' for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += '\n\n'.join(['@dispatch(' + str_signature(super_signature(s))
                         + f')\ndef {name}(...)' for s in amb])
    return text
```