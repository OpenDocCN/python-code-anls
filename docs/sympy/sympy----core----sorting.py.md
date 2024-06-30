# `D:\src\scipysrc\sympy\sympy\core\sorting.py`

```
# 导入必要的模块
from collections import defaultdict
from .sympify import sympify, SympifyError  # 导入sympify函数和SympifyError异常类
from sympy.utilities.iterables import iterable, uniq  # 导入iterable函数和uniq函数

# 定义公开接口的列表
__all__ = ['default_sort_key', 'ordered']

# 定义默认排序键的函数
def default_sort_key(item, order=None):
    """Return a key that can be used for sorting.

    The key has the structure:

    (class_key, (len(args), args), exponent.sort_key(), coefficient)

    This key is supplied by the sort_key routine of Basic objects when
    ``item`` is a Basic object or an object (other than a string) that
    sympifies to a Basic object. Otherwise, this function produces the
    key.

    The ``order`` argument is passed along to the sort_key routine and is
    used to determine how the terms *within* an expression are ordered.
    (See examples below) ``order`` options are: 'lex', 'grlex', 'grevlex',
    and reversed values of the same (e.g. 'rev-lex'). The default order
    value is None (which translates to 'lex').

    Examples
    ========

    >>> from sympy import S, I, default_sort_key, sin, cos, sqrt
    >>> from sympy.core.function import UndefinedFunction
    >>> from sympy.abc import x

    The following are equivalent ways of getting the key for an object:

    >>> x.sort_key() == default_sort_key(x)
    True

    Here are some examples of the key that is produced:

    >>> default_sort_key(UndefinedFunction('f'))
    ((0, 0, 'UndefinedFunction'), (1, ('f',)), ((1, 0, 'Number'),
        (0, ()), (), 1), 1)
    >>> default_sort_key('1')
    ((0, 0, 'str'), (1, ('1',)), ((1, 0, 'Number'), (0, ()), (), 1), 1)
    >>> default_sort_key(S.One)
    ((1, 0, 'Number'), (0, ()), (), 1)
    >>> default_sort_key(2)
    ((1, 0, 'Number'), (0, ()), (), 2)

    While sort_key is a method only defined for SymPy objects,
    default_sort_key will accept anything as an argument so it is
    more robust as a sorting key. For the following, using key=
    lambda i: i.sort_key() would fail because 2 does not have a sort_key
    method; that's why default_sort_key is used. Note, that it also
    handles sympification of non-string items likes ints:

    >>> a = [2, I, -I]
    >>> sorted(a, key=default_sort_key)
    [2, -I, I]

    The returned key can be used anywhere that a key can be specified for
    a function, e.g. sort, min, max, etc...:

    >>> a.sort(key=default_sort_key); a[0]
    2
    >>> min(a, key=default_sort_key)
    2

    Notes
    =====

    The key returned is useful for getting items into a canonical order
    that will be the same across platforms. It is not directly useful for
    sorting lists of expressions:

    >>> a, b = x, 1/x

    Since ``a`` has only 1 term, its value of sort_key is unaffected by
    ``order``:

    >>> a.sort_key() == a.sort_key('rev-lex')
    True

    If ``a`` and ``b`` are combined then the key will differ because there
    are terms that can be ordered:

    >>> eq = a + b
    >>> eq.sort_key() == eq.sort_key('rev-lex')
    False
    >>> eq.as_ordered_terms()
    [x, 1/x]
    """
    # 返回用于排序的键，根据SymPy对象的特定规则生成
    return (
        item.class_key(),                   # 获取对象的类键
        (len(item.args), item.args),        # 获取对象的参数数量及参数本身
        item.exponent.sort_key(),           # 获取指数的排序键
        item.coefficient                    # 获取系数
    )
    """
    根据默认排序键对给定项进行排序，并返回排序后的结果。

    如果项是基本表达式（Basic），则调用其 sort_key 方法进行排序。

    如果项是可迭代对象但不是字符串，则递归调用 default_sort_key 对每个元素进行排序。
    - 如果是字典，则将其项作为键值对进行排序。
    - 如果是集合，则对集合元素进行排序。
    - 否则，将列表或元组中的每个元素进行排序。

    如果项不是字符串，并且可以通过 sympify 转换为 SymPy 对象，则将其转换为 SymPy 对象，并继续排序。

    如果项是字符串，则将其分类为类索引 0 的元组形式，形式为 (1, (str(item),))。

    返回一个元组，包含以下内容：
    - 类索引（cls_index），这里默认为 0 或 10。
    - 序号（0），用于排序的辅助索引。
    - 项的类名（item.__class__.__name__）。
    - 项的排序键（args）。
    - S.One 的排序键（S.One）。
    """
    from .basic import Basic
    from .singleton import S

    if isinstance(item, Basic):
        # 如果项是 Basic 类型的对象，则调用其 sort_key 方法进行排序
        return item.sort_key(order=order)

    if iterable(item, exclude=str):
        # 如果项是可迭代的对象但不是字符串
        if isinstance(item, dict):
            # 如果是字典，则处理键值对
            args = item.items()
            unordered = True
        elif isinstance(item, set):
            # 如果是集合，则直接取集合元素
            args = item
            unordered = True
        else:
            # 对于元组或列表等，将每个元素递归地进行排序
            args = list(item)
            unordered = False

        # 对每个元素应用 default_sort_key 进行排序
        args = [default_sort_key(arg, order=order) for arg in args]

        if unordered:
            # 如果是无序的容器（如字典或集合），则对 args 进行排序
            args = sorted(args)

        # 设置类索引为 10，表示可迭代对象的排序结果
        cls_index, args = 10, (len(args), tuple(args))
    else:
        if not isinstance(item, str):
            try:
                # 尝试通过 sympify 将 item 转换为 SymPy 对象
                item = sympify(item, strict=True)
            except SympifyError:
                # 如果转换失败，例如 lambda 函数，则直接通过字符串处理
                pass
            else:
                if isinstance(item, Basic):
                    # 如果转换成功且是 Basic 对象，则返回其默认排序键
                    return default_sort_key(item)
                # 对于其他类型的 SymPy 对象，继续处理

        # 如果 item 是字符串，则设置类索引为 0，表示字符串类型
        cls_index, args = 0, (1, (str(item),))

    # 返回一个元组，其中包含类索引、序号、类名、排序键和 S.One 的排序键
    return (cls_index, 0, item.__class__.__name__
            ), args, S.One.sort_key(), S.One
# 计算表达式树中节点的数量
def _node_count(e):
    # 如果表达式 e 是浮点数，则节点数为 0.5
    if e.is_Float:
        return 0.5
    # 否则，节点数为 1 加上所有子节点的节点数之和
    return 1 + sum(map(_node_count, e.args))


def _nodes(e):
    """
    ordered() 的辅助函数，返回表达式 e 的节点数。
    对于 Basic 对象来说，节点数是表达式树中 Basic 节点的数量；
    对于其他对象，节点数通常是 1（除非对象是可迭代的或字典，在这种情况下返回节点数之和）。
    """
    # 导入必要的类
    from .basic import Basic
    from .function import Derivative

    # 如果 e 是 Basic 类型
    if isinstance(e, Basic):
        # 如果 e 是 Derivative 类型，计算其表达式部分的节点数，以及所有变量的节点数之和
        if isinstance(e, Derivative):
            return _nodes(e.expr) + sum(i[1] if i[1].is_Number else
                _nodes(i[1]) for i in e.variable_count)
        # 否则，调用 _node_count 计算节点数
        return _node_count(e)
    # 如果 e 是可迭代的对象
    elif iterable(e):
        # 返回 1 加上所有子元素的节点数之和
        return 1 + sum(_nodes(ei) for ei in e)
    # 如果 e 是字典类型
    elif isinstance(e, dict):
        # 返回 1 加上所有键和值的节点数之和
        return 1 + sum(_nodes(k) + _nodes(v) for k, v in e.items())
    else:
        # 其他情况下，返回节点数为 1
        return 1


def ordered(seq, keys=None, default=True, warn=False):
    """
    返回一个迭代器，按照 keys 指定的排序规则对 seq 进行排序，
    在有冲突时以保守的方式解决：如果应用一个 key 后，冲突仍然存在，
    则不会继续计算其他 key。

    如果未提供 keys 或 keys 无法完全解决所有冲突（但 default 为 True），
    则会应用两个默认的 keys：_nodes（较小的表达式排在前面）和
    default_sort_key（如果对象的 sort_key 被正确定义，应该能够解决所有冲突）。

    warn 为 True 时，如果没有剩余的 keys 可以解决冲突，将会引发错误。
    这在期望不存在不同项之间的冲突时可用。

    Examples
    ========

    >>> from sympy import ordered, count_ops
    >>> from sympy.abc import x, y

    count_ops 对于此列表不能完全解决冲突，前两个项将保持原始顺序（即排序是稳定的）：

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3],
    ...    count_ops, default=False, warn=False))
    ...
    [y + 2, x + 2, x**2 + y + 3]

    default_sort_key 允许解决冲突：

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3]))
    ...
    [x + 2, y + 2, x**2 + y + 3]

    在此示例中，按长度和总和排序序列：

    >>> seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]], [
    ...    lambda x: len(x),
    ...    lambda x: sum(x)]]
    ...
    >>> list(ordered(seq, keys, default=False, warn=False))
    [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]
    """
    pass  # ordered 函数本身没有代码实现，只有文档说明其功能和用法
    """
    If ``warn`` is True, an error will be raised if there were not
    enough keys to break ties:

    >>> list(ordered(seq, keys, default=False, warn=True))
    Traceback (most recent call last):
    ...
    ValueError: not enough keys to break ties


    Notes
    =====

    The decorated sort is one of the fastest ways to sort a sequence for
    which special item comparison is desired: the sequence is decorated,
    sorted on the basis of the decoration (e.g. making all letters lower
    case) and then undecorated. If one wants to break ties for items that
    have the same decorated value, a second key can be used. But if the
    second key is expensive to compute then it is inefficient to decorate
    all items with both keys: only those items having identical first key
    values need to be decorated. This function applies keys successively
    only when needed to break ties. By yielding an iterator, use of the
    tie-breaker is delayed as long as possible.

    This function is best used in cases when use of the first key is
    expected to be a good hashing function; if there are no unique hashes
    from application of a key, then that key should not have been used. The
    exception, however, is that even if there are many collisions, if the
    first group is small and one does not need to process all items in the
    list then time will not be wasted sorting what one was not interested
    in. For example, if one were looking for the minimum in a list and
    there were several criteria used to define the sort order, then this
    function would be good at returning that quickly if the first group
    of candidates is small relative to the number of items being processed.

    """

    # 创建一个默认值为列表的 defaultdict 对象 d
    d = defaultdict(list)
    # 如果 keys 存在
    if keys:
        # 如果 keys 是 list 或者 tuple 类型
        if isinstance(keys, (list, tuple)):
            keys = list(keys)
            # 弹出第一个键作为 f
            f = keys.pop(0)
        else:
            # 否则将 keys 赋给 f，并清空 keys
            f = keys
            keys = []
        # 遍历 seq 序列
        for a in seq:
            # 将 f(a) 的结果作为键，a 加入到 d 中对应键的列表中
            d[f(a)].append(a)
    else:
        # 如果 keys 不存在且 default=False，抛出 ValueError 异常
        if not default:
            raise ValueError('if default=False then keys must be provided')
        # 否则将整个 seq 加入到 d 的 None 键对应的列表中
        d[None].extend(seq)

    # 对 d.items() 排序后遍历键值对
    for k, value in sorted(d.items()):
        # 如果某个键对应的值长度大于 1
        if len(value) > 1:
            # 如果 keys 存在，则对 value 应用 ordered 函数
            if keys:
                value = ordered(value, keys, default, warn)
            # 如果 default 存在，则对 value 应用 ordered 函数
            elif default:
                value = ordered(value, (_nodes, default_sort_key,),
                               default=False, warn=warn)
            # 如果 warn 存在
            elif warn:
                # 获取 value 中唯一的元素列表 u
                u = list(uniq(value))
                # 如果 u 的长度大于 1，抛出 ValueError 异常
                if len(u) > 1:
                    raise ValueError(
                        'not enough keys to break ties: %s' % u)
        # 使用生成器语法 yield from 返回 value 的元素
        yield from value
```