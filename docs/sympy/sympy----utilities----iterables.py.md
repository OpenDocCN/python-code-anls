# `D:\src\scipysrc\sympy\sympy\utilities\iterables.py`

```
from collections import Counter, defaultdict, OrderedDict
from itertools import (
    chain, combinations, combinations_with_replacement, cycle, islice,
    permutations, product, groupby
)
# 引入需要的模块：Counter（计数器）、defaultdict（默认字典）、OrderedDict（有序字典）、
# chain（链式连接迭代器）、combinations（组合迭代器）、combinations_with_replacement
# （可重复组合迭代器）、cycle（循环迭代器）、islice（切片迭代器）、permutations（排列迭代器）、
# product（笛卡尔积迭代器）、groupby（分组迭代器）

# For backwards compatibility
from itertools import product as cartes # noqa: F401
# 引入 itertools 中的 product 函数，并命名为 cartes，用于向后兼容性

from operator import gt
# 引入 operator 模块中的 gt 运算符函数（大于运算符）

# this is the logical location of these functions
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)
# 导入 sympy 库中 utilities.enumerative 中的三个函数：multiset_partitions_taocp、list_visitor 和
# MultisetPartitionTraverser，这些函数的逻辑位置在此

from sympy.utilities.misc import as_int
# 导入 sympy 库中 utilities.misc 中的 as_int 函数，用于将输入转换为整数

from sympy.utilities.decorator import deprecated
# 导入 sympy 库中 utilities.decorator 中的 deprecated 装饰器，用于标记函数或类已经废弃

def is_palindromic(s, i=0, j=None):
    """
    Return True if the sequence is the same from left to right as it
    is from right to left in the whole sequence (default) or in the
    Python slice ``s[i: j]``; else False.

    Examples
    ========

    >>> from sympy.utilities.iterables import is_palindromic
    >>> is_palindromic([1, 0, 1])
    True
    >>> is_palindromic('abcbb')
    False
    >>> is_palindromic('abcbb', 1)
    False

    Normal Python slicing is performed in place so there is no need to
    create a slice of the sequence for testing:

    >>> is_palindromic('abcbb', 1, -1)
    True
    >>> is_palindromic('abcbb', -4, -1)
    True

    See Also
    ========

    sympy.ntheory.digits.is_palindromic: tests integers

    """
    i, j, _ = slice(i, j).indices(len(s))
    # 计算切片的起始索引和结束索引，确保它们在合理的范围内
    m = (j - i)//2
    # 计算需要比较的一半元素个数，如果长度是奇数，中间元素会被忽略
    return all(s[i + k] == s[j - 1 - k] for k in range(m))
    # 检查从左到右和从右到左的对应元素是否相等，是则返回 True，否则返回 False


def flatten(iterable, levels=None, cls=None):  # noqa: F811
    """
    Recursively denest iterable containers.

    >>> from sympy import flatten

    >>> flatten([1, 2, 3])
    [1, 2, 3]
    >>> flatten([1, 2, [3]])
    [1, 2, 3]
    >>> flatten([1, [2, 3], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> flatten([1.0, 2, (1, None)])
    [1.0, 2, 1, None]

    If you want to denest only a specified number of levels of
    nested containers, then set ``levels`` flag to the desired
    number of levels::

    >>> ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    >>> flatten(ls, levels=1)
    [(-2, -1), (1, 2), (0, 0)]

    If cls argument is specified, it will only flatten instances of that
    class, for example:

    >>> from sympy import Basic, S
    >>> class MyOp(Basic):
    ...     pass
    ...
    >>> flatten([MyOp(S(1), MyOp(S(2), S(3)))], cls=MyOp)
    [1, 2, 3]

    adapted from https://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks
    """
    from sympy.tensor.array import NDimArray
    # 导入 sympy 库中 tensor.array 的 NDimArray 类

    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError(
                "expected non-negative number of levels, got %s" % levels)
    # 如果 levels 参数不为 None，则根据其值进行递归展开的层数控制

    if cls is None:
        def reducible(x):
            return is_sequence(x, set)
    else:
        def reducible(x):
            return isinstance(x, cls)
    # 定义一个判断是否可以继续展开的函数 reducible，根据参数 cls 是否为 None 进行判断

    result = []
    # 初始化空列表 result，用于存储展开后的结果
    # 遍历可迭代对象中的每个元素
    for el in iterable:
        # 检查当前元素是否可约简
        if reducible(el):
            # 如果可约简，并且 el 具有 'args' 属性，且不是 NDimArray 类型
            if hasattr(el, 'args') and not isinstance(el, NDimArray):
                # 将 el 赋值为其 args 属性的值（假设 el 是一个对象，其属性 args 是另一个可迭代对象）
                el = el.args
            # 递归调用 flatten 函数，将 el 展平后的结果扩展到 result 中
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            # 如果当前元素不可约简，直接将其添加到 result 中
            result.append(el)

    # 返回最终展平后的结果列表
    return result
# 定义函数 unflatten，用于将 iter 分组成长度为 n 的元组列表。如果 iter 的长度不是 n 的倍数，则引发 ValueError 错误。
def unflatten(iter, n=2):
    if n < 1 or len(iter) % n:
        raise ValueError('iter length is not a multiple of %i' % n)
    # 使用 zip 和生成器表达式将 iter 分组成长度为 n 的元组列表并返回
    return list(zip(*(iter[i::n] for i in range(n))))


# 定义函数 reshape，根据 how 的模板对序列 seq 进行重塑
def reshape(seq, how):
    # 计算模板 how 的总和
    m = sum(flatten(how))
    # 计算 seq 的长度和商以及余数
    n, rem = divmod(len(seq), m)
    # 如果模板 how 的总和小于 0 或者余数不为 0，则引发 ValueError 错误
    if m < 0 or rem:
        raise ValueError('template must sum to positive number '
                         'that divides the length of the sequence')
    i = 0
    container = type(how)
    rv = [None]*n
    # 遍历 rv 的长度
    for k in range(len(rv)):
        _rv = []
        # 遍历 how 中的每个元素 hi
        for hi in how:
            # 如果 hi 是 int 类型，则从 seq 中取出对应长度的子序列
            if isinstance(hi, int):
                _rv.extend(seq[i: i + hi])
                i += hi
            else:
                # 否则，根据 hi 的模板调用 reshape 函数并将结果放入 _rv
                n = sum(flatten(hi))
                hi_type = type(hi)
                _rv.append(hi_type(reshape(seq[i: i + n], hi)[0]))
                i += n
        # 将 _rv 转换为 container 类型并赋值给 rv[k]
        rv[k] = container(_rv)
    # 返回与 seq 类型相同的 rv
    return type(seq)(rv)


# 定义函数 group，将序列 seq 拆分为相邻相等元素的子列表
def group(seq, multiple=True):
    # 如果 multiple 为 True，则返回相邻相同元素的子列表的列表
    if multiple:
        return [(list(g)) for _, g in groupby(seq)]
    # 否则，返回相邻相同元素及其出现次数的元组列表
    return [(k, len(list(g))) for k, g in groupby(seq)]


# 定义函数 _iproduct2，计算两个可能无限迭代器的笛卡尔积
def _iproduct2(iterable1, iterable2):
    it1 = iter(iterable1)
    it2 = iter(iterable2)
    elems1 = []
    elems2 = []
    sentinel = object()
    
    # 定义内部函数 append，用于迭代并将元素添加到指定列表中
    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)

    n = 0
    # 分别迭代 iterable1 和 iterable2，并将元素添加到 elems1 和 elems2 中
    append(it1, elems1)
    append(it2, elems2)
    # 当 n 小于等于 elems1 和 elems2 的总长度时，执行循环
    while n <= len(elems1) + len(elems2):
        # 对于 m 在 n-len(elems1)+1 到 len(elems2) 范围内的每一个值，生成元组
        for m in range(n-len(elems1)+1, len(elems2)):
            # 生成一个元组，包含 elems1 的第 n-m 个元素和 elems2 的第 m 个元素
            yield (elems1[n-m], elems2[m])
        # n 自增 1
        n += 1
        # 将 elems1 和 elems2 追加到 it1 和 it2 中
        append(it1, elems1)
        append(it2, elems2)
# 定义一个函数 iproduct，用于生成多个可迭代对象的笛卡尔积
def iproduct(*iterables):
    '''
    Cartesian product of iterables.

    Generator of the Cartesian product of iterables. This is analogous to
    itertools.product except that it works with infinite iterables and will
    yield any item from the infinite product eventually.

    Examples
    ========

    >>> from sympy.utilities.iterables import iproduct
    >>> sorted(iproduct([1,2], [3,4]))
    [(1, 3), (1, 4), (2, 3), (2, 4)]

    With an infinite iterator:

    >>> from sympy import S
    >>> (3,) in iproduct(S.Integers)
    True
    >>> (3, 4) in iproduct(S.Integers, S.Integers)

    .. seealso::

       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    '''
    # 如果没有可迭代对象，直接生成空元组并返回
    if len(iterables) == 0:
        yield ()
        return
    # 如果只有一个可迭代对象，逐个生成其元素组成的单元素元组
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    # 如果有两个可迭代对象，调用 _iproduct2 函数生成它们的笛卡尔积
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    # 如果有多个可迭代对象，分别处理第一个和其余部分，递归调用 iproduct 生成笛卡尔积
    else:
        first, others = iterables[0], iterables[1:]
        for ef, eo in _iproduct2(first, iproduct(*others)):
            yield (ef,) + eo


# 定义一个函数 multiset，将序列转换为多重集合形式，每个元素及其出现次数组成字典
def multiset(seq):
    """Return the hashable sequence in multiset form with values being the
    multiplicity of the item in the sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset
    >>> multiset('mississippi')
    {'i': 4, 'm': 1, 'p': 2, 's': 4}

    See Also
    ========

    group

    """
    # 使用 Counter 对象统计序列中每个元素的出现次数，并转换为字典返回
    return dict(Counter(seq).items())


# 定义一个函数 ibin，将整数转换为二进制形式的列表表示
def ibin(n, bits=None, str=False):
    """Return a list of length ``bits`` corresponding to the binary value
    of ``n`` with small bits to the right (last). If bits is omitted, the
    length will be the number required to represent ``n``. If the bits are
    desired in reversed order, use the ``[::-1]`` slice of the returned list.

    If a sequence of all bits-length lists starting from ``[0, 0,..., 0]``
    through ``[1, 1, ..., 1]`` are desired, pass a non-integer for bits, e.g.
    ``'all'``.

    If the bit *string* is desired pass ``str=True``.

    Examples
    ========

    >>> from sympy.utilities.iterables import ibin
    >>> ibin(2)
    [1, 0]
    >>> ibin(2, 4)
    [0, 0, 1, 0]

    If all lists corresponding to 0 to 2**n - 1, pass a non-integer
    for bits:

    >>> bits = 2
    >>> for i in ibin(2, 'all'):
    ...     print(i)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)

    If a bit string is desired of a given length, use str=True:

    >>> n = 123
    >>> bits = 10
    >>> ibin(n, bits, str=True)
    '0001111011'
    >>> ibin(n, bits, str=True)[::-1]  # small bits left
    '1101111000'
    >>> list(ibin(3, 'all', str=True))
    ['000', '001', '010', '011', '100', '101', '110', '111']

    """
    # 如果 n 小于 0，则抛出 ValueError 异常
    if n < 0:
        raise ValueError("negative numbers are not allowed")
    # 将 n 转换为整数
    n = as_int(n)

    # 如果未指定 bits，设置为 0
    if bits is None:
        bits = 0
    # 如果输入的 bits 参数不是整数，则尝试将其转换为整数类型
    try:
        bits = as_int(bits)
    # 如果转换过程中出现数值错误（即 bits 参数不能转换为整数），将 bits 设为 -1
    except ValueError:
        bits = -1
    # 如果 bits 参数成功转换为整数，则检查 n 的二进制位数是否大于 bits
    else:
        if n.bit_length() > bits:
            # 如果 n 的二进制位数大于 bits，则抛出 ValueError 异常
            raise ValueError(
                "`bits` must be >= {}".format(n.bit_length()))

    # 如果 str 参数为假值（如空字符串或 False），根据 bits 的值返回不同类型的结果
    if not str:
        # 如果 bits 大于等于 0，则返回一个列表，其中每个元素为 n 的二进制表示的每一位（0 或 1）
        if bits >= 0:
            return [1 if i == "1" else 0 for i in bin(n)[2:].rjust(bits, "0")]
        # 如果 bits 小于 0，则调用 variations 函数生成包含 n 个元素的列表，每个元素是 0 或 1 的变体
        else:
            return variations(range(2), n, repetition=True)
    else:
        # 如果 bits 大于等于 0，则返回 n 的二进制表示，并用 "0" 填充至 bits 位
        if bits >= 0:
            return bin(n)[2:].rjust(bits, "0")
        # 如果 bits 小于 0，则返回一个生成器，生成包含 2**n 个元素的列表，每个元素是 n 位的二进制表示
        else:
            return (bin(i)[2:].rjust(n, "0") for i in range(2**n))
def variations(seq, n, repetition=False):
    r"""Returns an iterator over the n-sized variations of ``seq`` (size N).
    ``repetition`` controls whether items in ``seq`` can appear more than once;

    Examples
    ========

    ``variations(seq, n)`` will return `\frac{N!}{(N - n)!}` permutations without
    repetition of ``seq``'s elements:

        >>> from sympy import variations
        >>> list(variations([1, 2], 2))
        [(1, 2), (2, 1)]

    ``variations(seq, n, True)`` will return the `N^n` permutations obtained
    by allowing repetition of elements:

        >>> list(variations([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 1), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(variations([0, 1], 3, repetition=False))
        []
        >>> list(variations([0, 1], 3, repetition=True))[:4]
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    .. seealso::

       `itertools.permutations
       <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,
       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    """
    # 如果不允许重复，将 seq 转换为元组
    if not repetition:
        seq = tuple(seq)
        # 如果 seq 的长度小于 n，则返回一个空迭代器
        if len(seq) < n:
            return iter(())  # 0 length iterator
        # 返回 seq 的 n 元素排列组合迭代器
        return permutations(seq, n)
    else:
        # 如果 n 为 0，则返回一个包含一个空元组的迭代器
        if n == 0:
            return iter(((),))  # yields 1 empty tuple
        else:
            # 返回包含所有允许重复的元素组合的迭代器
            return product(seq, repeat=n)


def subsets(seq, k=None, repetition=False):
    r"""Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.

    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The
    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,
    whereas there are `2^n` subsets all together. If `k` is ``None`` then all
    `2^n` subsets will be returned from shortest to longest.

    Examples
    ========

    >>> from sympy import subsets

    ``subsets(seq, k)`` will return the
    `\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":

        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    ``subsets(seq, k, repetition=True)`` will return the
    `\frac{(n - 1 + k)!}{k!(n - 1)!}`
    combinations *with* repetition:

        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    """
    # 如果 k 为 None，则根据 repetition 的值选择不同的组合生成器
    if k is None:
        # 如果不允许重复元素，返回不同长度的组合生成器链
        if not repetition:
            return chain.from_iterable((combinations(seq, k)  # 生成所有长度的组合
                                        for k in range(len(seq) + 1)))
        else:
            return chain.from_iterable((combinations_with_replacement(seq, k)  # 生成所有长度的可重复组合
                                        for k in range(len(seq) + 1)))
    else:
        # 如果不允许重复元素，返回指定长度的组合生成器
        if not repetition:
            return combinations(seq, k)  # 生成长度为 k 的所有组合
        else:
            return combinations_with_replacement(seq, k)  # 生成长度为 k 的所有可重复组合
# 定义一个函数，从迭代器中筛选出不在排除列表中的元素，并生成一个迭代器返回
def filter_symbols(iterator, exclude):
    # 将排除列表转换为集合以便快速查找
    exclude = set(exclude)
    # 遍历输入的迭代器
    for s in iterator:
        # 如果当前元素不在排除集合中，则将其 yield 返回
        if s not in exclude:
            yield s

# 生成符号的无限流，以给定的前缀和递增的下标命名，确保不在排除列表中出现
def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    # 将排除列表转换为集合以便快速查找
    exclude = set(exclude or [])
    # 如果未指定符号类别，从 sympy.core 模块导入 Symbol 类
    if cls is None:
        from sympy.core import Symbol
        cls = Symbol

    # 无限循环生成符号
    while True:
        # 构造符号名称，例如 'x0', 'x1', ...
        name = '%s%s' % (prefix, start)
        # 使用给定的类别和参数创建符号对象
        s = cls(name, *args, **assumptions)
        # 如果生成的符号不在排除集合中，则 yield 返回
        if s not in exclude:
            yield s
        start += 1

# 捕获函数 func() 的输出并返回
def capture(func):
    # 导入所需的模块和类
    from io import StringIO
    import sys

    # 保存当前的标准输出流
    stdout = sys.stdout
    # 将 sys.stdout 替换为 StringIO 对象，以便捕获打印输出
    sys.stdout = file = StringIO()
    try:
        # 调用目标函数，将其输出重定向到 StringIO
        func()
    finally:
        # 恢复标准输出流
        sys.stdout = stdout
    # 返回捕获的输出内容作为字符串
    return file.getvalue()

# 根据 keyfunc 对序列 seq 进行筛选和分类
def sift(seq, keyfunc, binary=False):
    # 如果 binary 参数为 False（默认情况），则返回一个字典
    if not binary:
        # 初始化一个空字典
        result = {}
        # 遍历序列 seq
        for item in seq:
            # 计算当前元素的 keyfunc 值
            key = keyfunc(item)
            # 将当前元素添加到结果字典的对应列表中
            if key not in result:
                result[key] = []
            result[key].append(item)
        return result
    else:
        # 如果 binary 参数为 True，返回一个元组 (T, F)，分别包含 keyfunc 为 True 和 False 的元素列表
        # 初始化两个空列表
        T = []
        F = []
        # 遍历序列 seq
        for item in seq:
            # 计算当前元素的 keyfunc 值
            key = keyfunc(item)
            # 根据 keyfunc 的返回值将元素分别加入 T 或 F 列表
            if key:
                T.append(item)
            else:
                F.append(item)
        # 返回包含 T 和 F 列表的元组
        return (T, F)
    # 如果 binary 参数为 False，则使用 defaultdict 创建一个空的列表字典 m
    if not binary:
        m = defaultdict(list)
        # 遍历 seq 序列，根据 keyfunc 函数对每个元素进行分类并添加到对应的列表中
        for i in seq:
            m[keyfunc(i)].append(i)
        # 返回分类后的结果字典 m
        return m
    # 如果 binary 参数为 True，则初始化两个空列表 F 和 T，并将它们作为 sift 的返回值
    sift = F, T = [], []
    # 遍历 seq 序列，根据 keyfunc 函数对每个元素进行分类，并根据分类结果将元素添加到对应的列表中
    for i in seq:
        try:
            sift[keyfunc(i)].append(i)
        # 如果 keyfunc 函数的输出不是二元的，则抛出 ValueError 异常
        except (IndexError, TypeError):
            raise ValueError('keyfunc gave non-binary output')
    # 返回分类后的两个列表 T 和 F
    return T, F
def take(iter, n):
    """
    Return ``n`` items from ``iter`` iterator.
    """
    return [ value for _, value in zip(range(n), iter) ]


def dict_merge(*dicts):
    """
    Merge dictionaries into a single dictionary.
    """
    merged = {}

    for dict in dicts:
        merged.update(dict)

    return merged


def common_prefix(*seqs):
    """
    Return the subsequence that is a common start of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_prefix
    >>> common_prefix(list(range(3)))
    [0, 1, 2]
    >>> common_prefix(list(range(3)), list(range(4)))
    [0, 1, 2]
    >>> common_prefix([1, 2, 3], [1, 2, 5])
    [1, 2]
    >>> common_prefix([1, 2, 3], [1, 3, 5])
    [1]
    """
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    
    # Determine the length of the common prefix among sequences
    i = 0
    for i in range(min(len(s) for s in seqs)):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i += 1
    
    return seqs[0][:i]


def common_suffix(*seqs):
    """
    Return the subsequence that is a common ending of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_suffix
    >>> common_suffix(list(range(3)))
    [0, 1, 2]
    >>> common_suffix(list(range(3)), list(range(4)))
    []
    >>> common_suffix([1, 2, 3], [9, 2, 3])
    [2, 3]
    >>> common_suffix([1, 2, 3], [9, 7, 3])
    [3]
    """
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    
    # Determine the length of the common suffix among sequences
    i = 0
    for i in range(-1, -min(len(s) for s in seqs) - 1, -1):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i -= 1
    
    if i == -1:
        return []
    else:
        return seqs[0][i + 1:]


def prefixes(seq):
    """
    Generate all prefixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import prefixes

    >>> list(prefixes([1,2,3,4]))
    [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[:i + 1]


def postfixes(seq):
    """
    Generate all postfixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import postfixes

    >>> list(postfixes([1,2,3,4]))
    [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[n - i - 1:]


def topological_sort(graph, key=None):
    r"""
    Topological sort of graph's vertices.

    Parameters
    ==========

    graph : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph to be sorted topologically.

    key : callable[T] (optional)
        Ordering key for vertices on the same level. By default the natural
        (e.g. lexicographic) ordering is used (in this case the base type
        must implement ordering relations).

    Examples
    ========

    ```
    # Topological sort example code can be inserted here
    ```
    """
    # 从参数中解包出顶点列表 V 和边列表 E
    V, E = graph
    
    # 初始化一个空列表 L 用于存储拓扑排序的结果
    L = []
    
    # 使用集合 S 存储顶点 V 的副本，并将边列表 E 转换为列表形式
    S = set(V)
    E = list(E)
    
    # 从集合 S 中移除所有出现在边列表 E 中作为目标的顶点
    S.difference_update(u for v, u in E)
    
    # 如果没有指定 key 函数，则定义一个默认的 key 函数返回其输入值
    if key is None:
        def key(value):
            return value
    
    # 根据 key 函数对集合 S 进行降序排序
    S = sorted(S, key=key, reverse=True)
    
    # 当集合 S 非空时执行循环
    while S:
        # 弹出集合 S 中的一个节点作为当前节点，并将其加入到结果列表 L 中
        node = S.pop()
        L.append(node)
    
        # 遍历边列表 E 的副本，并移除所有以当前节点为起点的边
        for u, v in list(E):
            if u == node:
                E.remove((u, v))
    
                # 检查是否存在其他以相同目标节点 v 为终点的边，若不存在则将其对应的顶点加入集合 S 中
                for _u, _v in E:
                    if v == _v:
                        break
                else:
                    kv = key(v)
    
                    # 根据 key 函数的值将顶点 v 插入到集合 S 中的适当位置
                    for i, s in enumerate(S):
                        ks = key(s)
    
                        if kv > ks:
                            S.insert(i, v)
                            break
                    else:
                        S.append(v)
    
    # 如果边列表 E 非空，则说明图中存在环路，抛出 ValueError 异常
    if E:
        raise ValueError("cycle detected")
    else:
        # 返回拓扑排序后的结果列表 L
        return L
# 定义一个函数，计算有向图的强连通分量，并按照反向拓扑顺序返回结果
def strongly_connected_components(G):
    r"""
    Strongly connected components of a directed graph in reverse topological
    order.
    
    Parameters
    ==========
    
    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose strongly connected components are to be found.
    
    Examples
    ========
    
    Consider a directed graph (in dot notation)::
    
        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }
    
    .. graphviz::
    
        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }
    
    where vertices are the letters A, B, C and D. This graph can be encoded
    using Python's elementary data structures as follows::
    
        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B'), ('B', 'D')]
    
    The strongly connected components of this graph can be computed as
    
        >>> from sympy.utilities.iterables import strongly_connected_components
    
        >>> strongly_connected_components((V, E))
        [['D'], ['B', 'C'], ['A']]
    
    This also gives the components in reverse topological order.
    
    Since the subgraph containing B and C has a cycle they must be together in
    a strongly connected component. A and D are connected to the rest of the
    graph but not in a cyclic manner so they appear as their own strongly
    connected components.
    
    Notes
    =====
    
    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.
    
    This function uses Tarjan's algorithm to compute the strongly connected
    components in `O(|V|+|E|)` (linear) time.
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/Strongly_connected_component
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    
    See Also
    ========
    
    sympy.utilities.iterables.connected_components
    
    """
    # 创建一个从顶点到其邻居的映射
    V, E = G
    Gmap = {vi: [] for vi in V}
    for v1, v2 in E:
        Gmap[v1].append(v2)
    # 调用内部函数来计算强连通分量
    return _strongly_connected_components(V, Gmap)


def _strongly_connected_components(V, Gmap):
    """More efficient internal routine for strongly_connected_components"""
    #
    # Here V is an iterable of vertices and Gmap is a dict mapping each vertex
    # to a list of neighbours e.g.:
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = {0: [2, 3], 1: [0]}
    #
    # For a large graph these data structures can often be created more
    # efficiently then those expected by strongly_connected_components() which
    # in this case would be
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = [(0, 2), (0, 3), (1, 0)]
    #
    # XXX: Maybe this should be the recommended function to use instead...
    #
    
    # 非递归的 Tarjan 算法:
    # 初始化低链接值和索引字典
    lowlink = {}
    indices = {}
    # 使用有序字典来存储节点和其相关信息，这里初始化一个空的有序字典
    stack = OrderedDict()
    
    # 用于存储当前递归调用的栈
    callstack = []
    
    # 用于存储所有找到的强连通分量
    components = []
    
    # 一个特殊的对象，表示不再有更多的子节点需要处理
    nomore = object()
    
    # 定义一个函数 start(v)，用于开始处理节点 v
    def start(v):
        # 获取当前节点 v 在 stack 中的索引，初始化 lowlink 和 indices
        index = len(stack)
        indices[v] = lowlink[v] = index
        # 将节点 v 压入 stack 中，并初始化其邻居迭代器到 callstack
        stack[v] = None
        callstack.append((v, iter(Gmap[v])))
    
    # 定义一个函数 finish(v1)，用于完成处理节点 v1
    def finish(v1):
        # 完成一个强连通分量的处理？
        if lowlink[v1] == indices[v1]:
            # 构建一个强连通分量，从 stack 中依次取出
            component = [stack.popitem()[0]]
            while component[-1] is not v1:
                component.append(stack.popitem()[0])
            components.append(component[::-1])
        # 弹出 callstack 的栈顶元素 v2，并处理它的父节点 v1
        v2, _ = callstack.pop()
        if callstack:
            v1, _ = callstack[-1]
            # 更新父节点 v1 的 lowlink 值
            lowlink[v1] = min(lowlink[v1], lowlink[v2])
    
    # 遍历图中的每个节点 v
    for v in V:
        if v in indices:
            continue
        # 对于尚未访问过的节点 v，开始处理它
        start(v)
        # 当 callstack 非空时继续处理
        while callstack:
            v1, it1 = callstack[-1]
            v2 = next(it1, nomore)
            # 已经处理完节点 v1 的所有子节点？
            if v2 is nomore:
                finish(v1)
            # 递归处理节点 v2
            elif v2 not in indices:
                start(v2)
            elif v2 in stack:
                # 更新节点 v1 的 lowlink 值
                lowlink[v1] = min(lowlink[v1], indices[v2])
    
    # 返回按照逆拓扑排序的强连通分量列表
    return components
# 计算无向图的连通分量或有向图的弱连通分量的函数
def connected_components(G):
    # 解包参数元组，获取图的顶点集合 V 和边集合 E
    V, E = G
    # 创建一个空列表用于存储无向图的边，确保每条边都以双向方式存储
    E_undirected = []
    # 遍历每条边 (v1, v2)，将其添加为 (v1, v2) 和 (v2, v1)
    for v1, v2 in E:
        E_undirected.extend([(v1, v2), (v2, v1)])
    # 调用另一个函数 strongly_connected_components，返回连通分量的列表
    return strongly_connected_components((V, E_undirected))


def rotate_left(x, y):
    """
    左旋转列表 x，旋转步数为 y。

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_left
    >>> a = [0, 1, 2]
    >>> rotate_left(a, 1)
    [1, 2, 0]
    """
    # 如果列表 x 为空，则返回空列表
    if len(x) == 0:
        return []
    # 计算有效的旋转步数 y，确保 y 在列表长度范围内
    y = y % len(x)
    # 返回左旋转后的列表结果
    return x[y:] + x[:y]


def rotate_right(x, y):
    """
    右旋转列表 x，旋转步数为 y。

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_right
    >>> a = [0, 1, 2]
    >>> rotate_right(a, 1)
    [2, 0, 1]
    """
    # 如果列表 x 为空，则返回空列表
    if len(x) == 0:
        return []
    # 计算有效的旋转步数 y，确保 y 在列表长度范围内
    y = len(x) - y % len(x)
    # 返回右旋转后的列表结果
    return x[y:] + x[:y]


def least_rotation(x, key=None):
    '''
    返回使字符串/列表/元组等字典序最小的左旋转步数。

    Examples
    ========

    >>> from sympy.utilities.iterables import least_rotation, rotate_left
    >>> a = [3, 1, 5, 1, 2]
    >>> least_rotation(a)
    3
    >>> rotate_left(a, _)
    [1, 2, 3, 1, 5]

    References
    ==========

    '''
    # 这个函数尚未实现完整
    '''
    Lexicographically minimal string rotation algorithm implementation.
    
    Source: https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation
    
    '''
    # 导入 sympy 库中的 Id 函数
    from sympy.functions.elementary.miscellaneous import Id
    # 如果 key 为 None，则将其设置为 Id 函数
    if key is None: key = Id
    # 将字符串 x 与其自身连接，以避免模运算
    S = x + x
    # 失败函数，初始化为全 -1 的列表，长度为 S 的长度
    f = [-1] * len(S)
    # 当前找到的字符串的最小旋转位置
    k = 0
    # 遍历字符串 S 的每个字符（从第二个字符开始）
    for j in range(1,len(S)):
        sj = S[j]  # 当前字符 S[j]
        i = f[j-k-1]  # 失败函数索引
        # 使用 KMP 算法更新失败函数
        while i != -1 and sj != S[k+i+1]:
            # 根据 key 函数比较 sj 和 S[k+i+1] 的大小
            if key(sj) < key(S[k+i+1]):
                k = j-i-1
            i = f[i]
        # 更新失败函数的值
        if sj != S[k+i+1]:
            if key(sj) < key(S[k]):
                k = j
            f[j-k] = -1
        else:
            f[j-k] = i+1
    # 返回最小旋转位置
    return k
def multiset_combinations(m, n, g=None):
    """
    Return the unique combinations of size ``n`` from multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_combinations
    >>> from itertools import combinations
    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]
    ['abb', 'aby', 'bby']

    >>> def count(f, s): return len(list(f(s, 3)))

    The number of combinations depends on the number of letters; the
    number of unique combinations depends on how the letters are
    repeated.

    >>> s1 = 'abracadabra'
    >>> s2 = 'banana tree'
    >>> count(combinations, s1), count(multiset_combinations, s1)
    (165, 23)
    >>> count(combinations, s2), count(multiset_combinations, s2)
    (165, 54)

    """
    from sympy.core.sorting import ordered  # 导入排序函数 ordered
    if g is None:  # 如果 g 为 None
        if isinstance(m, dict):  # 如果 m 是字典类型
            if any(as_int(v) < 0 for v in m.values()):  # 如果字典中任意值小于0
                raise ValueError('counts cannot be negative')  # 抛出值错误异常
            N = sum(m.values())  # 计算字典中值的总和
            if n > N:  # 如果 n 大于总和 N
                return  # 返回空
            g = [[k, m[k]] for k in ordered(m)]  # 构造排序后的键值对列表 g
        else:  # 如果 m 不是字典类型
            m = list(m)  # 将 m 转换为列表
            N = len(m)  # 计算列表 m 的长度
            if n > N:  # 如果 n 大于长度 N
                return  # 返回空
            try:
                m = multiset(m)  # 尝试将列表 m 转换为 multiset 对象
                g = [(k, m[k]) for k in ordered(m)]  # 构造排序后的键值对列表 g
            except TypeError:
                m = list(ordered(m))  # 对 m 进行排序并转换为列表
                g = [list(i) for i in group(m, multiple=False)]  # 使用 group 函数将 m 分组，构造 g
        del m  # 删除变量 m
    else:  # 如果 g 不为 None
        # 不检查计数，因为 g 用于内部使用
        N = sum(v for k, v in g)  # 计算 g 中所有计数值的总和
    if n > N or not n:  # 如果 n 大于总和 N 或者 n 为 0
        yield []  # 生成空列表
    else:
        for i, (k, v) in enumerate(g):  # 遍历 g 中的每个元素
            if v >= n:  # 如果计数 v 大于等于 n
                yield [k]*n  # 生成由 k 重复 n 次构成的列表
                v = n - 1  # 更新 v 的值
            for v in range(min(n, v), 0, -1):  # 对 v 进行范围内的递减循环
                for j in multiset_combinations(None, n - v, g[i + 1:]):  # 递归调用 multiset_combinations
                    rv = [k]*v + j  # 构造结果列表 rv
                    if len(rv) == n:  # 如果 rv 的长度等于 n
                        yield rv  # 生成 rv

def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    from sympy.core.sorting import ordered  # 导入排序函数 ordered
    if g is None:  # 如果 g 为 None
        if isinstance(m, dict):  # 如果 m 是字典类型
            if any(as_int(v) < 0 for v in m.values()):  # 如果字典中任意值小于0
                raise ValueError('counts cannot be negative')  # 抛出值错误异常
            g = [[k, m[k]] for k in ordered(m)]  # 构造排序后的键值对列表 g
        else:  # 如果 m 不是字典类型
            m = list(ordered(m))  # 对 m 进行排序并转换为列表
            g = [list(i) for i in group(m, multiple=False)]  # 使用 group 函数将 m 分组，构造 g
        del m  # 删除变量 m
    do = [gi for gi in g if gi[1] > 0]  # 构造过滤后的列表 do
    SUM = sum(gi[1] for gi in do)  # 计算 do 中所有计数值的总和
    if not do or size is not None and (size > SUM or size < 1):  # 如果 do 为空或者 size 不为 None 且 size 超出范围
        if not do and size is None or size == 0:  # 如果 do 为空且 size 为 None 或者 size 为 0
            yield []  # 生成空列表
        return  # 返回
    # 如果 size 等于 1，则对每个键值对 (k, v) 执行迭代，生成包含单个元素 [k] 的列表
    elif size == 1:
        for k, v in do:
            yield [k]
    
    # 如果 do 列表的长度为 1，则将第一个键值对 (k, v) 解包，并根据 size 的值确定生成列表的长度：
    elif len(do) == 1:
        k, v = do[0]
        # 如果 size 为 None，则使用 v；否则，使用 size 与 v 中的较小值（如果 size 大于 v，则返回 0）
        v = v if size is None else (size if size <= v else 0)
        # 生成包含单个元素 [k] 重复 v 次的列表
        yield [k for i in range(v)]
    
    # 如果所有键值对 (k, v) 中的 v 均为 1，则对键列表进行大小为 size 的排列组合，并生成结果列表
    elif all(v == 1 for k, v in do):
        # 使用 itertools 中的 permutations 函数，对键列表进行大小为 size 的排列组合，并逐个生成结果列表
        for p in permutations([k for k, v in do], size):
            yield list(p)
    
    # 如果不满足以上条件，则执行默认情况：
    else:
        # 如果 size 为 None，则将其设为 SUM
        size = size if size is not None else SUM
        # 对 do 列表中的每个键值对 (k, v) 进行迭代，修改 v 并递归调用 multiset_permutations 函数，
        # 生成包含键列表的长度为 size-1 的排列组合列表，并在每个结果列表前添加键 k
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1
# 根据给定的分区向量 `vector` 对序列 `seq` 进行分区，并返回分区后的结果列表 `p`
def _partition(seq, vector, m=None):
    # 如果未指定分区数 `m`，则根据向量 `vector` 中的最大值确定分区数
    if m is None:
        m = max(vector) + 1
    # 如果向量 `vector` 是一个整数，则将其解释为 `m`，而 `m` 则解释为 `vector`
    elif isinstance(vector, int):  # entered as m, vector
        vector, m = m, vector
    # 初始化一个空列表，用于存储分区后的结果
    p = [[] for i in range(m)]
    # 遍历序列 `seq` 和分区向量 `vector`，根据分区向量将元素分配到对应的分区中
    for i, v in enumerate(vector):
        p[v].append(seq[i])
    # 返回分区后的结果列表 `p`
    return p


# 枚举给定元素数 `n` 的所有分区，通过生成器返回当前分区数 `m` 和一个可变列表 `q`
def _set_partitions(n):
    """Cycle through all partitions of n elements, yielding the
    current number of partitions, ``m``, and a mutable list, ``q``
    such that ``element[i]`` is in part ``q[i]`` of the partition.

    NOTE: ``q`` is modified in place and generally should not be changed
    between function calls.

    Examples
    ========

    >>> from sympy.utilities.iterables import _set_partitions, _partition
    >>> for m, q in _set_partitions(3):
    ...     print('%s %s %s' % (m, q, _partition('abc', q, m)))
    1 [0, 0, 0] [['a', 'b', 'c']]
    2 [0, 0, 1] [['a', 'b'], ['c']]
    2 [0, 1, 0] [['a', 'c'], ['b']]
    2 [0, 1, 1] [['a'], ['b', 'c']]
    3 [0, 1, 2] [['a'], ['b'], ['c']]

    Notes
    =====

    This algorithm is similar to, and solves the same problem as,
    Algorithm 7.2.1.5H, from volume 4A of Knuth's The Art of Computer
    Programming.  Knuth uses the term "restricted growth string" where
    this code refers to a "partition vector". In each case, the meaning is
    the same: the value in the ith element of the vector specifies to
    which part the ith set element is to be assigned.

    At the lowest level, this code implements an n-digit big-endian
    counter (stored in the array q) which is incremented (with carries) to
    get the next partition in the sequence.  A special twist is that a
    digit is constrained to be at most one greater than the maximum of all
    the digits to the left of it.  The array p maintains this maximum, so
    that the code can efficiently decide when a digit can be incremented
    in place or whether it needs to be reset to 0 and trigger a carry to
    the next digit.  The enumeration starts with all the digits 0 (which
    corresponds to all the set elements being assigned to the same 0th
    part), and ends with 0123...n, which corresponds to each set element
    being assigned to a different, singleton, part.

    This routine was rewritten to use 0-based lists while trying to
    preserve the beauty and efficiency of the original algorithm.

    References
    ========
    combinatorics.partitions.Partition.from_rgs
    """
    pass  # 这是一个生成器函数，实际功能在调用时通过生成器的迭代来实现
    # 初始化 p 和 q 列表，长度为 n，用于存储排列和组合的状态
    p = [0]*n
    q = [0]*n
    # 初始化排列的数量 nc 为 1，并返回初始状态 q
    nc = 1
    yield nc, q  # 返回当前排列数量和状态 q
    
    # 当排列数量 nc 不等于 n 时，继续生成下一个排列
    while nc != n:
        m = n
        # 在当前位置 m 依次减少，直到找到一个可以操作的位置 i
        while 1:
            m -= 1
            i = q[m]
            # 如果位置 i 没有被使用过，则跳出循环
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        # 更新排列数量 nc
        nc += m - n
        p[0] += n - m
        # 如果 i 等于 nc，则说明新的位置 i 可以增加一个排列
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield nc, q  # 返回更新后的排列数量和状态 q
# 定义一个函数，用于生成给定多重集的唯一分区（以列表形式返回）
def multiset_partitions(multiset, m=None):
    """
    Return unique partitions of the given multiset (in list form).
    If ``m`` is None, all multisets will be returned, otherwise only
    partitions with ``m`` parts will be returned.

    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]
    will be supplied.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_partitions
    >>> list(multiset_partitions([1, 2, 3, 4], 2))
    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
    [[1], [2, 3, 4]]]
    >>> list(multiset_partitions([1, 2, 3, 4], 1))
    [[[1, 2, 3, 4]]]

    Only unique partitions are returned and these will be returned in a
    canonical order regardless of the order of the input:

    >>> a = [1, 2, 2, 1]
    >>> ans = list(multiset_partitions(a, 2))
    >>> a.sort()
    >>> list(multiset_partitions(a, 2)) == ans
    True
    >>> a = range(3, 1, -1)
    >>> (list(multiset_partitions(a)) ==
    ...  list(multiset_partitions(sorted(a))))
    True

    If m is omitted then all partitions will be returned:

    >>> list(multiset_partitions([1, 1, 2]))
    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]
    >>> list(multiset_partitions([1]*3))
    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    Counting
    ========

    The number of partitions of a set is given by the bell number:

    >>> from sympy import bell
    >>> len(list(multiset_partitions(5))) == bell(5) == 52
    True

    The number of partitions of length k from a set of size n is given by the
    Stirling Number of the 2nd kind:

    >>> from sympy.functions.combinatorial.numbers import stirling
    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15
    True

    These comments on counting apply to *sets*, not multisets.

    Notes
    =====

    When all the elements are the same in the multiset, the order
    of the returned partitions is determined by the ``partitions``
    routine. If one is counting partitions then it is better to use
    the ``nT`` function.

    See Also
    ========

    partitions
    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition
    sympy.functions.combinatorial.numbers.nT

    """
    # 此函数根据提供的输入调度到几个特殊情况的例程。
    pass
    # 如果 multiset 是整数类型，则将其视为整数 n，并进行处理
    if isinstance(multiset, int):
        n = multiset
        # 如果 m 存在且大于 n，则直接返回，无需继续计算
        if m and m > n:
            return
        # 将整数 n 转换成列表形式，即 [0, 1, ..., n-1]
        multiset = list(range(n))
        # 如果 m 等于 1，则生成单个分区并返回
        if m == 1:
            yield [multiset[:]]
            return

        # 对于给定的整数 n，通过 _set_partitions(n) 生成其所有的分区方式
        for nc, q in _set_partitions(n):
            # 如果 m 为 None 或者当前分区的部分数目等于 m，则生成对应的分区
            if m is None or nc == m:
                # 创建一个空列表 rv，用于存储分区结果
                rv = [[] for i in range(nc)]
                # 将 multiset 中的元素按照分区 q 分配到 rv 中的各个子列表中
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                # 返回当前分区的结果 rv
                yield rv
        return

    # 如果 multiset 是字符串且长度为 1，则将其转换成包含该字符串的列表
    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]

    # 如果 multiset 没有多样性（即所有元素都相同），处理方式如下
    if not has_variety(multiset):
        # 对于只包含相同元素的 multiset，其分区结果与整数 n 的分区结果类似
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        # 取 multiset 的第一个元素，并生成其按照给定分区方式的结果
        x = multiset[:1]
        for size, p in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                # 将元素 x 重复 p[k] 次，并按照 k 的顺序生成结果列表 rv
                for k in sorted(p):
                    rv.extend([x*k]*p[k])
                # 返回当前分区的结果 rv
                yield rv
    else:
        # 导入 ordered 函数，用于对多重集合进行排序
        from sympy.core.sorting import ordered
        # 将多重集合转换为有序列表
        multiset = list(ordered(multiset))
        # 获取多重集合的元素个数
        n = len(multiset)
        # 如果 m 存在且大于多重集合的元素个数，则直接返回
        if m and m > n:
            return
        # 如果 m 等于 1，直接生成包含整个多重集合的列表，并返回
        if m == 1:
            yield [multiset[:]]
            return

        # 将多重集合拆分为两个列表 -
        # 一个包含元素本身，另一个（与元素列表同长度）
        # 包含相应元素的重复次数。
        elements, multiplicities = zip(*group(multiset, False))

        # 如果元素列表长度小于多重集合长度，说明存在重复元素
        if len(elements) < len(multiset):
            # 一般情况 - 多重集合包含多个不同元素，并且至少有一个元素重复多次。
            if m:
                # 使用 MultisetPartitionTraverser 枚举具有特定重复次数的分区
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m-1, m):
                    yield list_visitor(state, elements)
            else:
                # 对于无限制重复次数的情况，使用 Taocp 的算法生成分区
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            # 集合分区情况 - 没有重复元素。与上述整数参数情况基本相同，
            # 对于某些情况可能存在优化，但目前尚未实现。
            for nc, q in _set_partitions(n):
                if m is None or nc == m:
                    # 生成分区，将索引映射回多重集合中的元素
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]
# 定义生成给定正整数 n 的所有分区的生成器函数
def partitions(n, m=None, k=None, size=False):
    """Generate all partitions of positive integer, n.

    Each partition is represented as a dictionary, mapping an integer
    to the number of copies of that integer in the partition.  For example,
    the first partition of 4 returned is {4: 1}, "4: one of them".

    Parameters
    ==========
    n : int
        正整数，要分区的数值
    m : int, optional
        分区中部分数的最大数量（助记：m，最大部分数）
    k : int, optional
        限制分区中保留的数值（助记：k，键）
    size : bool, default: False
        如果为 True，则返回元组 (M, P)，其中 M 是多重性的总和，P 是生成的分区。
        如果为 False，则只返回生成的分区。

    Examples
    ========

    >>> from sympy.utilities.iterables import partitions

    The numbers appearing in the partition (the key of the returned dict)
    are limited with k:

    >>> for p in partitions(6, k=2):  # doctest: +SKIP
    ...     print(p)
    {2: 3}
    {1: 2, 2: 2}
    {1: 4, 2: 1}
    {1: 6}

    The maximum number of parts in the partition (the sum of the values in
    the returned dict) are limited with m (default value, None, gives
    partitions from 1 through n):

    >>> for p in partitions(6, m=2):  # doctest: +SKIP
    ...     print(p)
    ...
    {6: 1}
    {1: 1, 5: 1}
    {2: 1, 4: 1}
    {3: 2}

    References
    ==========

    .. [1] modified from Tim Peter's version to allow for k and m values:
           https://code.activestate.com/recipes/218332-generator-for-integer-partitions/

    See Also
    ========

    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition

    """
    # 处理特殊情况：n 小于等于 0，或者 m 和 k 的限制使得无法生成有效分区
    if (n <= 0 or
        m is not None and m < 1 or
        k is not None and k < 1 or
        m and k and m*k < n):
        # 对于这些输入，空集是唯一的处理方式，返回 {} 表示空集，与计数约定一致，例如 nT(0) == 1。
        if size:
            yield 0, {}
        else:
            yield {}
        return

    # 确定要使用的 m 和 k 的值
    if m is None:
        m = n
    else:
        m = min(m, n)
    k = min(k or n, n)

    # 将 n, m, k 转换为整数
    n, m, k = as_int(n), as_int(m), as_int(k)
    
    # 计算商 q 和余数 r
    q, r = divmod(n, k)
    ms = {k: q}  # 初始的多重性字典，每个键值对表示一个分区元素和其出现的次数
    keys = [k]  # ms.keys()，从大到小排序的键列表

    # 处理余数 r 的情况
    if r:
        ms[r] = 1
        keys.append(r)

    # 计算剩余的部分数目 room
    room = m - q - bool(r)

    # 如果 size 为 True，则返回多重性的总和和其副本
    if size:
        yield sum(ms.values()), ms.copy()
    else:
        yield ms.copy()
    while keys != [1]:
        # 当 keys 不等于 [1] 时执行循环

        # 重用任何一个1。
        if keys[-1] == 1:
            # 如果 keys 列表的最后一个元素是1，则执行以下操作：
            del keys[-1]
            # 删除 keys 列表的最后一个元素
            reuse = ms.pop(1)
            # 弹出 ms 字典中键为1的值，并赋给 reuse
            room += reuse
            # 将 reuse 的值加到 room 中
        else:
            reuse = 0
            # 否则，reuse 设为0

        while 1:
            # 这是一个无限循环，直到中断条件满足为止

            # 令 i 为大于1的最小键。重用一个 i 的实例。
            i = keys[-1]
            # 将 i 设为 keys 列表的最后一个元素
            newcount = ms[i] = ms[i] - 1
            # 将 ms 字典中键为 i 的值减1，并赋给 newcount，并更新 ms 中键为 i 的值
            reuse += i
            # 将 i 加到 reuse 中
            if newcount == 0:
                del keys[-1], ms[i]
                # 如果新计数为0，则删除 keys 列表的最后一个元素和 ms 中键为 i 的条目

            room += 1
            # 将 room 的值加1

            # 将剩余部分分解成大小为 i-1 的片段。
            i -= 1
            # i 减1
            q, r = divmod(reuse, i)
            # 将 reuse 除以 i，并返回商和余数
            need = q + bool(r)
            # 如果 r 不为0，则 need 为 q+1，否则为 q

            if need > room:
                # 如果 need 大于 room，则执行以下操作：
                if not keys:
                    return
                    # 如果 keys 为空列表，则返回
                continue
                # 继续循环

            ms[i] = q
            # 将 q 赋给 ms 字典中键为 i 的值
            keys.append(i)
            # 将 i 添加到 keys 列表中
            if r:
                ms[r] = 1
                # 如果 r 不为0，则将键为 r 的值设为1，并加到 ms 中
                keys.append(r)
                # 将 r 添加到 keys 列表中
            break
            # 中断当前循环

        room -= need
        # 将 room 减去 need 的值
        if size:
            yield sum(ms.values()), ms.copy()
            # 如果 size 为真，则生成 ms 字典中值的总和和其副本
        else:
            yield ms.copy()
            # 否则，生成 ms 的副本
# 定义一个生成有序整数 n 的分区的生成器函数
def ordered_partitions(n, m=None, sort=True):
    # 如果 n 小于 1 或者 m 不为 None 且小于 1，则返回空集作为唯一处理这些输入的方式，
    # 并且返回 {} 来表示它与计数约定一致，例如 nT(0) == 1。
    yield []
    return
    if m is None:
        # 如果 m 为 None，则生成所有不同的整数划分
        a = [1]*n  # 初始化数组 a，全部为 1
        y = -1  # 设定 y 的初始值为 -1
        v = n  # 设定 v 的初始值为 n
        while v > 0:
            v -= 1
            x = a[v] + 1  # 计算 x，为数组 a[v] 的值加 1
            while y >= 2 * x:
                a[v] = x  # 更新数组 a[v] 的值为 x
                y -= x  # 更新 y，减去 x
                v += 1  # 更新 v，加 1
            w = v + 1  # 计算 w，为 v 的值加 1
            while x <= y:
                a[v] = x  # 更新数组 a[v] 的值为 x
                a[w] = y  # 更新数组 a[w] 的值为 y
                yield a[:w + 1]  # 生成当前数组 a 的一个切片作为结果
                x += 1  # 更新 x，加 1
                y -= 1  # 更新 y，减 1
            a[v] = x + y  # 更新数组 a[v] 的值为 x + y
            y = a[v] - 1  # 更新 y，为 a[v] 的值减 1
            yield a[:w]  # 生成当前数组 a 的一个切片作为结果
    elif m == 1:
        yield [n]  # 如果 m 等于 1，则生成一个只包含 n 的列表作为结果
    elif n == m:
        yield [1]*n  # 如果 n 等于 m，则生成一个全部为 1 的长度为 n 的列表作为结果
    else:
        # 递归生成大小为 m 的划分
        for b in range(1, n//m + 1):
            a = [b]*m  # 初始化数组 a，全部为 b，长度为 m
            x = n - b*m  # 计算剩余的数值 x
            if not x:
                if sort:
                    yield a  # 如果 x 为 0 并且 sort 为真，则生成当前数组 a 作为结果
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]  # 更新数组 a 的末尾部分
                    yield a  # 生成当前数组 a 作为结果
                    a[-mi:] = [b]*mi  # 恢复数组 a 的末尾部分为全部为 b 的长度为 mi 的列表
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]  # 更新数组 a 的末尾部分
                        yield a  # 生成当前数组 a 作为结果
                        a[-mi:] = [b]*mi  # 恢复数组 a 的末尾部分为全部为 b 的长度为 mi 的列表
def binary_partitions(n):
    """
    Generates the binary partition of *n*.

    A binary partition consists only of numbers that are
    powers of two. Each step reduces a `2^{k+1}` to `2^k` and
    `2^k`. Thus 16 is converted to 8 and 8.

    Examples
    ========

    >>> from sympy.utilities.iterables import binary_partitions
    >>> for i in binary_partitions(5):
    ...     print(i)
    ...
    [4, 1]
    [2, 2, 1]
    [2, 1, 1, 1]
    [1, 1, 1, 1, 1]

    References
    ==========

    .. [1] TAOCP 4, section 7.2.1.5, problem 64

    """
    from math import ceil, log2
    # 计算比 n 大的最小的 2 的幂次方
    power = int(2**(ceil(log2(n))))
    acc = 0  # 初始化累加器
    partition = []  # 初始化分区列表
    while power:
        if acc + power <= n:
            # 如果累加器加上当前幂次方不超过 n，则将当前幂次方加入分区列表，并更新累加器
            partition.append(power)
            acc += power
        power >>= 1  # 将当前幂次方右移一位，相当于除以 2

    # 计算最后一个数的索引
    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition  # 返回当前分区列表
        if partition[last_num] == 2:
            # 如果最后一个数是 2，则将其替换为 1，并在分区列表末尾添加一个 1
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)  # 在分区列表末尾添加一个 1
        partition[last_num] >>= 1  # 将最后一个数右移一位
        x = partition[last_num + 1] = partition[last_num]  # 更新最后一个数
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                # 如果 x 小于等于分区列表剩余长度减一，则删除分区列表后面的 x-1 个元素
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x  # 更新最后一个数
            else:
                x >>= 1  # 否则将 x 右移一位继续处理
    yield [1]*n  # 返回所有元素都是 1 的分区列表


def has_dups(seq):
    """Return True if there are any duplicate elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_dups, Dict, Set
    >>> has_dups((1, 2, 1))
    True
    >>> has_dups(range(3))
    False
    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))
    True
    """
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()  # 初始化空集合
    try:
        return any(True for s in seq if s in unique or unique.add(s))
        # 如果序列中有任何元素在集合中或添加失败，则返回 True
    except TypeError:
        return len(seq) != len(list(uniq(seq)))


def has_variety(seq):
    """Return True if there are any different elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_variety

    >>> has_variety((1, 2, 1))
    True
    >>> has_variety((1, 1, 1))
    False
    """
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s  # 将第一个元素作为哨兵值
        else:
            if s != sentinel:
                return True  # 如果当前元素与哨兵值不相同，则返回 True
    return False  # 如果序列所有元素都相同，则返回 False


def uniq(seq, result=None):
    """
    Yield unique elements from ``seq`` as an iterator. The second
    parameter ``result``  is used internally; it is not necessary
    to pass anything for this.

    Note: changing the sequence during iteration will raise a
    RuntimeError if the size of the sequence is known; if you pass
    an iterator and advance the iterator you will change the
    output of this routine but there will be no warning.

    Examples
    ========

    >>> from sympy.utilities.iterables import uniq

    """
    # 初始化数据序列
    >>> dat = [1, 4, 1, 5, 4, 2, 1, 2]
    # 检查 uniq 函数返回的结果类型是否为 list 或 tuple
    >>> type(uniq(dat)) in (list, tuple)
    # 应返回 False，因为 uniq 返回的是生成器对象，而不是 list 或 tuple
    False

    # 检查 uniq 函数对数据序列的唯一化操作
    >>> list(uniq(dat))
    # 应返回序列中的唯一值列表
    [1, 4, 5, 2]
    # 使用生成器表达式作为参数调用 uniq 函数，结果应该与上一行相同
    >>> list(uniq(x for x in dat))
    # 应返回序列中的唯一值列表
    [1, 4, 5, 2]
    # 对嵌套列表调用 uniq 函数，返回嵌套列表中的唯一子列表
    >>> list(uniq([[1], [2, 1], [1]]))
    # 应返回嵌套列表中的唯一子列表
    [[1], [2, 1]]
    """
    # 尝试获取序列的长度，若无法获取则设为 None
    try:
        n = len(seq)
    except TypeError:
        n = None
    
    # 定义检查函数，用于验证迭代过程中序列大小是否改变
    def check():
        # 检查序列大小是否在迭代过程中改变；
        # 若 n 不为 None，则对象不支持大小变化，如迭代器无法更改
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    
    try:
        # 初始化一个集合用于存储已经出现过的元素
        seen = set()
        # 初始化结果列表，如果未提供则为空列表
        result = result or []
        # 枚举序列中的元素及其索引
        for i, s in enumerate(seq):
            # 如果当前元素 s 不在 seen 中，则将其添加到 seen 中，并 yield 返回
            if not (s in seen or seen.add(s)):
                yield s
                # 检查序列大小是否变化
                check()
    except TypeError:
        # 处理在不支持直接迭代的情况下的唯一化操作
        if s not in result:
            yield s
            # 检查序列大小是否变化
            check()
            # 将新的唯一元素添加到结果列表中
            result.append(s)
        # 若序列支持索引访问，则递归调用 uniq 函数处理剩余部分
        if hasattr(seq, '__getitem__'):
            yield from uniq(seq[i + 1:], result)
        else:
            yield from uniq(seq, result)
# 定义生成贝尔排列的函数，生成所有长度为 n 的排列，每个排列与前一个只有一对相邻元素交换位置
def generate_bell(n):
    """Return permutations of [0, 1, ..., n - 1] such that each permutation
    differs from the last by the exchange of a single pair of neighbors.
    The ``n!`` permutations are returned as an iterator. In order to obtain
    the next permutation from a random starting permutation, use the
    ``next_trotterjohnson`` method of the Permutation class (which generates
    the same sequence in a different manner).

    Examples
    ========

    >>> from itertools import permutations
    >>> from sympy.utilities.iterables import generate_bell
    >>> from sympy import zeros, Matrix

    This is the sort of permutation used in the ringing of physical bells,
    and does not produce permutations in lexicographical order. Rather, the
    permutations differ from each other by exactly one inversion, and the
    position at which the swapping occurs varies periodically in a simple
    fashion. Consider the first few permutations of 4 elements generated
    by ``permutations`` and ``generate_bell``:

    >>> list(permutations(range(4)))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    >>> list(generate_bell(4))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]

    Notice how the 2nd and 3rd lexicographical permutations have 3 elements
    out of place whereas each "bell" permutation always has only two
    elements out of place relative to the previous permutation (and so the
    signature (+/-1) of a permutation is opposite of the signature of the
    previous permutation).

    How the position of inversion varies across the elements can be seen
    by tracing out where the largest number appears in the permutations:

    >>> m = zeros(4, 24)
    >>> for i, p in enumerate(generate_bell(4)):
    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero
    >>> m.print_nonzero('X')
    [XXX  XXXXXX  XXXXXX  XXX]
    [XX XX XXXX XX XXXX XX XX]
    [X XXXX XX XXXX XX XXXX X]
    [ XXXXXX  XXXXXX  XXXXXX ]

    See Also
    ========

    sympy.combinatorics.permutations.Permutation.next_trotterjohnson

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Method_ringing

    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018

    .. [3] https://web.archive.org/web/20160313023044/http://programminggeeks.com/bell-algorithm-for-permutation/

    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

    .. [5] Generating involutions, derangements, and relatives by ECO
           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010

    """
    # 将 n 转换为整数
    n = as_int(n)
    # 如果 n 小于 1，则抛出值错误异常
    if n < 1:
        raise ValueError('n must be a positive integer')
    # 如果 n 等于 1，生成并返回唯一的长度为 1 的排列 (0,)
    if n == 1:
        yield (0,)
    # 如果 n 等于 2，生成并返回长度为 2 的两个排列 (0, 1) 和 (1, 0)
    elif n == 2:
        yield (0, 1)
        yield (1, 0)
    # 如果 n 等于 3，生成并返回长度为 3 的六个排列
    elif n == 3:
        yield from [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    else:
        m = n - 1
        op = [0] + [-1]*m
        l = list(range(n))
        while True:
            yield tuple(l)
            # find biggest element with op
            big = None, -1  # idx, value
            for i in range(n):
                # 检查是否有激活的操作符，并找到当前位置的最大元素
                if op[i] and l[i] > big[1]:
                    big = i, l[i]
            i, _ = big
            if i is None:
                break  # 没有剩余的操作符了
            # 根据指示方向与相邻元素交换
            j = i + op[i]
            l[i], l[j] = l[j], l[i]
            op[i], op[j] = op[j], op[i]
            # 如果交换后的元素位于末尾，或者与同方向的相邻元素比较后更大，则关闭操作符
            if j == 0 or j == m or l[j + op[j]] > l[j]:
                op[j] = 0
            # 左侧任何更大的元素得到 +1 的操作符
            for i in range(j):
                if l[i] > l[j]:
                    op[i] = 1
            # 右侧任何更大的元素得到 -1 的操作符
            for i in range(j + 1, n):
                if l[i] > l[j]:
                    op[i] = -1
def generate_involutions(n):
    """
    Generates involutions.

    An involution is a permutation that when multiplied
    by itself equals the identity permutation. In this
    implementation the involutions are generated using
    Fixed Points.

    Alternatively, an involution can be considered as
    a permutation that does not contain any cycles with
    a length that is greater than two.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_involutions
    >>> list(generate_involutions(3))
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    >>> len(list(generate_involutions(4)))
    10

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PermutationInvolution.html

    """
    idx = list(range(n))  # 创建一个包含0到n-1的列表，作为索引
    for p in permutations(idx):  # 遍历所有索引的排列组合
        for i in idx:  # 遍历索引列表
            if p[p[i]] != i:  # 检查是否存在非自身的固定点
                break  # 如果存在，则退出内层循环
        else:
            yield p  # 如果没有非自身的固定点，则生成该排列作为一个生成器的输出


def multiset_derangements(s):
    """Generate derangements of the elements of s *in place*.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_derangements, uniq

    Because the derangements of multisets (not sets) are generated
    in place, copies of the return value must be made if a collection
    of derangements is desired or else all values will be the same:

    >>> list(uniq([i for i in multiset_derangements('1233')]))
    [[None, None, None, None]]
    >>> [i.copy() for i in multiset_derangements('1233')]
    [['3', '3', '1', '2'], ['3', '3', '2', '1']]
    >>> [''.join(i) for i in multiset_derangements('1233')]
    ['3312', '3321']
    """
    from sympy.core.sorting import ordered  # 导入排序函数
    # 创建可哈希元素的多重集合字典，或将元素重新映射为整数
    try:
        ms = multiset(s)  # 尝试创建s的多重集合对象
    except TypeError:
        # 为每个元素分配一个规范整数值
        key = dict(enumerate(ordered(uniq(s))))  # 创建元素与整数的映射关系字典
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)  # 将s中的元素映射为整数并存储在h中
                    break
        for i in multiset_derangements(h):  # 递归调用生成整数列表的错排
            yield [key[j] for j in i]  # 将整数列表转换回原始元素的列表
        return

    mx = max(ms.values())  # 计算多重集合中任意元素的最大重复次数
    n = len(s)  # 计算元素的总数

    ## special cases

    # 1) one element has more than half the total cardinality of s: no
    # derangements are possible.
    if mx*2 > n:
        return  # 如果某个元素的重复次数超过总数的一半，则没有错排可能

    # 2) all elements appear once: singletons
    if len(ms) == n:
        yield from _set_derangements(s)  # 如果所有元素都只出现一次，则生成单个元素的错排
        return

    # find the first element that is repeated the most to place
    # in the following two special cases where the selection
    # is unambiguous: either there are two elements with multiplicity
    # of mx or else there is only one with multiplicity mx
    for M in ms:
        if ms[M] == mx:
            break  # 找到重复次数最多的第一个元素M

    inonM = [i for i in range(n) if s[i] != M]  # 非M元素的位置列表
    iM = [i for i in range(n) if s[i] == M]  # M元素的位置列表
    rv = [None]*n  # 创建一个长度为n的None列表作为结果向量
    # 3) half are the same
    if 2*mx == n:
        # M goes into non-M locations
        for i in inonM:
            rv[i] = M
        # permutations of non-M go to M locations
        for p in multiset_permutations([s[i] for i in inonM]):
            for i, pi in zip(iM, p):
                rv[i] = pi
            yield rv
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return


    # 4) single repeat covers all but 1 of the non-repeats:
    # if there is one repeat then the multiset of the values
    # of ms would be {mx: 1, 1: n - mx}, i.e. there would
    # be n - mx + 1 values with the condition that n - 2*mx = 1
    if n - 2*mx == 1 and len(ms.values()) == n - mx + 1:
        for i, i1 in enumerate(inonM):
            ifill = inonM[:i] + inonM[i+1:]
            # Fill non-repeat positions with M
            for j in ifill:
                rv[j] = M
            # Generate permutations for the positions of the repeat
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for j, pi in zip(iM, p):
                    rv[j] = pi
                k = i1
                # Swap elements to generate permutations
                for j in iM:
                    rv[j], rv[k] = rv[k], rv[j]
                    yield rv
                    k = j
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return


    ## general case is handled with 3 helpers:
    #    1) `finish_derangements` will place the last two elements
    #       which have arbitrary multiplicities, e.g. for multiset
    #       {c: 3, a: 2, b: 2}, the last two elements are a and b
    #    2) `iopen` will tell where a given element can be placed
    #    3) `do` will recursively place elements into subsets of
    #        valid locations
    def finish_derangements():
        """Place the last two elements into the partially completed
        derangement, and yield the results.
        """
        
        a = take[1][0]  # 取倒数第二个元素
        a_ct = take[1][1]  # 倒数第二个元素的数量
        b = take[0][0]  # 取最后一个待放置的元素
        b_ct = take[0][1]  # 最后一个待放置元素的数量

        # 将 rv 中尚未分配的索引按照 s 中元素的情况分为三类
        forced_a = []  # 必须放置元素 a 的位置
        forced_b = []  # 必须放置元素 b 的位置
        open_free = []  # 可以放置元素 a 或 b 的位置
        for i in range(len(s)):
            if rv[i] is None:
                if s[i] == a:
                    forced_b.append(i)
                elif s[i] == b:
                    forced_a.append(i)
                else:
                    open_free.append(i)

        if len(forced_a) > a_ct or len(forced_b) > b_ct:
            # 如果无法生成一个完全不重复的排列，则返回
            return

        # 将 forced_a 中的位置放置元素 a
        for i in forced_a:
            rv[i] = a
        # 将 forced_b 中的位置放置元素 b
        for i in forced_b:
            rv[i] = b

        # 对于 open_free 中的位置，组合放置元素 a，其余位置放置元素 b
        for a_place in combinations(open_free, a_ct - len(forced_a)):
            for a_pos in a_place:
                rv[a_pos] = a
            for i in open_free:
                if rv[i] is None:  # 对于未放置元素 a 的位置，放置元素 b
                    rv[i] = b
                yield rv  # 生成当前排列结果
                # 清理/撤销最后的放置
                for i in open_free:
                    rv[i] = None

        # 额外的清理工作 - 清空 forced_a 和 forced_b 中的位置
        for i in forced_a:
            rv[i] = None
        for i in forced_b:
            rv[i] = None

    def iopen(v):
        # 返回可以放置元素 v 的索引位置：
        # 位置必须为空且在 s 中对应位置不含有 v
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if j == 1:
            # 使用特殊方法处理最后两个元素（无论其重复多少次）
            yield from finish_derangements()
        else:
            # 将 M 的 mx 个元素放置到可以放置的位置子集中
            M, mx = take[j]
            for i in combinations(iopen(M), mx):
                # 放置 M
                for ii in i:
                    rv[ii] = M
                # 递归放置下一个元素
                yield from do(j - 1)
                # 将放置了 M 的位置再次标记为可用于放置其他元素
                for ii in i:
                    rv[ii] = None

    # 按照多重性的递减顺序处理元素
    take = sorted(ms.items(), key=lambda x:(x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None]*n
def random_derangement(t, choice=None, strict=True):
    """Return a list of elements in which none are in the same positions
    as they were originally. If an element fills more than half of the positions
    then an error will be raised since no derangement is possible. To obtain
    a derangement of as many items as possible--with some of the most numerous
    remaining in their original positions--pass `strict=False`. To produce a
    pseudorandom derangment, pass a pseudorandom selector like `choice` (see
    below).

    Examples
    ========

    >>> from sympy.utilities.iterables import random_derangement
    >>> t = 'SymPy: a CAS in pure Python'
    >>> d = random_derangement(t)
    >>> all(i != j for i, j in zip(d, t))
    True

    A predictable result can be obtained by using a pseudorandom
    generator for the choice:

    >>> from sympy.core.random import seed, choice as c
    >>> seed(1)
    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert len(set(d)) != 1  # we got different values

    By reseeding, the same sequence can be obtained:

    >>> seed(1)
    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert d == d2
    """
    if choice is None:
        import secrets
        choice = secrets.choice

    def shuffle(rv):
        '''Knuth shuffle'''
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            rv[i], rv[j] = rv[j], rv[i]

    def pick(rv, n):
        '''shuffle rv and return the first n values'''
        shuffle(rv)
        return rv[:n]

    # Calculate the multiplicity of each character in `t`
    ms = multiset(t)
    tot = len(t)
    
    # Sort characters based on their multiplicity
    ms = sorted(ms.items(), key=lambda x: x[1])

    # Determine the most plentiful element and its count
    M, mx = ms[-1]
    n = len(t)
    
    # Calculate the difference in positions needed for a derangement
    xs = 2 * mx - tot
    
    # If there are excess positions for the most plentiful element
    if xs > 0:
        # If strict mode is enabled, raise an error
        if strict:
            raise ValueError('no derangement possible')
        
        # Otherwise, choose indices where the most plentiful element resides
        opts = [i for (i, c) in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        
        # Remove elements that must stay in place and recurse to derange the rest
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        
        # Insert the elements that stayed in place back into their original positions
        for i in stay:
            rv.insert(i, ms[-1][0])
        
        return ''.join(rv) if type(t) is str else rv
    
    # Calculate a normal derangement when no excess positions are available
    if n == len(ms):
        # Approximately 1/3 of the attempts will succeed in creating a derangement
        rv = list(t)
        while True:
            shuffle(rv)
            if all(i != j for i, j in zip(rv, t)):
                break
    else:
      # 一般情况处理分支
        rv = [None]*n
        while True:
            j = 0
            while j > -len(ms):  # 从最多到最少处理元素
                j -= 1
                e, c = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break # 重新尝试
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv
# 定义一个生成无重复元素的集合s的排列的生成器函数
def _set_derangements(s):
    """
    yield derangements of items in ``s`` which are assumed to contain
    no repeated elements
    """
    # 如果集合s的长度小于2，直接返回
    if len(s) < 2:
        return
    # 如果集合s的长度为2，生成一种两个元素的错位排列并返回
    if len(s) == 2:
        yield [s[1], s[0]]
        return
    # 如果集合s的长度为3，生成两种三个元素的错位排列并返回
    if len(s) == 3:
        yield [s[1], s[2], s[0]]
        yield [s[2], s[0], s[1]]
        return
    # 使用排列函数permutations生成集合s的所有排列
    for p in permutations(s):
        # 检查生成的排列p与原始集合s是否有相同位置的元素相同
        if not any(i == j for i, j in zip(p, s)):
            # 如果没有相同位置的元素相同，则生成该排列的列表形式并返回
            yield list(p)


# 定义一个生成集合s的唯一错位排列的生成器函数
def generate_derangements(s):
    """
    Return unique derangements of the elements of iterable ``s``.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 2]))
    [[2, 2, 0, 1], [2, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========

    sympy.functions.combinatorial.factorials.subfactorial

    """
    # 如果集合s中没有重复元素，则使用_set_derangements生成唯一错位排列
    if not has_dups(s):
        yield from _set_derangements(s)
    else:
        # 如果集合s中有重复元素，则使用multiset_derangements生成错位排列
        for p in multiset_derangements(s):
            yield list(p)


# 定义一个生成项链（necklaces）的生成器函数，该项链可以选择是否允许翻转
def necklaces(n, k, free=False):
    """
    A routine to generate necklaces that may (free=True) or may not
    (free=False) be turned over to be viewed. The "necklaces" returned
    are comprised of ``n`` integers (beads) with ``k`` different
    values (colors). Only unique necklaces are returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import necklaces, bracelets
    >>> def show(s, i):
    ...     return ''.join(s[j] for j in i)

    The "unrestricted necklace" is sometimes also referred to as a
    "bracelet" (an object that can be turned over, a sequence that can
    be reversed) and the term "necklace" is used to imply a sequence
    that cannot be reversed. So ACB == ABC for a bracelet (rotate and
    reverse) while the two are different for a necklace since rotation
    alone cannot make the two sequences the same.

    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)

    >>> B = [show('ABC', i) for i in bracelets(3, 3)]
    >>> N = [show('ABC', i) for i in necklaces(3, 3)]
    >>> set(N) - set(B)
    {'ACB'}

    >>> list(necklaces(4, 2))
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

    >>> [show('.o', i) for i in bracelets(4, 2)]
    ['....', '...o', '..oo', '.o.o', '.ooo', 'oooo']

    References
    ==========

    .. [1] https://mathworld.wolfram.com/Necklace.html

    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,
        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;
        https://doi.org/10.1016/0196-6774(92)90047-G

    """
    # FKM算法生成项链
    if k == 0 and n > 0:
        return
    # 初始化长度为n的全零列表a
    a = [0]*n
    # 返回全零项链的元组形式
    yield tuple(a)
    # 如果项链长度为0，直接返回
    if n == 0:
        return
    # 进入无限循环，生成器函数会在每次调用时继续执行这个循环
    while True:
        # 将变量 i 初始化为 n - 1
        i = n - 1
        # 当 a[i] 等于 k - 1 时，向前递减 i，直到不再满足条件为止
        while a[i] == k - 1:
            i -= 1
            # 如果 i 变为 -1，即所有的 a[i] 都等于 k - 1，则返回，结束函数执行
            if i == -1:
                return
        # 将 a[i] 的值加一
        a[i] += 1
        # 将数组 a 中的特定范围内元素设为与 a[j] 相同的值
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        # 检查条件，如果 n 能被 (i + 1) 整除，并且不是自由排列（如果不自由或所有 a 都小于等于 a[j] 反转后的组合），则执行以下操作
        if n % (i + 1) == 0 and (not free or all(a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1))):
            # 不需要测试 j = n - 1。
            # 产生一个元组，作为生成器的输出
            yield tuple(a)
def bracelets(n, k):
    """Wrapper to necklaces to return a free (unrestricted) necklace."""
    # 调用 necklaces 函数，返回一个自由的项链（无限制项链）
    return necklaces(n, k, free=True)


def generate_oriented_forest(n):
    """
    This algorithm generates oriented forests.

    An oriented graph is a directed graph having no symmetric pair of directed
    edges. A forest is an acyclic graph, i.e., it has no cycles. A forest can
    also be described as a disjoint union of trees, which are graphs in which
    any two vertices are connected by exactly one simple path.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_oriented_forest
    >>> list(generate_oriented_forest(4))
    [[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 2, 1], [0, 1, 2, 0], \
    [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]

    References
    ==========

    .. [1] T. Beyer and S.M. Hedetniemi: constant time generation of
           rooted trees, SIAM J. Computing Vol. 9, No. 4, November 1980

    .. [2] https://stackoverflow.com/questions/1633833/oriented-forest-taocp-algorithm-in-python

    """
    # 初始化一个 P 列表，从 -1 到 n，用于存储当前状态的森林结构
    P = list(range(-1, n))
    while True:
        # 生成当前 P 列表中除去第一个元素的子列表作为森林的一种表示方式，并返回
        yield P[1:]
        if P[n] > 0:
            P[n] = P[P[n]]
        else:
            # 寻找可以变换的位置，更新 P 列表以生成下一个森林结构
            for p in range(n - 1, 0, -1):
                if P[p] != 0:
                    target = P[p] - 1
                    for q in range(p - 1, 0, -1):
                        if P[q] == target:
                            break
                    offset = p - q
                    for i in range(p, n + 1):
                        P[i] = P[i - offset]
                    break
            else:
                break


def minlex(seq, directed=True, key=None):
    r"""
    Return the rotation of the sequence in which the lexically smallest
    elements appear first, e.g. `cba \rightarrow acb`.

    The sequence returned is a tuple, unless the input sequence is a string
    in which case a string is returned.

    If ``directed`` is False then the smaller of the sequence and the
    reversed sequence is returned, e.g. `cba \rightarrow abc`.

    If ``key`` is not None then it is used to extract a comparison key from each element in iterable.

    Examples
    ========

    >>> from sympy.combinatorics.polyhedron import minlex
    >>> minlex((1, 2, 0))
    (0, 1, 2)
    >>> minlex((1, 0, 2))
    (0, 2, 1)
    >>> minlex((1, 0, 2), directed=False)
    (0, 1, 2)

    >>> minlex('11010011000', directed=True)
    '00011010011'
    >>> minlex('11010011000', directed=False)
    '00011001011'

    >>> minlex(('bb', 'aaa', 'c', 'a'))
    ('a', 'bb', 'aaa', 'c')
    >>> minlex(('bb', 'aaa', 'c', 'a'), key=len)
    ('c', 'a', 'bb', 'aaa')

    """
    # 导入必要的函数
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    # 找到序列的最小旋转，使得字典序最小的元素排在最前面
    best = rotate_left(seq, least_rotation(seq, key=key))
    if not directed:
        rseq = seq[::-1]
        rbest = rotate_left(rseq, least_rotation(rseq, key=key))
        # 如果不考虑方向，则比较正序和逆序的结果，选取最小的那个
        best = min(best, rbest, key=key)
    # 如果 seq 不是字符串，则将 best 转换为元组后返回，否则直接返回 best。
    return tuple(best) if not isinstance(seq, str) else best
# 将序列 `seq` 按照给定的比较操作符 `op` 分组成连续元素相同的子列表
def runs(seq, op=gt):
    cycles = []  # 初始化空列表用于存放分组后的子列表
    seq = iter(seq)  # 将输入的序列转换为迭代器
    try:
        run = [next(seq)]  # 尝试获取序列的第一个元素作为起始值
    except StopIteration:
        return []  # 如果序列为空，则直接返回空列表
    while True:
        try:
            ei = next(seq)  # 获取迭代器的下一个元素
        except StopIteration:
            break  # 如果迭代器已经结束，则退出循环
        if op(ei, run[-1]):  # 如果当前元素和当前运行中列表的最后一个元素满足比较操作
            run.append(ei)  # 将当前元素添加到当前运行中列表
            continue  # 继续处理下一个元素
        else:
            cycles.append(run)  # 如果不满足比较操作，则将当前运行中列表添加到结果列表中
            run = [ei]  # 并重新开始一个新的运行中列表
    if run:
        cycles.append(run)  # 处理完所有元素后，将最后一个运行中列表添加到结果列表中
    return cycles  # 返回分组后的结果列表


# 返回序列 `l` 的长度为 `n` 的所有可能分区的生成器
def sequence_partitions(l, n, /):
    if n == 1 and l:
        yield [l]  # 如果 n 为 1 并且 l 非空，则直接将 l 作为一个分区返回
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield [l[:i]] + part  # 递归地生成剩余部分的分区，并与当前部分组合后返回


# 返回序列 `l` 的长度为 `n` 的所有可能分区的生成器，包括空序列
def sequence_partitions_empty(l, n, /):
    # 与 `sequence_partitions` 类似，但是允许空分区
    # l : Sequence[T]
    #     A sequence of any Python objects (can be possibly empty)
    # n : int
    #     A positive integer
    # 
    # Yields
    # ======
    # out : list[Sequence[T]]
    #     A list of sequences where each sequence's concatenation equals `l`.
    #     The type of `out` conforms to the type of `l`.
    # 
    # Examples
    # ========
    # 
    # >>> from sympy.utilities.iterables import sequence_partitions_empty
    # >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):
    # ...     print(out)
    # [[], [1, 2, 3, 4]]
    # [[1], [2, 3, 4]]
    # [[1, 2], [3, 4]]
    # [[1, 2, 3], [4]]
    # [[1, 2, 3, 4], []]
    # 
    # See Also
    # ========
    # 
    # sequence_partitions

    # 如果 n 小于 1，则直接返回（不产生任何输出）
    if n < 1:
        return
    # 如果 n 等于 1，则生成一个包含整个 l 的列表并 yield
    if n == 1:
        yield [l]
        return
    # 遍历 l 的每个可能的分割点 i
    for i in range(0, len(l) + 1):
        # 递归调用 sequence_partitions_empty 来生成 n-1 部分的所有可能分割
        for part in sequence_partitions_empty(l[i:], n - 1):
            # 将当前分割点 i 前后两部分合并，并 yield
            yield [l[:i]] + part
def kbins(l, k, ordered=None):
    """
    Return sequence ``l`` partitioned into ``k`` bins.

    Examples
    ========

    The default is to give the items in the same order, but grouped
    into k partitions without any reordering:

    >>> from sympy.utilities.iterables import kbins
    >>> for p in kbins(list(range(5)), 2):
    ...     print(p)
    ...
    [[0], [1, 2, 3, 4]]
    [[0, 1], [2, 3, 4]]
    [[0, 1, 2], [3, 4]]
    [[0, 1, 2, 3], [4]]

    The ``ordered`` flag is either None (to give the simple partition
    of the elements) or is a 2 digit integer indicating whether the order of
    the bins and the order of the items in the bins matters. Given::

        A = [[0], [1, 2]]
        B = [[1, 2], [0]]
        C = [[2, 1], [0]]
        D = [[0], [2, 1]]

    the following values for ``ordered`` have the shown meanings::

        00 means A == B == C == D
        01 means A == B
        10 means A == D
        11 means A == A

    >>> for ordered_flag in [None, 0, 1, 10, 11]:
    ...     print('ordered = %s' % ordered_flag)
    ...     for p in kbins(list(range(3)), 2, ordered=ordered_flag):
    ...         print('     %s' % p)
    ...
    ordered = None
         [[0], [1, 2]]
         [[0, 1], [2]]
    ordered = 0
         [[0, 1], [2]]
         [[0, 2], [1]]
         [[0], [1, 2]]
    ordered = 1
         [[0], [1, 2]]
         [[0], [2, 1]]
         [[1], [0, 2]]
         [[1], [2, 0]]
         [[2], [0, 1]]
         [[2], [1, 0]]
    ordered = 10
         [[0, 1], [2]]
         [[2], [0, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[0], [1, 2]]
         [[1, 2], [0]]
    ordered = 11
         [[0], [1, 2]]
         [[0, 1], [2]]
         [[0], [2, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[1, 0], [2]]
         [[1], [2, 0]]
         [[1, 2], [0]]
         [[2], [0, 1]]
         [[2, 0], [1]]
         [[2], [1, 0]]
         [[2, 1], [0]]

    See Also
    ========

    partitions, multiset_partitions

    """
    if ordered is None:
        # 如果 ordered 参数为 None，则调用 sequence_partitions 函数生成简单分区结果
        yield from sequence_partitions(l, k)
    elif ordered == 11:
        # 如果 ordered 参数为 11，则生成所有可能的多重集排列，并对每个排列调用 sequence_partitions 函数
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from sequence_partitions(pl, k)
    elif ordered == 00:
        # 如果 ordered 参数为 00，则调用 multiset_partitions 函数生成多重集分区结果
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        # 如果 ordered 参数为 10，则先调用 multiset_partitions 函数生成多重集分区结果，
        # 然后对每个分区调用 permutations 函数生成排列，并将结果作为生成器的一部分
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        # 如果 ordered 参数为 1，则首先计算 partitions 函数得到所有可能的分区方式，然后对每种分区方式
        # 调用 multiset_permutations 函数生成所有可能的多重集排列，并按照分区方式组织结果
        for kgot, p in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                # 按照分区的大小和数量对多重集排列进行切片，生成最终的分区结果
                for size, multiplicity in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i: j])
                        i = j
                yield rv
    else:
        # 如果 ordered 不是 '00', '01', '10' 或 '11' 中的一个，抛出 ValueError 异常
        raise ValueError(
            'ordered must be one of 00, 01, 10 or 11, not %s' % ordered)
# 定义一个函数，用于生成所有可能的非零元素符号排列的迭代器
def permute_signs(t):
    """Return iterator in which the signs of non-zero elements
    of t are permuted.

    Examples
    ========

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2)]
    """
    # 使用 product 函数生成所有可能的符号组合
    for signs in product(*[(1, -1)]*(len(t) - t.count(0))):
        # 将生成的符号组合转换为列表，并根据 t 中的非零元素对应位置修改其符号
        signs = list(signs)
        # 生成新的元组，其中非零元素的符号已经被调整
        yield type(t)([i*signs.pop() if i else i for i in t])


# 定义一个函数，返回符号和元素顺序都进行排列且所有返回值唯一的迭代器
def signed_permutations(t):
    """Return iterator in which the signs of non-zero elements
    of t and the order of the elements are permuted and all
    returned values are unique.

    Examples
    ========

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),
    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),
    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),
    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),
    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]
    """
    # 返回一个生成器表达式，对 multiset_permutations 返回的每个元素调用 permute_signs
    return (type(t)(i) for j in multiset_permutations(t)
        for i in permute_signs(j))


# 定义一个函数，返回一个生成器，依次生成将序列 s 左旋或右旋一个位置后的所有序列
def rotations(s, dir=1):
    """Return a generator giving the items in s as list where
    each subsequent list has the items rotated to the left (default)
    or right (``dir=-1``) relative to the previous list.

    Examples
    ========

    >>> from sympy import rotations
    >>> list(rotations([1,2,3]))
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    >>> list(rotations([1,2,3], -1))
    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    """
    # 将输入序列转换为列表
    seq = list(s)
    # 生成每次旋转后的序列，并返回生成器
    for i in range(len(seq)):
        yield seq
        seq = rotate_left(seq, dir)


# 定义一个函数，实现一个“round-robin”（交替）迭代器的生成
def roundrobin(*iterables):
    """roundrobin recipe taken from itertools documentation:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe credited to George Sakkis
    """
    # 使用 cycle 和 islice 函数创建一个“round-robin”生成器
    nexts = cycle(iter(it).__next__ for it in iterables)

    pending = len(iterables)
    while pending:
        try:
            # 依次调用每个迭代器的 __next__ 方法
            for nxt in nexts:
                yield nxt()
        except StopIteration:
            # 处理迭代器耗尽的情况
            pending -= 1
            nexts = cycle(islice(nexts, pending))


class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to
    return true when iterable() is called on its instances because
    calling list() on the instance, for example, would result in
    an infinite loop.
    """
    pass


# 定义一个函数，判断对象是否为 SymPy 可迭代对象
def iterable(i, exclude=(str, dict, NotIterable)):
    """
    Return a boolean indicating whether ``i`` is SymPy iterable.
    True also indicates that the iterator is finite, e.g. you can
    call list(...) on the instance.

    When SymPy is working with iterables, it is almost always assuming
    that the iterable is not a string or a mapping, so those are excluded
    by default. If you want a pure Python definition, make exclude=None. To
    """
    exclude multiple items, pass them as a tuple.
    
    You can also set the _iterable attribute to True or False on your class,
    which will override the checks here, including the exclude test.
    
    As a rule of thumb, some SymPy functions use this to check if they should
    recursively map over an object. If an object is technically iterable in
    the Python sense but does not desire this behavior (e.g., because its
    iteration is not finite, or because iteration might induce an unwanted
    computation), it should disable it by setting the _iterable attribute to False.
    
    See also: is_sequence
    
    Examples
    ========
    
    >>> from sympy.utilities.iterables import iterable
    >>> from sympy import Tuple
    >>> things = [[1], (1,), set([1]), Tuple(1), (j for j in [1, 2]), {1:2}, '1', 1]
    >>> for i in things:
    ...     print('%s %s' % (iterable(i), type(i)))
    True <... 'list'>
    True <... 'tuple'>
    True <... 'set'>
    True <class 'sympy.core.containers.Tuple'>
    True <... 'generator'>
    False <... 'dict'>
    False <... 'str'>
    False <... 'int'>
    
    >>> iterable({}, exclude=None)
    True
    >>> iterable({}, exclude=str)
    True
    >>> iterable("no", exclude=str)
    False
    
    """
    # 检查对象是否可以迭代
    
    # 如果对象有 '_iterable' 属性，则返回该属性的值
    if hasattr(i, '_iterable'):
        return i._iterable
    # 尝试迭代对象，如果抛出 TypeError 异常，则对象不可迭代，返回 False
    try:
        iter(i)
    except TypeError:
        return False
    # 如果传入了 exclude 参数，并且对象是 exclude 指定的类型，则返回 False
    if exclude:
        return not isinstance(i, exclude)
    # 如果没有 exclude 参数，或者对象不是 exclude 指定的类型，则返回 True
    return True
# 定义一个函数，判断参数 i 是否符合 SymPy 中序列的定义
# 如果 include 参数被设置为某种类型，将该类型也包含在序列的定义中
def is_sequence(i, include=None):
    # 检查参数 i 是否具有 __getitem__ 属性，并且是可迭代的（iterable）
    return (hasattr(i, '__getitem__') and
            iterable(i) or
            bool(include) and
            isinstance(i, include))


# 标记函数 postorder_traversal 已经被废弃，推荐使用 sympy.postorder_traversal
# 提供了一个示例用法，指引用户如何迁移
@deprecated(
    """
    Using postorder_traversal from the sympy.utilities.iterables submodule is
    deprecated.

    Instead, use postorder_traversal from the top-level sympy namespace, like

        sympy.postorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def postorder_traversal(node, keys=None):
    # 从 sympy.core.traversal 模块导入 postorder_traversal 函数，并调用
    from sympy.core.traversal import postorder_traversal as _postorder_traversal
    return _postorder_traversal(node, keys=keys)


# 标记函数 interactive_traversal 已经被废弃，推荐使用 sympy.interactive_traversal
# 提供了一个示例用法，指引用户如何迁移
@deprecated(
    """
    Using interactive_traversal from the sympy.utilities.iterables submodule
    is deprecated.

    Instead, use interactive_traversal from the top-level sympy namespace,
    like

        sympy.interactive_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def interactive_traversal(expr):
    # 从 sympy.interactive.traversal 模块导入 interactive_traversal 函数，并调用
    from sympy.interactive.traversal import interactive_traversal as _interactive_traversal
    return _interactive_traversal(expr)


# 标记函数 default_sort_key 已经被废弃，推荐直接从 sympy 导入 default_sort_key
# 提供了一个示例用法，指引用户如何迁移
@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-sympy-core-compatibility",
)
def default_sort_key(*args, **kwargs):
    # 从 sympy 模块导入 default_sort_key 函数，并调用
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)


# 标记函数 ordered 已经被废弃，推荐直接从 sympy 导入 ordered
# 提供了一个示例用法，指引用户如何迁移
@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-sympy-core-compatibility",
)
def ordered(*args, **kwargs):
    # 从 sympy 模块导入 ordered 函数，并调用
    from sympy import ordered as _ordered
    return _ordered(*args, **kwargs)
```