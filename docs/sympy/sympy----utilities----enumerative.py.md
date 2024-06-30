# `D:\src\scipysrc\sympy\sympy\utilities\enumerative.py`

```
"""
Algorithms and classes to support enumerative combinatorics.

Currently just multiset partitions, but more could be added.

Terminology (following Knuth, algorithm 7.1.2.5M TAOCP)
*multiset* aaabbcccc has a *partition* aaabc | bccc

The submultisets, aaabc and bccc of the partition are called
*parts*, or sometimes *vectors*.  (Knuth notes that multiset
partitions can be thought of as partitions of vectors of integers,
where the ith element of the vector gives the multiplicity of
element i.)

The values a, b and c are *components* of the multiset.  These
correspond to elements of a set, but in a multiset can be present
with a multiplicity greater than 1.

The algorithm deserves some explanation.

Think of the part aaabc from the multiset above.  If we impose an
ordering on the components of the multiset, we can represent a part
with a vector, in which the value of the first element of the vector
corresponds to the multiplicity of the first component in that
part. Thus, aaabc can be represented by the vector [3, 1, 1].  We
can also define an ordering on parts, based on the lexicographic
ordering of the vector (leftmost vector element, i.e., the element
with the smallest component number, is the most significant), so
that [3, 1, 1] > [3, 1, 0] and [3, 1, 1] > [2, 1, 4].  The ordering
on parts can be extended to an ordering on partitions: First, sort
the parts in each partition, left-to-right in decreasing order. Then
partition A is greater than partition B if A's leftmost/greatest
part is greater than B's leftmost part.  If the leftmost parts are
equal, compare the second parts, and so on.

In this ordering, the greatest partition of a given multiset has only
one part.  The least partition is the one in which the components
are spread out, one per part.

The enumeration algorithms in this file yield the partitions of the
argument multiset in decreasing order.  The main data structure is a
stack of parts, corresponding to the current partition.  An
important invariant is that the parts on the stack are themselves in
decreasing order.  This data structure is decremented to find the
next smaller partition.  Most often, decrementing the partition will
only involve adjustments to the smallest parts at the top of the
stack, much as adjacent integers *usually* differ only in their last
few digits.

Knuth's algorithm uses two main operations on parts:

Decrement - change the part so that it is smaller in the
  (vector) lexicographic order, but reduced by the smallest amount possible.
  For example, if the multiset has vector [5,
  3, 1], and the bottom/greatest part is [4, 2, 1], this part would
  decrement to [4, 2, 0], while [4, 0, 0] would decrement to [3, 3,
  1].  A singleton part is never decremented -- [1, 0, 0] is not
  decremented to [0, 3, 1].  Instead, the decrement operator needs
  to fail for this case.  In Knuth's pseudocode, the decrement
  operator is step m5.
"""
"""
Spread unallocated multiplicity - Once a part has been decremented,
  it cannot be the rightmost part in the partition.  There is some
  multiplicity that has not been allocated, and new parts must be
  created above it in the stack to use up this multiplicity.  To
  maintain the invariant that the parts on the stack are in
  decreasing order, these new parts must be less than or equal to
  the decremented part.
  For example, if the multiset is [5, 3, 1], and its most
  significant part has just been decremented to [5, 3, 0], the
  spread operation will add a new part so that the stack becomes
  [[5, 3, 0], [0, 0, 1]].  If the most significant part (for the
  same multiset) has been decremented to [2, 0, 0] the stack becomes
  [[2, 0, 0], [2, 0, 0], [1, 3, 1]].  In the pseudocode, the spread
  operation for one part is step m2.  The complete spread operation
  is a loop of steps m2 and m3.

In order to facilitate the spread operation, Knuth stores, for each
component of each part, not just the multiplicity of that component
in the part, but also the total multiplicity available for this
component in this part or any lesser part above it on the stack.

One added twist is that Knuth does not represent the part vectors as
arrays. Instead, he uses a sparse representation, in which a
component of a part is represented as a component number (c), plus
the multiplicity of the component in that part (v) as well as the
total multiplicity available for that component (u).  This saves
time that would be spent skipping over zeros.

"""

class PartComponent:
    """
    Internal class used in support of the multiset partitions
    enumerators and the associated visitor functions.

    Represents one component of one part of the current partition.

    A stack of these, plus an auxiliary frame array, f, represents a
    partition of the multiset.

    Knuth's pseudocode makes c, u, and v separate arrays.
    """

    __slots__ = ('c', 'u', 'v')

    def __init__(self):
        # Component number of the part component
        self.c = 0   # Component number
        # Amount of unpartitioned multiplicity in component c
        self.u = 0   # The as yet unpartitioned amount in component c
                     # *before* it is allocated by this triple
        # Amount of component c in the current part (v <= u)
        self.v = 0   # Amount of c component in the current part
                     # (v<=u).  An invariant of the representation is
                     # that the next higher triple for this component
                     # (if there is one) will have a value of u-v in
                     # its u attribute.

    def __repr__(self):
        """
        Return a string representation of the PartComponent object.
        This method is primarily for debug/algorithm animation purposes.
        """
        return 'c:%d u:%d v:%d' % (self.c, self.u, self.v)

    def __eq__(self, other):
        """
        Define value-oriented equality for PartComponent objects.

        Returns True if both objects are instances of PartComponent
        and their c, u, v attributes are equal; otherwise, returns False.
        """
        return (isinstance(other, self.__class__) and
                self.c == other.c and
                self.u == other.u and
                self.v == other.v)

    def __ne__(self, other):
        """
        Define the inequality comparison for PartComponent objects.

        Returns True if self and other are not equal as defined by __eq__;
        otherwise, returns False.
        """
        return not self == other
# 这个函数试图忠实地实现《计算机程序设计艺术》第4A卷中第7.1.2.5M算法。这包括使用（大部分）相同的变量名等等。
# 这使得代码非常接近底层Python实现。

# 与Knuth的伪代码不同之处包括：
# - 使用PartComponent结构/对象代替3个数组
# - 将函数设计为生成器
# - 将GOTO语句（有些困难地）映射到Python控制结构。
# - Knuth使用基于1的编号来表示组件，而这里的代码使用基于0的编号。
# - 将变量l重命名为lpart。
# - 标志变量x取True/False值，而不是1/0。

def multiset_partitions_taocp(multiplicities):
    """枚举一个多重集的分区。

    Parameters
    ==========

    multiplicities
        多重集组件的整数重数列表。

    Yields
    ======

    state
        编码特定分区的内部数据结构。
        通常，这个输出会被传递给一个访问函数，该函数将这个数据结构与组件本身的信息结合起来，生成实际的分区。

        除非用户希望创建自己的访问函数，否则他们不太需要查看这个数据结构的内部。但是，供参考，它是一个包含以下三个组件的列表：

        f
            用于将pstack分成部分的框架数组。

        lpart
            指向最顶部部分的基地址。

        pstack
            是PartComponent对象的数组。

        “state”输出提供了对枚举函数内部数据结构的一瞥。客户端应将其视为只读；修改数据结构将导致不可预测的（几乎肯定是错误的）结果。
        此外，“state”的组件在每次迭代中都会被就地修改。因此，必须在每次循环迭代中调用访问者函数。累积“state”实例并稍后处理它们将不起作用。

    Examples
    ========

    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> # 变量components和multiplicities表示多重集'abb'
    >>> components = 'ab'
    >>> multiplicities = [1, 2]
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(list_visitor(state, components) for state in states)
    [[['a', 'b', 'b']],
     [['a', 'b'], ['b']],
     [['a'], ['b', 'b']],
     [['a'], ['b'], ['b']]]

    See Also
    ========

    sympy.utilities.iterables.multiset_partitions：以多重集作为输入，并直接产生多重集的分区。它分派给多个函数之一，包括此函数，用于实现。
    大多数用户会发现它比multiset_partitions_taocp更方便使用。
    """
    # 重要变量。
    # m 是组件数，即不同元素的数量。
    m = len(multiplicities)
    # n 是基数，总元素数量，无论是否不同。
    n = sum(multiplicities)
    
    # 主要数据结构，f 将 pstack 分段成部分。参见 list_visitor() 中的示例代码，
    # 表明这个内部状态如何对应于一个分区。
    #
    # 注意：为堆栈分配的空间是保守的。Knuth 的练习 7.2.1.5.68 给出了如何收紧这个界限的一些线索，
    # 但这部分尚未实现。
    pstack = [PartComponent() for i in range(n * m + 1)]
    f = [0] * (n + 1)
    
    # 步骤 M1 在 Knuth 的算法中（初始化）
    # 初始状态 - 整个多重集合在一个部分中。
    for j in range(m):
        ps = pstack[j]
        ps.c = j
        ps.u = multiplicities[j]
        ps.v = multiplicities[j]
    
    # 其他变量
    f[0] = 0
    a = 0
    lpart = 0
    f[1] = m
    b = m  # 一般情况下，当前堆栈帧从 a 到 b - 1
    
    while True:
        while True:
            # 步骤 M2 （从 u 中减去 v）
            k = b
            x = False
            for j in range(a, b):
                pstack[k].u = pstack[j].u - pstack[j].v
                if pstack[k].u == 0:
                    x = True
                elif not x:
                    pstack[k].c = pstack[j].c
                    pstack[k].v = min(pstack[j].v, pstack[k].u)
                    x = pstack[k].u < pstack[j].v
                    k = k + 1
                else:  # x is True
                    pstack[k].c = pstack[j].c
                    pstack[k].v = pstack[k].u
                    k = k + 1
                # 注意：当 x 为 True 时，v 已经发生了改变
    
            # 步骤 M3 （如果非零则推送）
            if k > b:
                a = b
                b = k
                lpart = lpart + 1
                f[lpart + 1] = b
                # 返回到 M2
            else:
                break  # 继续到 M4
    
        # M4  访问一个分区
        state = [f, lpart, pstack]
        yield state
    
        # M5 （减少 v）
        while True:
            j = b-1
            while (pstack[j].v == 0):
                j = j - 1
            if j == a and pstack[j].v == 1:
                # M6 （回溯）
                if lpart == 0:
                    return
                lpart = lpart - 1
                b = a
                a = f[lpart]
                # 返回到 M5
            else:
                pstack[j].v = pstack[j].v - 1
                for k in range(j + 1, b):
                    pstack[k].v = pstack[k].u
                break  # 转到 M2
# --------------- Visitor functions for multiset partitions ---------------
# 用于多重集分区的访问函数
# A visitor takes the partition state generated by
# multiset_partitions_taocp or other enumerator, and produces useful
# output (such as the actual partition).
# 访问函数接受由 multiset_partitions_taocp 或其他枚举器生成的分区状态，并生成有用的输出（例如实际分区）。

def factoring_visitor(state, primes):
    """Use with multiset_partitions_taocp to enumerate the ways a
    number can be expressed as a product of factors.  For this usage,
    the exponents of the prime factors of a number are arguments to
    the partition enumerator, while the corresponding prime factors
    are input here.
    
    Examples
    ========
    
    To enumerate the factorings of a number we can think of the elements of the
    partition as being the prime factors and the multiplicities as being their
    exponents.
    
    >>> from sympy.utilities.enumerative import factoring_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> from sympy import factorint
    >>> primes, multiplicities = zip(*factorint(24).items())
    >>> primes
    (2, 3)
    >>> multiplicities
    (3, 1)
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(factoring_visitor(state, primes) for state in states)
    [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3], [6, 2, 2], [2, 2, 2, 3]]
    """
    f, lpart, pstack = state
    factoring = []
    for i in range(lpart + 1):
        factor = 1
        for ps in pstack[f[i]: f[i + 1]]:
            if ps.v > 0:
                factor *= primes[ps.c] ** ps.v
        factoring.append(factor)
    return factoring


def list_visitor(state, components):
    """Return a list of lists to represent the partition.
    
    Examples
    ========
    
    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> states = multiset_partitions_taocp([1, 2, 1])
    >>> s = next(states)
    >>> list_visitor(s, 'abc')  # for multiset 'a b b c'
    [['a', 'b', 'b', 'c']]
    >>> s = next(states)
    >>> list_visitor(s, [1, 2, 3])  # for multiset '1 2 2 3
    [[1, 2, 2], [3]]
    """
    f, lpart, pstack = state

    partition = []
    for i in range(lpart+1):
        part = []
        for ps in pstack[f[i]:f[i+1]]:
            if ps.v > 0:
                part.extend([components[ps.c]] * ps.v)
        partition.append(part)

    return partition


class MultisetPartitionTraverser():
    """
    Has methods to ``enumerate`` and ``count`` the partitions of a multiset.

    This implements a refactored and extended version of Knuth's algorithm
    7.1.2.5M [AOCP]_."

    The enumeration methods of this class are generators and return
    data structures which can be interpreted by the same visitor
    functions used for the output of ``multiset_partitions_taocp``.

    Examples
    ========

    >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
    >>> m = MultisetPartitionTraverser()
    >>> m.count_partitions([4,4,4,2])
    127750
    >>> m.count_partitions([3,3,3])
    """
    # 用于“枚举”和“计数”多重集的分区的方法。
    # 实现了Knuth算法7.1.2.5M的重构和扩展版本。
    # 此类的枚举方法是生成器，返回可以由用于 multiset_partitions_taocp 输出的相同访问函数解释的数据结构。
    """
    # 初始化方法，设置调试标志和追踪变量
    def __init__(self):
        # 调试模式，默认关闭
        self.debug = False
        # 跟踪变量。用于收集算法本身的统计数据，对于代码的最终用户没有特定的好处
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.pstack = None
        self.f = None
        self.lpart = 0
        self.discarded = 0
        # dp_stack 是包含 (part_key, start_count) 对的列表的列表
        self.dp_stack = []

        # dp_map 是 part_key 到 count 的映射，其中 count 表示多重集合的数量，
        # 这些多重集合是这个键的部分及其任何减少的后代
        #
        # 因此，当我们在映射中找到一个部分时，我们将其计数值添加到运行总数中，
        # 停止枚举，并进行回溯

        # 如果没有定义 dp_map，则初始化为空字典
        if not hasattr(self, 'dp_map'):
            self.dp_map = {}

    # 用于理解/调试算法的追踪方法。不会在最终用户代码中激活。
    def db_trace(self, msg):
        if self.debug:
            # XXX: animation_visitor 未定义... 显然这不起作用，并且未经测试。下面的注释中的以前代码。
            raise RuntimeError
            #letters = 'abcdefghijklmnopqrstuvwxyz'
            #state = [self.f, self.lpart, self.pstack]
            #print("DBG:", msg,
            #      ["".join(part) for part in list_visitor(state, letters)],
            #      animation_visitor(state))

    #
    # 枚举的辅助方法
    #
    """
    def _initialize_enumeration(self, multiplicities):
        """Allocates and initializes the partition stack.

        This is called from the enumeration/counting routines, so
        there is no need to call it separately."""

        num_components = len(multiplicities)
        # cardinality is the total number of elements, whether or not distinct
        cardinality = sum(multiplicities)

        # pstack is the partition stack, which is segmented by
        # f into parts.
        self.pstack = [PartComponent() for i in
                       range(num_components * cardinality + 1)]
        self.f = [0] * (cardinality + 1)

        # Initial state - entire multiset in one part.
        for j in range(num_components):
            ps = self.pstack[j]
            ps.c = j
            ps.u = multiplicities[j]
            ps.v = multiplicities[j]

        self.f[0] = 0
        self.f[1] = num_components
        self.lpart = 0

    # The decrement_part() method corresponds to step M5 in Knuth's
    # algorithm.  This is the base version for enum_all().  Modified
    # versions of this method are needed if we want to restrict
    # sizes of the partitions produced.
    def decrement_part(self, part):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        If you think of the v values in the part as a multi-digit
        integer (least significant digit on the right) this is
        basically decrementing that integer, but with the extra
        constraint that the leftmost digit cannot be decremented to 0.

        Parameters
        ==========

        part
           The part, represented as a list of PartComponent objects,
           which is to be decremented.

        """
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            if j == 0 and part[j].v > 1 or j > 0 and part[j].v > 0:
                # found val to decrement
                part[j].v -= 1
                # Reset trailing parts back to maximum
                for k in range(j + 1, plen):
                    part[k].v = part[k].u
                return True
        return False

    # Version to allow number of parts to be bounded from above.
    # Corresponds to (a modified) step M5.
    def decrement_part_large(self, part, amt, lb):
        """Decrements part, while respecting size constraint.

        A part can have no children which are of sufficient size (as
        indicated by ``lb``) unless that part has sufficient
        unallocated multiplicity.  When enforcing the size constraint,
        this method will decrement the part (if necessary) by an
        amount needed to ensure sufficient unallocated multiplicity.

        Returns True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        amt
            Can only take values 0 or 1.  A value of 1 means that the
            part must be decremented, and then the size constraint is
            enforced.  A value of 0 means just to enforce the ``lb``
            size constraint.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        """

        if amt == 1:
            # 如果 amt 为 1，则需要首先尝试减少 part 的大小
            # 在执行“足够未分配的重复性”约束之前。最简单的方法是调用常规的减少方法。
            if not self.decrement_part(part):
                return False

        # 接下来，执行任何必要的额外减少以遵守“足够未分配的重复性”约束（如果不可能则失败）。
        min_unalloc = lb - self.lpart
        if min_unalloc <= 0:
            return True
        total_mult = sum(pc.u for pc in part)
        total_alloc = sum(pc.v for pc in part)
        if total_mult <= min_unalloc:
            return False

        deficit = min_unalloc - (total_mult - total_alloc)
        if deficit <= 0:
            return True

        # 逆序遍历 part 列表
        for i in range(len(part) - 1, -1, -1):
            if i == 0:
                if part[0].v > deficit:
                    part[0].v -= deficit
                    return True
                else:
                    return False  # 这种情况不应发生，因为上面已经进行了检查
            else:
                if part[i].v >= deficit:
                    part[i].v -= deficit
                    return True
                else:
                    deficit -= part[i].v
                    part[i].v = 0
    # 减少给定范围内的部分（pstack的子范围），如果可能的话，返回True表示成功减少了该部分。

    Parameters
    ==========

    part
        要减少的部分（栈顶部的部分）

    ub
        分区中允许的最大部分数，由调用遍历返回。

    lb
        调用枚举生成的分区必须具有超过此值的部分数。

    Notes
    =====

    结合了_decrement_small和_decrement_large方法的约束。如果成功返回，part至少已经减少了一次，但如果需要满足lb约束，可能会减少更多次。

    # 在范围情况下的约束只是强制执行_decrement_small和_decrement_large情况的约束。
    # 注意_large调用的第二个参数为0，这是仅在需要时进行减少以进行约束强制执行的信号。
    # 'and'运算符的短路和从左到右的顺序对于此正确工作是重要的。
    return self.decrement_part_small(part, ub) and \
        self.decrement_part_large(part, 0, lb)
    def spread_part_multiplicity(self):
        """
        Returns True if a new part has been created, and
        adjusts pstack, f and lpart as needed.

        Notes
        =====

        Spreads unallocated multiplicity from the current top part
        into a new part created above the current on the stack.  This
        new part is constrained to be less than or equal to the old in
        terms of the part ordering.

        This call does nothing (and returns False) if the current top
        part has no unallocated multiplicity.

        """
        j = self.f[self.lpart]  # 当前顶部部分的基址
        k = self.f[self.lpart + 1]  # 当前部分的上界；下一个潜在的基址
        base = k  # 保存以便后续比较

        changed = False  # 当新部分严格小于（而不是小于或等于）旧部分时设置为True
                         # 
        for j in range(self.f[self.lpart], self.f[self.lpart + 1]):
            self.pstack[k].u = self.pstack[j].u - self.pstack[j].v
            if self.pstack[k].u == 0:
                changed = True
            else:
                self.pstack[k].c = self.pstack[j].c
                if changed:  # 将所有可用的重复性放入此部分
                    self.pstack[k].v = self.pstack[k].u
                else:  # 仍然保持顺序约束
                    if self.pstack[k].u < self.pstack[j].v:
                        self.pstack[k].v = self.pstack[k].u
                        changed = True
                    else:
                        self.pstack[k].v = self.pstack[j].v
                k = k + 1
        if k > base:
            # 调整堆栈上的新部分
            self.lpart = self.lpart + 1
            self.f[self.lpart + 1] = k
            return True
        return False

    def top_part(self):
        """
        Return current top part on the stack, as a slice of pstack.
        """
        return self.pstack[self.f[self.lpart]:self.f[self.lpart + 1]]

    # multiset_partitions_taocp()的接口和功能完全相同，
    # 但是有些人可能会发现这个重构版本更容易理解。
    def enum_all(self, multiplicities):
        """Enumerate the partitions of a multiset.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_all([2,2])
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'a'], ['b'], ['b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        multiset_partitions_taocp:
            which provides the same result as this method, but is
            about twice as fast.  Hence, enum_all is primarily useful
            for testing.  Also see the function for a discussion of
            states and visitors.

        """
        # 初始化枚举过程
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass

            # M4  访问一个分区
            state = [self.f, self.lpart, self.pstack]
            yield state

            # M5 (减少 v)
            while not self.decrement_part(self.top_part()):
                # M6 (回溯)
                if self.lpart == 0:
                    return
                self.lpart -= 1
    def enum_small(self, multiplicities, ub):
        """Enumerate multiset partitions with no more than ``ub`` parts.

        Equivalent to enum_range(multiplicities, 0, ub)

        Parameters
        ==========

        multiplicities
             list of multiplicities of the components of the multiset.

        ub
            Maximum number of parts

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_small([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        The implementation is based, in part, on the answer given to
        exercise 69, in Knuth [AOCP]_.

        See Also
        ========

        enum_all, enum_large, enum_range

        """

        # Keep track of iterations which do not yield a partition.
        # Clearly, we would like to keep this number small.
        self.discarded = 0  # 初始化一个变量，用于记录未生成分区的迭代次数
        if ub <= 0:
            return  # 如果最大部分数小于等于0，直接返回
        self._initialize_enumeration(multiplicities)  # 初始化枚举器的状态
        while True:
            while self.spread_part_multiplicity():
                self.db_trace('spread 1')  # 跟踪调试信息：扩展部分多重性
                if self.lpart >= ub:
                    self.discarded += 1  # 增加未生成分区的计数
                    self.db_trace('  Discarding')  # 跟踪调试信息：丢弃状态
                    self.lpart = ub - 2  # 设置当前最后一个部分的数目
                    break  # 跳出内层循环
            else:
                # M4  Visit a partition
                state = [self.f, self.lpart, self.pstack]  # 保存当前分区的状态
                yield state  # 生成当前分区的状态

            # M5 (Decrease v)
            while not self.decrement_part_small(self.top_part(), ub):
                self.db_trace("Failed decrement, going to backtrack")  # 跟踪调试信息：减少失败，回溯中
                # M6 (Backtrack)
                if self.lpart == 0:
                    return  # 如果最后一个部分数目为0，直接返回
                self.lpart -= 1  # 减少当前最后一个部分的数目
                self.db_trace("Backtracked to")  # 跟踪调试信息：回溯到某个状态
            self.db_trace("decrement ok, about to expand")  # 跟踪调试信息：减少成功，即将扩展
    def enum_large(self, multiplicities, lb):
        """Enumerate the partitions of a multiset with lb < num(parts)
        
        Equivalent to enum_range(multiplicities, lb, sum(multiplicities))
        
        Parameters
        ==========
        
        multiplicities
            list of multiplicities of the components of the multiset.
        
        lb
            Number of parts in the partition must be greater than
            this lower bound.
        
        
        Examples
        ========
        
        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_large([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a'], ['b'], ['b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]
        
        See Also
        ========
        
        enum_all, enum_small, enum_range
        
        """
        # Initialize discarded count
        self.discarded = 0
        # If lower bound is greater than or equal to the sum of multiplicities, return immediately
        if lb >= sum(multiplicities):
            return
        # Initialize the enumeration process
        self._initialize_enumeration(multiplicities)
        # Decrease the top part's value and generate partitions
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            # Spread part multiplicities and handle large partitions
            while self.spread_part_multiplicity():
                # Attempt to decrement the top part and handle failure cases
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # Rare/impossible failure case
                    self.discarded += 1
                    good_partition = False
                    break
            
            # Visit a partition if it's good
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state
            
            # Decrease v and backtrack if necessary
            while not self.decrement_part_large(self.top_part(), 1, lb):
                if self.lpart == 0:
                    return
                self.lpart -= 1
    def enum_range(self, multiplicities, lb, ub):
        """
        Enumerate the partitions of a multiset with
        ``lb < num(parts) <= ub``.

        In particular, if partitions with exactly ``k`` parts are
        desired, call with ``(multiplicities, k - 1, k)``.  This
        method generalizes enum_all, enum_small, and enum_large.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_range([2,2], 1, 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        """

        # combine the constraints of the _large and _small
        # enumerations.
        self.discarded = 0
        # 如果上界小于等于0或者下界大于等于多重集的总和，则直接返回
        if ub <= 0 or lb >= sum(multiplicities):
            return
        # 初始化枚举过程
        self._initialize_enumeration(multiplicities)
        # 减少顶部部分的大约束，直到满足下界 lb
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            # 扩展部分的多重性
            while self.spread_part_multiplicity():
                self.db_trace("spread 1")
                # 如果无法满足大约束，则失败
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # 失败的情况 - 在范围情况下可能出现？
                    self.db_trace("  Discarding (large cons)")
                    self.discarded += 1
                    good_partition = False
                    break
                # 如果顶部部分的长度大于等于上界 ub，则丢弃当前分区
                elif self.lpart >= ub:
                    self.discarded += 1
                    good_partition = False
                    self.db_trace("  Discarding small cons")
                    self.lpart = ub - 2
                    break

            # 访问一个分区
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state

            # 减少 v
            while not self.decrement_part_range(self.top_part(), lb, ub):
                self.db_trace("Failed decrement, going to backtrack")
                # 回溯
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace("Backtracked to")
            self.db_trace("decrement ok, about to expand")
    # 定义一个方法，用于计算具有给定元素多重度的多重集合的分区数量
    def count_partitions_slow(self, multiplicities):
        """Returns the number of partitions of a multiset whose elements
        have the multiplicities given in ``multiplicities``.

        Primarily for comparison purposes.  It follows the same path as
        enumerate, and counts, rather than generates, the partitions.

        See Also
        ========

        count_partitions
            Has the same calling interface, but is much faster.

        """
        # 初始化分区计数器
        self.pcount = 0
        # 初始化分区枚举状态
        self._initialize_enumeration(multiplicities)
        # 进入无限循环，枚举分区
        while True:
            # 执行分布式部分多重度的扩展，直到无法继续
            while self.spread_part_multiplicity():
                pass

            # 访问（计数）一个分区，增加分区计数器
            self.pcount += 1

            # 减少当前部分的多重度
            while not self.decrement_part(self.top_part()):
                # 回溯到上一个分区
                if self.lpart == 0:
                    return self.pcount
                # 减少分区计数器，回到上一个分区
                self.lpart -= 1
# 定义一个辅助函数，为 MultisetPartitionTraverser.count_partitions 函数创建一个键值，
# 该键值只包含可能影响该部分计数的信息。任何无关的信息只会降低动态编程的效率。

def part_key(part):
    """
    Helper for MultisetPartitionTraverser.count_partitions that
    creates a key for ``part``, that only includes information which can
    affect the count for that part.  (Any irrelevant information just
    reduces the effectiveness of dynamic programming.)

    Notes
    =====

    This member function is a candidate for future exploration. There
    are likely symmetries that can be exploited to coalesce some
    ``part_key`` values, and thereby save space and improve
    performance.

    """
    # 组件编号对于计算分区是无关紧要的，因此在记忆键中不包含它。
    rval = []  # 创建一个空列表 rval
    for ps in part:  # 遍历 part 中的每个元素 ps
        rval.append(ps.u)  # 将 ps.u 添加到 rval 中
        rval.append(ps.v)  # 将 ps.v 添加到 rval 中
    return tuple(rval)  # 返回 rval 转换成元组后的结果作为键值
```