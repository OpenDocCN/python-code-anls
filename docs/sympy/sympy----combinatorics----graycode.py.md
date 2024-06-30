# `D:\src\scipysrc\sympy\sympy\combinatorics\graycode.py`

```
# 导入 sympy.core 模块中的 Basic 和 Integer 类
from sympy.core import Basic, Integer

# 导入 random 模块
import random

# 定义 GrayCode 类，继承自 Basic 类
class GrayCode(Basic):
    """
    GrayCode 类表示格雷码，是一个 n 维立方体上的哈密顿路径。
    每个顶点由二进制值表示，哈密顿路径访问每个顶点一次。
    格雷码解决了依次生成 n 个对象的所有可能子集的问题，
    每个子集可以通过添加或删除单个对象从前一个子集获得。
    """

    _skip = False  # 标志变量，用于控制生成过程中的跳过状态
    _current = 0   # 当前生成的格雷码索引
    _rank = None   # 格雷码的秩，用于计算统计信息和子集相关的应用
    def __new__(cls, n, *args, **kw_args):
        """
        默认构造函数。

        它接受一个参数 ``n``，表示Gray码的维度。也可以指定起始Gray码字符串（``start``）或起始“排名”（``rank``）；
        默认情况下，从rank = 0（'0...0'）开始。

        示例
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a
        GrayCode(3)
        >>> a.n
        3

        >>> a = GrayCode(3, start='100')
        >>> a.current
        '100'

        >>> a = GrayCode(4, rank=4)
        >>> a.current
        '0110'
        >>> a.rank
        4

        """
        # 如果 n 小于 1 或者 n 不是整数，则抛出 ValueError 异常
        if n < 1 or int(n) != n:
            raise ValueError(
                'Gray code dimension must be a positive integer, not %i' % n)
        # 将 n 转换为整数类型
        n = Integer(n)
        # 将参数重新组合为元组
        args = (n,) + args
        # 调用父类的构造函数创建对象
        obj = Basic.__new__(cls, *args)
        # 如果关键字参数中包含 'start'
        if 'start' in kw_args:
            # 设置当前 Gray 码字符串为给定的 start
            obj._current = kw_args["start"]
            # 如果当前 Gray 码字符串长度超过 n，则抛出异常
            if len(obj._current) > n:
                raise ValueError('Gray code start has length %i but '
                'should not be greater than %i' % (len(obj._current), n))
        # 如果关键字参数中包含 'rank'
        elif 'rank' in kw_args:
            # 如果 rank 不是整数，则抛出异常
            if int(kw_args["rank"]) != kw_args["rank"]:
                raise ValueError('Gray code rank must be a positive integer, '
                'not %i' % kw_args["rank"])
            # 计算有效的 rank 值，并设置当前 Gray 码字符串
            obj._rank = int(kw_args["rank"]) % obj.selections
            obj._current = obj.unrank(n, obj._rank)
        # 返回创建的对象
        return obj

    def next(self, delta=1):
        """
        返回当前值在规范顺序中距离 ``delta``（默认为 1）的Gray码。

        示例
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3, start='110')
        >>> a.next().current
        '111'
        >>> a.next(-1).current
        '010'
        """
        # 返回从当前 Gray 码值开始的第 delta 个Gray码对象
        return GrayCode(self.n, rank=(self.rank + delta) % self.selections)

    @property
    def selections(self):
        """
        返回Gray码中的比特向量数量。

        示例
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a.selections
        8
        """
        # 返回2的n次方，即Gray码中的比特向量数量
        return 2**self.n

    @property
    def n(self):
        """
        返回Gray码的维度。

        示例
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(5)
        >>> a.n
        5
        """
        # 返回Gray码对象的第一个参数，即维度n
        return self.args[0]
    def generate_gray(self, **hints):
        """
        Generates the sequence of bit vectors of a Gray Code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(start='011'))
        ['011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(rank=4))
        ['110', '111', '101', '100']

        See Also
        ========

        skip

        References
        ==========

        .. [1] Knuth, D. (2011). The Art of Computer Programming,
               Vol 4, Addison Wesley

        """
        bits = self.n  # 获取当前 Gray Code 对象的比特数
        start = None  # 初始化起始值为 None
        if "start" in hints:  # 如果 hints 字典中包含 'start' 键
            start = hints["start"]  # 则将 start 设为 hints 中 'start' 键对应的值
        elif "rank" in hints:  # 否则，如果 hints 字典中包含 'rank' 键
            start = GrayCode.unrank(self.n, hints["rank"])  # 则通过 unrank 方法获取对应的 Gray Code 值
        if start is not None:  # 如果 start 不为 None
            self._current = start  # 将当前 Gray Code 对象的当前值设为 start
        current = self.current  # 获取当前 Gray Code 对象的当前值
        graycode_bin = gray_to_bin(current)  # 将当前值转换为 Gray Code 二进制表示
        if len(graycode_bin) > self.n:  # 如果 Gray Code 二进制表示的长度大于比特数
            raise ValueError('Gray code start has length %i but should '
            'not be greater than %i' % (len(graycode_bin), bits))  # 抛出数值错误异常
        self._current = int(current, 2)  # 将当前 Gray Code 对象的当前值转换为整数
        graycode_int = int(''.join(graycode_bin), 2)  # 将 Gray Code 二进制表示转换为整数
        for i in range(graycode_int, 1 << bits):  # 遍历从 graycode_int 到 2^bits 之间的整数
            if self._skip:  # 如果需要跳过当前生成的位向量
                self._skip = False  # 将 _skip 标志设为 False
            else:
                yield self.current  # 生成当前的位向量
            bbtc = (i ^ (i + 1))  # 计算当前和下一个 Gray Code 数字的二进制表示的异或
            gbtc = (bbtc ^ (bbtc >> 1))  # 计算 Gray Code 的二进制表示的异或结果
            self._current = (self._current ^ gbtc)  # 更新当前 Gray Code 对象的当前值
        self._current = 0  # 将当前 Gray Code 对象的当前值重置为 0

    def skip(self):
        """
        Skips the bit generation.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> for i in a.generate_gray():
        ...     if i == '010':
        ...         a.skip()
        ...     print(i)
        ...
        000
        001
        011
        010
        111
        101
        100

        See Also
        ========

        generate_gray
        """
        self._skip = True  # 设置 _skip 标志为 True，用于跳过位生成过程
    def rank(self):
        """
        Ranks the Gray code.

        A ranking algorithm determines the position (or rank)
        of a combinatorial object among all the objects w.r.t.
        a given order. For example, the 4 bit binary reflected
        Gray code (BRGC) '0101' has a rank of 6 as it appears in
        the 6th position in the canonical ordering of the family
        of 4 bit Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> GrayCode(3, start='100').rank
        7
        >>> GrayCode(3, rank=7).current
        '100'

        See Also
        ========

        unrank

        References
        ==========

        .. [1] https://web.archive.org/web/20200224064753/http://statweb.stanford.edu/~susan/courses/s208/node12.html

        """
        if self._rank is None:
            # Convert the current Gray code to binary and then to integer rank
            self._rank = int(gray_to_bin(self.current), 2)
        return self._rank

    @property
    def current(self):
        """
        Returns the currently referenced Gray code as a bit string.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(3, start='100').current
        '100'
        """
        # Retrieve the current Gray code or default to '0', convert to binary if necessary
        rv = self._current or '0'
        if not isinstance(rv, str):
            rv = bin(rv)[2:]
        # Right justify the string representation to the length of n
        return rv.rjust(self.n, '0')

    @classmethod
    def unrank(self, n, rank):
        """
        Unranks an n-bit sized Gray code of rank k. This method exists
        so that a derivative GrayCode class can define its own code of
        a given rank.

        The string here is generated in reverse order to allow for tail-call
        optimization.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(5, rank=3).current
        '00010'
        >>> GrayCode.unrank(5, 3)
        '00010'

        See Also
        ========

        rank
        """
        # Internal recursive function to generate the Gray code based on rank and n
        def _unrank(k, n):
            if n == 1:
                return str(k % 2)
            m = 2**(n - 1)
            if k < m:
                return '0' + _unrank(k, n - 1)
            return '1' + _unrank(m - (k % m) - 1, n - 1)
        return _unrank(rank, n)
# 生成指定长度为 n 的随机比特串
def random_bitstring(n):
    return ''.join([random.choice('01') for i in range(n)])


# 将格雷码转换为二进制编码
# 假设使用大端编码
def gray_to_bin(bin_list):
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        # 根据格雷码规则，计算每一位的二进制值
        b += str(int(b[i - 1] != bin_list[i]))
    return ''.join(b)


# 将二进制编码转换为格雷码
# 假设使用大端编码
def bin_to_gray(bin_list):
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        # 根据格雷码规则，计算每一位的格雷码值
        b += str(int(bin_list[i]) ^ int(bin_list[i - 1]))
    return ''.join(b)


# 根据比特串获取超集中定义的子集
def get_subset_from_bitstring(super_set, bitstring):
    # 如果超集和比特串的长度不相等，抛出值错误异常
    if len(super_set) != len(bitstring):
        raise ValueError("The sizes of the lists are not equal")
    # 返回根据比特串定义的子集列表
    return [super_set[i] for i, j in enumerate(bitstring)
            if bitstring[i] == '1']


# 根据格雷码集合生成其对应的子集序列
def graycode_subsets(gray_code_set):
    # 使用格雷码生成器逐个生成格雷码
    for bitstring in list(GrayCode(len(gray_code_set)).generate_gray()):
        # 使用比特串获取子集的方法生成子集列表并逐个返回
        yield get_subset_from_bitstring(gray_code_set, bitstring)
```