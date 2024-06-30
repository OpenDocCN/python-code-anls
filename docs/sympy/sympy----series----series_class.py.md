# `D:\src\scipysrc\sympy\sympy\series\series_class.py`

```
"""
Contains the base class for series
Made using sequences in mind
"""

# 导入需要的模块
from sympy.core.expr import Expr  # 导入 Expr 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.cache import cacheit  # 导入 cacheit 函数


class SeriesBase(Expr):
    """Base Class for series"""

    @property
    def interval(self):
        """The interval on which the series is defined"""
        # 返回序列定义的区间，由子类实现具体逻辑
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the series. This point is included"""
        # 返回序列的起始点（包含在内），由子类实现具体逻辑
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the series. This point is included"""
        # 返回序列的结束点（包含在内），由子类实现具体逻辑
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the series expansion"""
        # 返回序列展开的长度，由子类实现具体逻辑
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        # 返回受限变量的元组，对于基类来说为空元组
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).
        """
        # 返回对象中的符号，排除那些具有特定值的符号（即虚拟符号）
        return ({j for i in self.args for j in i.free_symbols}
                .difference(self.variables))

    @cacheit
    def term(self, pt):
        """Term at point pt of a series"""
        # 返回序列在指定点 pt 处的项，如果 pt 超出定义范围，则引发 IndexError
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_term(pt)

    def _eval_term(self, pt):
        # 由子类实现的方法，返回序列在指定点 pt 处的项
        raise NotImplementedError("The _eval_term method should be added to"
                                  "%s to return series term so it is available"
                                  "when 'term' calls it."
                                  % self.func)

    def _ith_point(self, i):
        """
        Returns the i'th point of a series
        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        ========

        TODO
        """
        # 返回序列的第 i 个点，若起始点为负无穷，则从末尾返回点，假设第一个点索引为零
        if self.start is S.NegativeInfinity:
            initial = self.stop
            step = -1
        else:
            initial = self.start
            step = 1

        return initial + i*step

    def __iter__(self):
        # 迭代器方法，返回序列的迭代器
        i = 0
        while i < self.length:
            pt = self._ith_point(i)
            yield self.term(pt)
            i += 1

    def __getitem__(self, index):
        # 索引方法，根据索引返回序列中的项或切片
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.term(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.term(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]
```