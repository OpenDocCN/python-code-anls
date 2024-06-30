# `D:\src\scipysrc\scipy\scipy\_build_utils\tempita\_looper.py`

```
"""
Helper for looping over sequences, particular in templates.

Often in a loop in a template it's handy to know what's next up,
previously up, if this is the first or last item in the sequence, etc.
These can be awkward to manage in a normal Python loop, but using the
looper you can get a better sense of the context.  Use like::

    >>> for loop, item in looper(['a', 'b', 'c']):
    ...     print loop.number, item
    ...     if not loop.last:
    ...         print '---'
    1 a
    ---
    2 b
    ---
    3 c

"""

# 定义基本字符串类型为 bytes 或 str
basestring_ = (bytes, str)

# 定义可以导出的类、函数等
__all__ = ['looper']


class looper:
    """
    Helper for looping (particularly in templates)

    Use this like::

        for loop, item in looper(seq):
            if loop.first:
                ...
    """

    def __init__(self, seq):
        self.seq = seq

    def __iter__(self):
        return looper_iter(self.seq)

    def __repr__(self):
        return '<%s for %r>' % (
            self.__class__.__name__, self.seq)


class looper_iter:
    """
    Iterator class for looper

    This class iterates over the sequence provided by looper.
    """

    def __init__(self, seq):
        self.seq = list(seq)  # 将输入的序列转换为列表
        self.pos = 0  # 初始化位置索引为 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.seq):  # 如果位置索引超出序列长度
            raise StopIteration  # 抛出迭代结束异常
        result = loop_pos(self.seq, self.pos), self.seq[self.pos]  # 创建 loop_pos 实例和当前序列项的元组
        self.pos += 1  # 索引位置向后移动一位
        return result  # 返回包含 loop_pos 实例和当前序列项的元组


class loop_pos:
    """
    Class representing the position in the loop

    This class provides properties and methods to access information
    about the current position in the loop.
    """

    def __init__(self, seq, pos):
        self.seq = seq  # 保存序列的引用
        self.pos = pos  # 保存当前位置索引

    def __repr__(self):
        return '<loop pos=%r at %r>' % (
            self.seq[self.pos], self.pos)  # 返回当前位置的字符串表示形式

    @property
    def index(self):
        return self.pos  # 返回当前位置的索引

    @property
    def number(self):
        return self.pos + 1  # 返回当前位置的序号（从 1 开始）

    @property
    def item(self):
        return self.seq[self.pos]  # 返回当前位置的序列项

    @property
    def __next__(self):
        try:
            return self.seq[self.pos + 1]  # 返回下一个序列项
        except IndexError:
            return None  # 如果索引超出范围，则返回 None

    @property
    def previous(self):
        if self.pos == 0:
            return None
        return self.seq[self.pos - 1]  # 返回上一个序列项，如果当前是第一个则返回 None

    @property
    def odd(self):
        return not self.pos % 2  # 返回当前位置是否为奇数索引

    @property
    def even(self):
        return self.pos % 2  # 返回当前位置是否为偶数索引

    @property
    def first(self):
        return self.pos == 0  # 返回当前位置是否为第一个位置

    @property
    def last(self):
        return self.pos == len(self.seq) - 1  # 返回当前位置是否为最后一个位置

    @property
    def length(self):
        return len(self.seq)  # 返回序列的长度

    def first_group(self, getter=None):
        """
        Returns true if this item is the start of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        if self.first:
            return True
        return self._compare_group(self.item, self.previous, getter)

    def _compare_group(self, item, previous, getter):
        # 私有方法，用于比较当前项和前一项是否在同一组中
        # 根据 getter 参数指定的规则进行比较
        pass  # 这里未提供具体实现，应根据具体需求实现
    def last_group(self, getter=None):
        """
        Returns true if this item is the end of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        # 检查是否为最后一个元素
        if self.last:
            return True
        # 调用比较方法，判断当前元素和下一个元素是否在同一组
        return self._compare_group(self.item, self.__next__, getter)

    def _compare_group(self, item, other, getter):
        # 比较两个元素是否在同一组
        if getter is None:
            # 如果 getter 为 None，则直接比较两个元素是否相等
            return item != other
        elif (isinstance(getter, basestring_)
              and getter.startswith('.')):
            # 如果 getter 是字符串并且以点号开头，表示要获取对象的属性值
            getter = getter[1:]  # 去掉开头的点号
            if getter.endswith('()'):
                # 如果属性名以 () 结尾，表示调用对象的方法来获取属性值
                getter = getter[:-2]  # 去掉末尾的括号
                return getattr(item, getter)() != getattr(other, getter)()
            else:
                # 否则直接比较对象的属性值
                return getattr(item, getter) != getattr(other, getter)
        elif hasattr(getter, '__call__'):
            # 如果 getter 是可调用对象（函数），则调用函数来获取属性值进行比较
            return getter(item) != getter(other)
        else:
            # 否则假定 getter 是字典键或列表索引，比较对应的元素值
            return item[getter] != other[getter]
```