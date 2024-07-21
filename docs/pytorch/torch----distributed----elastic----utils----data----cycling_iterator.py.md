# `.\pytorch\torch\distributed\elastic\utils\data\cycling_iterator.py`

```py
# 定义一个迭代器类 CyclingIterator，用于包装一个可以通过多个周期循环的生成器函数
class CyclingIterator:
    """
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    """

    # 初始化方法，接受参数 n: 循环次数，generator_fn: 生成器函数，start_epoch: 起始周期
    def __init__(self, n: int, generator_fn, start_epoch=0):
        # 设定循环次数
        self._n = n
        # 当前周期数
        self._epoch = start_epoch
        # 保存生成器函数的引用
        self._generator_fn = generator_fn
        # 调用生成器函数获取初始的迭代器对象
        self._iter = generator_fn(self._epoch)

    # 定义迭代器的 __iter__ 方法，返回迭代器自身
    def __iter__(self):
        return self

    # 定义迭代器的 __next__ 方法，用于获取下一个元素
    def __next__(self):
        try:
            # 尝试从当前迭代器中获取下一个元素
            return next(self._iter)
        except StopIteration as eod:  # eod == end of data
            # 捕获迭代器抛出的 StopIteration 异常
            if self._epoch < self._n - 1:
                # 如果当前周期小于设定的循环次数减一
                self._epoch += 1
                # 增加周期数
                self._iter = self._generator_fn(self._epoch)
                # 调用生成器函数获取新的迭代器对象
                return self.__next__()  # 递归调用自身获取下一个元素
            else:
                # 如果已经达到设定的循环次数，则抛出 StopIteration 异常
                raise eod
```