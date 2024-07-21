# `.\pytorch\torch\_dynamo\polyfill.py`

```
"""
Python polyfills for common builtins.
"""
# 导入数学模块
import math
# 导入类型提示模块
from typing import Any, Callable, Sequence

# 导入PyTorch库
import torch


# 检查迭代器中所有元素是否为真
def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True


# 检查迭代器中是否有任何元素为真
def any(iterator):
    for elem in iterator:
        if elem:
            return True
    return False


# 查找迭代器中特定项的索引
def index(iterator, item, start=0, end=None):
    for i, elem in enumerate(list(iterator))[start:end]:
        if item == elem:
            return i
    # 如果未找到项，抛出值错误
    raise ValueError(f"{item} is not in {type(iterator)}")


# 生成重复项指定次数的生成器
def repeat(item, count):
    for i in range(count):
        yield item


# 将角度转换为弧度
def radians(x):
    return math.pi / 180.0 * x


# 累积计算梯度更新
def accumulate_grad(x, new_grad):
    new_grad = torch.clone(new_grad)
    if x.grad is None:
        x.grad = new_grad
    else:
        x.grad.add_(new_grad)


# 比较两个列表或元组的大小关系
def list_cmp(op: Callable[[Any, Any], bool], left: Sequence[Any], right: Sequence[Any]):
    """emulate `(1,2,3) > (1,2)` etc"""
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)
    return op(len(left), len(right))


# 检查两个集合是否不相交
def set_isdisjoint(set1, set2):
    for x in set1:
        if x in set2:
            return False
    return True


# 返回满足条件的元素之后的迭代器
def dropwhile(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) -> 6 4 1
    iterable = iter(iterable)
    for x in iterable:
        if not predicate(x):
            yield x
            break
    yield from iterable


# 获取对象属性并调用对应函数
def getattr_and_trace(*args, **kwargs):
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)
```