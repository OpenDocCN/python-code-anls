# `.\pytorch\torch\fx\experimental\unification\unification_tools.py`

```py
# mypy: allow-untyped-defs
# 引入collections模块，提供额外的数据结构和操作
import collections
# 引入operator模块，提供内置操作函数的函数
import operator
# 从functools模块中引入reduce函数，用于在序列上应用二进制函数，以至于每个序列元素归约为单个值
from functools import reduce
# 从collections.abc模块中引入Mapping抽象基类，用于验证对象是否是映射类型（如字典）
from collections.abc import Mapping

# 导出的公共接口名单
__all__ = ('merge', 'merge_with', 'valmap', 'keymap', 'itemmap',
           'valfilter', 'keyfilter', 'itemfilter',
           'assoc', 'dissoc', 'assoc_in', 'update_in', 'get_in')


def _get_factory(f, kwargs):
    # 从kwargs中获取factory参数，默认为dict
    factory = kwargs.pop('factory', dict)
    # 如果kwargs非空，则抛出TypeError异常，指出多余的关键字参数
    if kwargs:
        raise TypeError(f"{f.__name__}() got an unexpected keyword argument '{kwargs.popitem()[0]}'")
    return factory


def merge(*dicts, **kwargs):
    """ Merge a collection of dictionaries

    >>> merge({1: 'one'}, {2: 'two'})
    {1: 'one', 2: 'two'}

    Later dictionaries have precedence

    >>> merge({1: 2, 3: 4}, {3: 3, 4: 4})
    {1: 2, 3: 3, 4: 4}

    See Also:
        merge_with
    """
    # 如果dicts中只有一个参数且不是Mapping类型，则将其解包成单独的字典
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    # 获取合并函数的工厂函数，默认为dict类型
    factory = _get_factory(merge, kwargs)

    # 初始化结果字典
    rv = factory()
    # 将所有字典合并到结果字典中
    for d in dicts:
        rv.update(d)
    return rv


def merge_with(func, *dicts, **kwargs):
    """ Merge dictionaries and apply function to combined values

    A key may occur in more than one dict, and all values mapped from the key
    will be passed to the function as a list, such as func([val1, val2, ...]).

    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
    {1: 1, 2: 2, 3: 30}

    See Also:
        merge
    """
    # 如果dicts中只有一个参数且不是Mapping类型，则将其解包成单独的字典
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    # 获取合并函数的工厂函数，默认为dict类型
    factory = _get_factory(merge_with, kwargs)

    # 初始化结果字典
    result = factory()
    # 遍历所有字典
    for d in dicts:
        # 遍历字典中的键值对
        for k, v in d.items():
            # 如果键不在结果字典中，则初始化为一个列表，将值存入
            if k not in result:
                result[k] = [v]
            else:
                # 如果键已存在结果字典中，则将值追加到列表中
                result[k].append(v)
    # 对结果字典中的值应用函数func，生成新的结果字典
    return valmap(func, result, factory)


def valmap(func, d, factory=dict):
    """ Apply function to values of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> valmap(sum, bills)  # doctest: +SKIP
    {'Alice': 65, 'Bob': 45}

    See Also:
        keymap
        itemmap
    """
    # 初始化结果字典
    rv = factory()
    # 将字典d的键值对按键值顺序压缩成元组，应用func函数后更新到结果字典中
    rv.update(zip(d.keys(), map(func, d.values())))
    return rv


def keymap(func, d, factory=dict):
    """ Apply function to keys of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> keymap(str.lower, bills)  # doctest: +SKIP
    {'alice': [20, 15, 30], 'bob': [10, 35]}

    See Also:
        valmap
        itemmap
    """
    # 初始化结果字典
    rv = factory()
    # 将字典d的键应用func函数后映射为新的键，并与原始值一起更新到结果字典中
    rv.update(zip(map(func, d.keys()), d.values()))
    return rv


def itemmap(func, d, factory=dict):
    """ Apply function to items of dictionary

    >>> accountids = {"Alice": 10, "Bob": 20}
    >>> itemmap(reversed, accountids)  # doctest: +SKIP
    {10: "Alice", 20: "Bob"}

    See Also:
        keymap
        valmap
    """
    # 初始化结果字典
    rv = factory()
    # 将字典d的每个项应用func函数后更新到结果字典中
    rv.update(map(func, d.items()))
    return rv


def valfilter(predicate, d, factory=dict):
    """ 根据值过滤字典中的项

    >>> iseven = lambda x: x % 2 == 0  # 定义一个判断偶数的 lambda 函数
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}  # 创建一个示例字典
    >>> valfilter(iseven, d)  # 使用 iseven 函数过滤字典 d 的值
    {1: 2, 3: 4}  # 返回过滤后的结果字典，只包含符合条件的键值对

    See Also:
        keyfilter  # 查看键过滤函数的相关文档
        itemfilter  # 查看项过滤函数的相关文档
        valmap  # 查看值映射函数的相关文档
    """
    rv = factory()  # 调用工厂函数创建一个新的返回值容器
    for k, v in d.items():  # 遍历字典 d 的键值对
        if predicate(v):  # 判断值 v 是否满足给定的条件 predicate
            rv[k] = v  # 将满足条件的键值对添加到返回值容器中
    return rv  # 返回过滤后的字典
# Filter dictionary items based on a key predicate function, returning a new dictionary
def keyfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by key

    >>> iseven = lambda x: x % 2 == 0  # Define a lambda function to check if a number is even
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}  # Example dictionary
    >>> keyfilter(iseven, d)  # Filter dictionary 'd' by even keys
    {2: 3, 4: 5}

    See Also:
        valfilter  # Related function to filter by value
        itemfilter  # Related function to filter by item (key-value pair)
        keymap  # Related function to transform keys
    """
    rv = factory()  # Initialize an empty dictionary using the provided factory function
    for k, v in d.items():  # Iterate through key-value pairs in dictionary 'd'
        if predicate(k):  # Check if the predicate function returns True for key 'k'
            rv[k] = v  # If True, add key-value pair to result dictionary 'rv'
    return rv  # Return the filtered dictionary


# Filter dictionary items based on an item predicate function, returning a new dictionary
def itemfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by item

    >>> def isvalid(item):  # Define a function to check if an item meets certain criteria
    ...     k, v = item  # Unpack the key-value pair from the item
    ...     return k % 2 == 0 and v < 4  # Return True if key is even and value is less than 4

    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}  # Example dictionary
    >>> itemfilter(isvalid, d)  # Filter dictionary 'd' using 'isvalid' function
    {2: 3}

    See Also:
        keyfilter  # Related function to filter by key
        valfilter  # Related function to filter by value
        itemmap  # Related function to transform items
    """
    rv = factory()  # Initialize an empty dictionary using the provided factory function
    for item in d.items():  # Iterate through items (key-value pairs) in dictionary 'd'
        if predicate(item):  # Check if the predicate function returns True for the item
            k, v = item  # Unpack the key-value pair
            rv[k] = v  # If True, add key-value pair to result dictionary 'rv'
    return rv  # Return the filtered dictionary


# Create a new dictionary with an additional or updated key-value pair
def assoc(d, key, value, factory=dict):
    """ Return a new dict with new key value pair

    New dict has d[key] set to value. Does not modify the initial dictionary.

    >>> assoc({'x': 1}, 'x', 2)  # Example usage: update existing key 'x' with value 2
    {'x': 2}
    >>> assoc({'x': 1}, 'y', 3)   # Example usage: add new key 'y' with value 3
    {'x': 1, 'y': 3}
    """
    d2 = factory()  # Initialize an empty dictionary using the provided factory function
    d2.update(d)  # Copy all key-value pairs from dictionary 'd' to 'd2'
    d2[key] = value  # Set or update key 'key' with value 'value'
    return d2  # Return the new dictionary 'd2'


# Return a new dictionary with specified keys removed
def dissoc(d, *keys, **kwargs):
    """ Return a new dict with the given key(s) removed.

    New dict has d[key] deleted for each supplied key.
    Does not modify the initial dictionary.

    >>> dissoc({'x': 1, 'y': 2}, 'y')  # Example: Remove key 'y' from dictionary
    {'x': 1}
    >>> dissoc({'x': 1, 'y': 2}, 'y', 'x')  # Example: Remove keys 'x' and 'y' from dictionary
    {}
    >>> dissoc({'x': 1}, 'y')  # Example: Attempt to remove non-existent key 'y'
    {'x': 1}
    """
    factory = _get_factory(dissoc, kwargs)  # Get the factory function for creating a new dictionary
    d2 = factory()  # Initialize an empty dictionary using the factory function

    if len(keys) < len(d) * .6:  # Check if the number of keys to remove is less than 60% of total keys in 'd'
        d2.update(d)  # Copy all key-value pairs from dictionary 'd' to 'd2'
        for key in keys:  # Iterate through keys to remove
            if key in d2:  # Check if key exists in 'd2'
                del d2[key]  # Delete key from 'd2'
    else:
        remaining = set(d)  # Create a set of all keys in dictionary 'd'
        remaining.difference_update(keys)  # Remove keys to be deleted from 'remaining'
        for k in remaining:  # Iterate through remaining keys
            d2[k] = d[k]  # Copy key-value pair to 'd2' if key is not in 'keys'
    return d2  # Return the new dictionary 'd2'


# Return a new dictionary with nested key updated to a new value
def assoc_in(d, keys, value, factory=dict):
    """ Return a new dict with new, potentially nested, key value pair

    >>> purchase = {'name': 'Alice',
    ...             'order': {'items': ['Apple', 'Orange'],
    ...                       'costs': [0.50, 1.25]},
    ...             'credit card': '5555-1234-1234-1234'}
    >>> assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])  # Example: Update nested key 'order.costs' with new value
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']}}
    """
    return update_in(d, keys, lambda x: value, value, factory)


# Update value in potentially nested dictionary based on given keys and update function
def update_in(d, keys, func, default=None, factory=dict):
    """ Update value in a (potentially) nested dictionary

    inputs:
    d - dictionary on which to operate
    keys - list or tuple giving the location of the value to be changed in d
    func - function to operate on that value

    If keys == [k0,..,kX] and d[k0]..[kX] == v, update_in returns a copy of the
    original dictionary with v replaced by func(v), but does not mutate the
    """
    Update a nested dictionary `d` with a value computed by `func`.

    If `k0` is not a key in `d`, `update_in` creates nested dictionaries up to the depth
    specified by the keys list `keys`, with the innermost value set to `func(default)`.

    >>> inc = lambda x: x + 1
    >>> update_in({'a': 0}, ['a'], inc)
    {'a': 1}

    >>> transaction = {'name': 'Alice',
    ...                'purchase': {'items': ['Apple', 'Orange'],
    ...                             'costs': [0.50, 1.25]},
    ...                'credit card': '5555-1234-1234-1234'}
    >>> update_in(transaction, ['purchase', 'costs'], sum) # doctest: +SKIP
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'purchase': {'costs': 1.75, 'items': ['Apple', 'Orange']}}

    >>> # updating a value when k0 is not in d
    >>> update_in({}, [1, 2, 3], str, default="bar")
    {1: {2: {3: 'bar'}}}
    >>> update_in({1: 'foo'}, [2, 3, 4], inc, 0)
    {1: 'foo', 2: {3: {4: 1}}}
    """
    # Initialize an iterator over the keys list
    ks = iter(keys)
    # Fetch the first key from the iterator
    k = next(ks)

    # Initialize rv and inner with the result of calling factory()
    rv = inner = factory()
    # Update rv with the contents of dictionary d
    rv.update(d)

    # Iterate over remaining keys
    for key in ks:
        # Check if current key k exists in d
        if k in d:
            # Update d to point to the nested dictionary at key k
            d = d[k]
            # Create a new temporary dictionary dtemp initialized by factory()
            dtemp = factory()
            # Update dtemp with contents of d
            dtemp.update(d)
        else:
            # If key k is not in d, initialize d and dtemp with factory()
            d = dtemp = factory()

        # Update inner[k] to point to the new nested dictionary dtemp
        inner[k] = inner = dtemp
        # Move to the next key
        k = key

    # Check if the last key k exists in d
    if k in d:
        # Update inner[k] with the result of applying func to d[k]
        inner[k] = func(d[k])
    else:
        # If key k is not in d, update inner[k] with func applied to default
        inner[k] = func(default)
    # Return the updated rv dictionary
    return rv
# 返回嵌套数据结构（如字典和列表）中指定键序列的值，如果未找到则返回默认值``default``。
# 如果指定了``no_default``，则可能引发``KeyError``或``IndexError``异常。
def get_in(keys, coll, default=None, no_default=False):
    try:
        # 使用``reduce``和``operator.getitem``获取嵌套数据结构中指定键序列的值
        return reduce(operator.getitem, keys, coll)
    except (KeyError, IndexError, TypeError):
        # 如果指定了``no_default``，则抛出异常；否则返回默认值``default``
        if no_default:
            raise
        return default


# 根据索引返回一个函数，该函数能从给定对象中获取相应的项
def getter(index):
    if isinstance(index, list):
        if len(index) == 1:
            # 如果索引是单个元素，返回一个函数，该函数返回对象中对应索引的项
            index = index[0]
            return lambda x: (x[index],)
        elif index:
            # 如果索引是列表且不为空，返回``operator.itemgetter``函数，该函数能获取对象中多个索引位置的项
            return operator.itemgetter(*index)
        else:
            # 如果索引为空列表，返回一个函数，该函数返回空元组
            return lambda x: ()
    else:
        # 如果索引不是列表，返回``operator.itemgetter``函数，该函数能获取对象中对应索引位置的项
        return operator.itemgetter(index)


# 根据指定的键函数对集合进行分组
def groupby(key, seq):
    if not callable(key):
        # 如果``key``不是可调用对象，将其转换为``getter``函数
        key = getter(key)
    # 使用``collections.defaultdict``创建一个默认值为``list.append``的字典
    d = collections.defaultdict(lambda: [].append)  # type: ignore[var-annotated]
    for item in seq:
        # 根据``key(item)``将``item``添加到字典``d``中对应键的列表中
        d[key(item)](item)
    rv = {}
    # 将``d``中的列表转换为其原始对象并存储在``rv``字典中
    for k, v in d.items():
        rv[k] = v.__self__  # type: ignore[var-annotated, attr-defined]
    return rv


# 返回序列中的第一个元素
def first(seq):
    return next(iter(seq))
```