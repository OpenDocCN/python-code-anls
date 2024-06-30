# `D:\src\scipysrc\sympy\sympy\utilities\memoization.py`

```
# 导入 functools 模块中的 wraps 装饰器函数
from functools import wraps

# 定义一个装饰器函数 recurrence_memo，用于处理由递归定义的序列的记忆化
def recurrence_memo(initial):
    """
    Memo decorator for sequences defined by recurrence

    Examples
    ========

    >>> from sympy.utilities.memoization import recurrence_memo
    >>> @recurrence_memo([1]) # 0! = 1
    ... def factorial(n, prev):
    ...     return n * prev[-1]
    >>> factorial(4)
    24
    >>> factorial(3) # use cache values
    6
    >>> factorial.cache_length() # cache length can be obtained
    5
    >>> factorial.fetch_item(slice(2, 4))
    [2, 6]

    """
    # 初始化缓存为初始值
    cache = initial

    # 定义装饰器函数 decorator
    def decorator(f):
        @wraps(f)
        def g(n):
            # 获取当前缓存的长度
            L = len(cache)
            # 如果请求的 n 小于缓存长度，直接返回缓存中的值
            if n < L:
                return cache[n]
            # 如果请求的 n 超过了缓存长度，根据递归定义生成缺失的值并添加到缓存中
            for i in range(L, n + 1):
                cache.append(f(i, cache))
            # 返回生成的值
            return cache[-1]
        
        # 添加额外的方法到函数 g 中
        g.cache_length = lambda: len(cache)  # 获取缓存长度的函数
        g.fetch_item = lambda x: cache[x]    # 获取指定索引范围内的缓存值的函数
        return g
    
    # 返回装饰器函数
    return decorator


# 定义一个装饰器函数 assoc_recurrence_memo，用于处理从基础序列开始定义的相关序列的记忆化
def assoc_recurrence_memo(base_seq):
    """
    Memo decorator for associated sequences defined by recurrence starting from base

    base_seq(n) -- callable to get base sequence elements

    XXX works only for Pn0 = base_seq(0) cases
    XXX works only for m <= n cases
    """

    # 初始化缓存为空列表
    cache = []

    # 定义装饰器函数 decorator
    def decorator(f):
        @wraps(f)
        def g(n, m):
            # 获取当前缓存的长度
            L = len(cache)
            # 如果请求的 n 小于缓存长度，直接返回缓存中的值
            if n < L:
                return cache[n][m]

            # 如果请求的 n 超过了缓存长度，根据递归定义生成缺失的值并添加到缓存中
            for i in range(L, n + 1):
                # 获取基础序列的值
                F_i0 = base_seq(i)
                F_i_cache = [F_i0]
                cache.append(F_i_cache)

                # XXX 只适用于 m <= n 的情况
                # 生成相关序列
                for j in range(1, i + 1):
                    F_ij = f(i, j, cache)
                    F_i_cache.append(F_ij)

            # 返回生成的值
            return cache[n][m]

        # 返回装饰后的函数
        return g
    
    # 返回装饰器函数
    return decorator
```