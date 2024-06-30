# `D:\src\scipysrc\sympy\sympy\strategies\core.py`

```
""" Generic SymPy-Independent Strategies """
# 引入类型提示和特定模块
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout

# 定义类型变量
_S = TypeVar('_S')
_T = TypeVar('_T')

# 定义恒等函数，返回其输入
def identity(x: _T) -> _T:
    return x

# 定义递归应用规则函数
def exhaust(rule: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Apply a rule repeatedly until it has no effect """
    def exhaustive_rl(expr: _T) -> _T:
        new, old = rule(expr), expr
        while new != old:
            new, old = rule(new), new
        return new
    return exhaustive_rl

# 定义带缓存的规则函数
def memoize(rule: Callable[[_S], _T]) -> Callable[[_S], _T]:
    """Memoized version of a rule

    Notes
    =====

    This cache can grow infinitely, so it is not recommended to use this
    than ``functools.lru_cache`` unless you need very heavy computation.
    """
    cache: dict[_S, _T] = {}

    def memoized_rl(expr: _S) -> _T:
        if expr in cache:
            return cache[expr]
        else:
            result = rule(expr)
            cache[expr] = result
            return result
    return memoized_rl

# 定义条件应用规则函数
def condition(
    cond: Callable[[_T], bool], rule: Callable[[_T], _T]
) -> Callable[[_T], _T]:
    """ Only apply rule if condition is true """
    def conditioned_rl(expr: _T) -> _T:
        if cond(expr):
            return rule(expr)
        return expr
    return conditioned_rl

# 定义链式应用规则函数
def chain(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """
    Compose a sequence of rules so that they apply to the expr sequentially
    """
    def chain_rl(expr: _T) -> _T:
        for rule in rules:
            expr = rule(expr)
        return expr
    return chain_rl

# 定义调试函数
def debug(rule, file=None):
    """ Print out before and after expressions each time rule is used """
    if file is None:
        file = stdout

    def debug_rl(*args, **kwargs):
        expr = args[0]
        result = rule(*args, **kwargs)
        if result != expr:
            file.write("Rule: %s\n" % rule.__name__)
            file.write("In:   %s\nOut:  %s\n\n" % (expr, result))
        return result
    return debug_rl

# 定义空安全规则函数
def null_safe(rule: Callable[[_T], _T | None]) -> Callable[[_T], _T]:
    """ Return original expr if rule returns None """
    def null_safe_rl(expr: _T) -> _T:
        result = rule(expr)
        if result is None:
            return expr
        return result
    return null_safe_rl

# 定义异常处理规则函数
def tryit(rule: Callable[[_T], _T], exception) -> Callable[[_T], _T]:
    """ Return original expr if rule raises exception """
    def try_rl(expr: _T) -> _T:
        try:
            return rule(expr)
        except exception:
            return expr
    return try_rl

# 定义选择性应用规则函数
def do_one(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Try each of the rules until one works. Then stop. """
    def do_one_rl(expr: _T) -> _T:
        for rl in rules:
            result = rl(expr)
            if result != expr:
                return result
        return expr
    return do_one_rl

# 开始定义 switch 函数
def switch(
    key: Callable[[_S], _T],
    ruledict: Mapping[_T, Callable[[_S], _S]]



# key 是一个类型为 Callable 的变量，接受类型为 _S 的参数，返回类型为 _T 的结果
# ruledict 是一个类型为 Mapping 的变量，将类型为 _T 的键映射到接受类型为 _S 参数、返回类型为 _S 结果的 Callable 函数
# 选择一个基于函数 key 在结果上调用的规则
def switch_rl(expr: _S) -> _S:
    rl = ruledict.get(key(expr), identity)
    return rl(expr)

# 返回一个规则选择函数，用于最小化目标函数
def minimize(
    *rules: Callable[[_S], _T],  # 可变数量的规则函数，每个函数接受 _S 类型参数并返回 _T 类型结果
    objective=_identity  # 目标函数，默认为 _identity 函数
) -> Callable[[_S], _T]:
    """ Select result of rules that minimizes objective

    >>> from sympy.strategies import minimize
    >>> inc = lambda x: x + 1
    >>> dec = lambda x: x - 1
    >>> rl = minimize(inc, dec)
    >>> rl(4)
    3

    >>> rl = minimize(inc, dec, objective=lambda x: -x)  # maximize
    >>> rl(4)
    5
    """
    def minrule(expr: _S) -> _T:
        # 对于给定的表达式 expr，计算每个规则函数在 expr 上的结果，选择使目标函数最小化的结果
        return min([rule(expr) for rule in rules], key=objective)
    return minrule
```