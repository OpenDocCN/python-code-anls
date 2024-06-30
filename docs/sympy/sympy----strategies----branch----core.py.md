# `D:\src\scipysrc\sympy\sympy\strategies\branch\core.py`

```
""" Generic SymPy-Independent Strategies """


# 定义一个生成器函数，返回其输入值
def identity(x):
    yield x


# 定义一个函数，用于重复应用一个分支规则，直到规则不再产生变化
def exhaust(brule):
    """ Apply a branching rule repeatedly until it has no effect """
    def exhaust_brl(expr):
        seen = {expr}  # 记录已经处理过的表达式集合
        for nexpr in brule(expr):
            if nexpr not in seen:
                seen.add(nexpr)
                yield from exhaust_brl(nexpr)
        if seen == {expr}:  # 如果集合中仅包含初始表达式，表示没有变化
            yield expr
    return exhaust_brl


# 定义一个函数，当应用分支规则时调用指定的函数，处理规则前后的表达式
def onaction(brule, fn):
    def onaction_brl(expr):
        for result in brule(expr):
            if result != expr:
                fn(brule, expr, result)  # 调用指定的处理函数
            yield result
    return onaction_brl


# 定义一个函数，用于在规则应用时将输入和输出的表达式打印出来
def debug(brule, file=None):
    """ Print the input and output expressions at each rule application """
    if not file:
        from sys import stdout
        file = stdout

    def write(brl, expr, result):
        file.write("Rule: %s\n" % brl.__name__)  # 打印规则的名称
        file.write("In: %s\nOut: %s\n\n" % (expr, result))  # 打印输入和输出的表达式

    return onaction(brule, write)


# 定义一个函数，将多个分支规则合并成一个规则
def multiplex(*brules):
    """ Multiplex many branching rules into one """
    def multiplex_brl(expr):
        seen = set()  # 记录已经处理过的表达式集合
        for brl in brules:
            for nexpr in brl(expr):
                if nexpr not in seen:
                    seen.add(nexpr)
                    yield nexpr
    return multiplex_brl


# 定义一个函数，根据条件选择是否应用分支规则
def condition(cond, brule):
    """ Only apply branching rule if condition is true """
    def conditioned_brl(expr):
        if cond(expr):  # 如果条件成立则应用规则
            yield from brule(expr)
        else:
            pass
    return conditioned_brl


# 定义一个函数，只输出满足条件的结果
def sfilter(pred, brule):
    """ Yield only those results which satisfy the predicate """
    def filtered_brl(expr):
        yield from filter(pred, brule(expr))
    return filtered_brl


# 定义一个函数，确保至少输出一个结果
def notempty(brule):
    def notempty_brl(expr):
        yielded = False
        for nexpr in brule(expr):
            yielded = True
            yield nexpr
        if not yielded:
            yield expr
    return notempty_brl


# 定义一个函数，执行多个分支规则中的任意一个
def do_one(*brules):
    """ Execute one of the branching rules """
    def do_one_brl(expr):
        yielded = False
        for brl in brules:
            for nexpr in brl(expr):
                yielded = True
                yield nexpr
            if yielded:
                return
    return do_one_brl


# 定义一个函数，按顺序组合多个规则，依次应用到表达式上
def chain(*brules):
    """
    Compose a sequence of brules so that they apply to the expr sequentially
    """
    def chain_brl(expr):
        if not brules:
            yield expr
            return

        head, tail = brules[0], brules[1:]
        for nexpr in head(expr):
            yield from chain(*tail)(nexpr)

    return chain_brl


# 定义一个函数，将普通规则转换为分支规则
def yieldify(rl):
    """ Turn a rule into a branching rule """
    def brl(expr):
        yield rl(expr)
    return brl
```