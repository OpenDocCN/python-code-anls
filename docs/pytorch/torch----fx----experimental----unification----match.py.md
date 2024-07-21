# `.\pytorch\torch\fx\experimental\unification\match.py`

```
# mypy: allow-untyped-defs
# 导入相关函数和类，从自定义模块中引入，忽略类型定义检查
from .core import unify, reify  # type: ignore[attr-defined]
from .variable import isvar
from .utils import _toposort, freeze
from .unification_tools import groupby, first  # type: ignore[import]

# 定义一个调度器类 Dispatcher
class Dispatcher:
    def __init__(self, name):
        self.name = name  # 初始化调度器名称
        self.funcs = {}   # 初始化函数字典
        self.ordering = []  # 初始化顺序列表

    # 添加函数及其签名到调度器中
    def add(self, signature, func):
        self.funcs[freeze(signature)] = func  # 冻结签名后作为键，函数作为值存储
        self.ordering = ordering(self.funcs)  # 根据函数字典重新确定顺序

    # 调用实例时的行为，解析参数并调用合适的函数
    def __call__(self, *args, **kwargs):
        func, s = self.resolve(args)  # 解析参数获取函数和解构结果
        return func(*args, **kwargs)  # 调用解析得到的函数

    # 解析参数，查找匹配的函数
    def resolve(self, args):
        n = len(args)  # 参数个数
        for signature in self.ordering:
            if len(signature) != n:  # 如果签名长度不符合参数个数，跳过
                continue
            s = unify(freeze(args), signature)  # 尝试统一参数与签名
            if s is not False:  # 如果可以统一
                result = self.funcs[signature]  # 获取对应函数
                return result, s  # 返回函数和统一结果
        # 如果未找到匹配的函数，抛出未实现错误
        raise NotImplementedError("No match found. \nKnown matches: "
                                  + str(self.ordering) + "\nInput: " + str(args))

    # 注册函数到调度器中
    def register(self, *signature):
        def _(func):
            self.add(signature, func)  # 调用 add 方法添加函数及其签名
            return self
        return _


# VarDispatcher 类继承自 Dispatcher 类
class VarDispatcher(Dispatcher):
    """ A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher('d')  # 创建 VarDispatcher 实例
    >>> x = var('x')  # 创建变量 x
    >>> @d.register('inc', x)  # 注册 inc 函数，带有 x 变量
    ... def f(x):
    ...     return x + 1  # 实现 inc 函数
    >>> @d.register('double', x)  # 注册 double 函数，带有 x 变量
    ... def f(x):
    ...     return x * 2  # 实现 double 函数
    >>> d('inc', 10)  # 调用 inc 函数，参数为 10
    11
    >>> d('double', 10)  # 调用 double 函数，参数为 10
    20
    """
    
    # 重写 __call__ 方法，解析参数并调用合适的函数，使用解构结果构建字典
    def __call__(self, *args, **kwargs):
        func, s = self.resolve(args)  # 解析参数获取函数和解构结果
        d = {k.token: v for k, v in s.items()}  # 构建解构结果的字典映射
        return func(**d)  # 调用解析得到的函数，传入解构结果的字典映射


global_namespace = {}  # type: ignore[var-annotated]

# match 函数定义，用于注册函数到命名空间的调度器中
def match(*signature, **kwargs):
    namespace = kwargs.get('namespace', global_namespace)  # 获取命名空间，默认为全局命名空间
    dispatcher = kwargs.get('Dispatcher', Dispatcher)  # 获取调度器类，默认为 Dispatcher

    def _(func):
        name = func.__name__  # 获取函数名称

        if name not in namespace:  # 如果函数名称不在命名空间中
            namespace[name] = dispatcher(name)  # 创建新的调度器实例并加入命名空间
        d = namespace[name]  # 获取对应调度器实例

        d.add(signature, func)  # 向调度器实例中添加函数及其签名

        return d
    return _

# 判断是否 a 比 b 更具体的函数
def supercedes(a, b):
    """ ``a`` is a more specific match than ``b`` """
    if isvar(b) and not isvar(a):  # 如果 b 是变量而 a 不是变量，则 a 更具体
        return True
    s = unify(a, b)  # 尝试统一 a 和 b
    if s is False:  # 如果无法统一，则 a 不是比 b 更具体的匹配
        return False
    s = {k: v for k, v in s.items() if not isvar(k) or not isvar(v)}  # 过滤掉变量对应的统一结果
    if reify(a, s) == a:  # 如果 a 在 s 下重构后与原始 a 相同，则 a 更具体
        return True
    if reify(b, s) == b:  # 如果 b 在 s 下重构后与原始 b 相同，则 a 不是比 b 更具体的匹配
        return False

# 比较函数 a 和 b 的顺序，a 应该先于 b 进行检查
def edge(a, b, tie_breaker=hash):
    """ A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):  # 如果 a 比 b 更具体
        if supercedes(b, a):  # 如果 b 也比 a 更具体
            return tie_breaker(a) > tie_breaker(b)  # 使用 tie_breaker 比较 a 和 b
        else:
            return True  # 否则应该先检查 a
    return False  # 否则应该先检查 b

# 返回一个合理的签名顺序列表，首先检查签名
def ordering(signatures):
    """ A sane ordering of signatures to check, first to last
    """
    # 将签名列表转换为元组列表，并存储在 signatures 变量中
    signatures = list(map(tuple, signatures))
    
    # 使用 edge 函数和 signatures 列表生成所有符合条件的边的列表，存储在 edges 变量中
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    
    # 按照第一个元素（first）进行分组 edges 列表，并存储在 edges 变量中
    edges = groupby(first, edges)
    
    # 对 signatures 列表中的每个元素 s 进行遍历
    for s in signatures:
        # 如果 s 不在 edges 中，则在 edges 中创建一个空列表
        if s not in edges:
            edges[s] = []
    
    # 将 edges 变量转换为字典，其中每个键是 signatures 中的元素，对应的值是 edges 中对应元素的第二个值列表
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[attr-defined, assignment]
    
    # 调用 _toposort 函数，并传入 edges 字典作为参数，返回拓扑排序结果
    return _toposort(edges)
```