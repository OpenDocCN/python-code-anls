# `D:\src\scipysrc\sympy\sympy\logic\algorithms\minisat22_wrapper.py`

```
from sympy.assumptions.cnf import EncodedCNF

# 定义函数用于判断 CNF 表达式是否可满足
def minisat22_satisfiable(expr, all_models=False, minimal=False):

    # 如果输入的表达式不是 EncodedCNF 类型，则将其转换为 EncodedCNF 对象
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    # 导入 Minisat22 求解器
    from pysat.solvers import Minisat22

    # 如果 CNF 数据中包含 {0}，则返回 UNSAT (不可满足)
    if {0} in expr.data:
        if all_models:
            # 如果需要返回所有模型，则返回一个生成器，仅包含 False
            return (f for f in [False])
        # 否则直接返回 False
        return False

    # 创建 Minisat22 求解器对象并传入 CNF 数据
    r = Minisat22(expr.data)

    # 如果需要最小化解，设置变量的相位为负数
    if minimal:
        r.set_phases([-(i+1) for i in range(r.nof_vars())])

    # 解决 CNF 表达式
    if not r.solve():
        return False

    # 如果不需要返回所有模型，则返回一个字典，映射变量名到布尔值
    if not all_models:
        return {expr.symbols[abs(lit) - 1]: lit > 0 for lit in r.get_model()}

    else:
        # 如果需要返回所有模型，创建一个生成器函数
        def _gen(results):
            satisfiable = False
            while results.solve():
                # 获取当前模型
                sol = results.get_model()
                # 生成一个字典，映射变量名到布尔值
                yield {expr.symbols[abs(lit) - 1]: lit > 0 for lit in sol}
                # 如果是最小化模式，添加一个条款以排除当前模型
                if minimal:
                    results.add_clause([-i for i in sol if i > 0])
                else:
                    results.add_clause([-i for i in sol])
                satisfiable = True
            # 如果没有可满足的模型，则生成一个 False
            if not satisfiable:
                yield False
            raise StopIteration

        # 返回生成器函数
        return _gen(r)
```