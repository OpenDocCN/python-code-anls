# `.\pytorch\torch\fx\experimental\unify_refinements.py`

```py
# 导入必要的模块和类
# 允许未类型化的函数定义，用于静态类型检查
from torch.fx.experimental.graph_gradual_typechecker import Refine
# 导入张量类型
from torch.fx.tensor_type import TensorType
# 导入统一化相关函数，类型提示忽略定义属性
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]

# 对追踪过的模型进行单次符号类型推断
def infer_symbolic_types_single_pass(traced):
    """
    调用我们的符号推断器一次。
    """
    r = Refine(traced)
    r.refine()
    # 对约束进行统一化，得到最一般的解
    mgu = unify_eq(r.constraints)
    # 替换图中所有类型
    substitute_all_types(traced.graph, mgu)

# 对追踪过的模型进行双次符号类型推断
def infer_symbolic_types(traced):
    """
    调用我们的符号推断器两次。
    当一次推断不足以推断出所有信息时，这是有用的，例如广播的情况。
    """
    r = Refine(traced)
    r.refine()
    # 对约束进行统一化，得到最一般的解
    mgu = unify_eq(r.constraints)
    # 替换图中所有类型
    substitute_all_types(traced.graph, mgu)

    r = Refine(traced)
    r.refine()
    # 对约束进行统一化，得到最一般的解
    mgu = unify_eq(r.constraints)
    # 替换图中所有类型
    substitute_all_types(traced.graph, mgu)

    # 计算符号关系
    r.symbolic_relations()

# 将等式约束转换为统一化库可用的格式
def convert_eq(list_of_eq):
    """
    将等式约束转换为正确的格式，以便统一化库使用。
    """
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return tuple(lhs), tuple(rhs)

# 对一组等式约束应用统一化
def unify_eq(list_of_eq):
    """
    对一组等式约束应用统一化。
    """
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)

# 对类型映射应用最一般的统一化解
def substitute_solution_one_type(mapping, t):
    """
    对类型映射应用最一般的统一化解。
    """
    if isinstance(t, Var):
        if t in mapping.keys():
            return mapping[t]
        else:
            return t

    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.__args__:
            if typ in mapping.keys():
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))

    elif isinstance(t, list):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return new_type

    elif isinstance(t, tuple):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return tuple(new_type)

    else:
        return t

# 对图中的所有类型应用最一般的统一化解，直到达到固定点
def substitute_all_types(graph, mapping):
    """
    对图中的所有类型应用最一般的统一化解，直到达到固定点。
    如果输入和输出图相同，我们会收敛。
    """
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping.keys():
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True

    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)

# 检查两个图的类型是否相等，用于固定点检查
def check_for_type_equality(g1, g2):
    """
    用于固定点检查的类型相等性检查。
    """
    We do not use graph equality but instead type
    equality.
    """
    # 依次遍历两个图的节点列表，并比较它们的类型
    for n, m in zip(g1.nodes, g2.nodes):
        # 如果两个节点的类型不相同，返回 False
        if n.type != m.type:
            return False
    # 如果所有节点类型相同，则返回 True
    return True
```