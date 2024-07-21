# `.\pytorch\torch\_inductor\optimize_indexing.py`

```py
# mypy: allow-untyped-defs
# 引入math模块，用于数学运算
import math

# 引入sympy模块，用于符号计算
import sympy

# 引入torch模块，用于深度学习任务
import torch
# 从torch.utils._sympy.value_ranges中引入ValueRanges类
from torch.utils._sympy.value_ranges import ValueRanges
# 从当前包中的ir模块中引入LoopBody类
from .ir import LoopBody
# 从当前包中的utils模块中引入dominated_nodes函数
from .utils import dominated_nodes


# 定义函数val_expressable_in_32_bits，用于判断值是否可以用32位表示
def val_expressable_in_32_bits(val):
    # 如果val具有is_Boolean属性，返回True
    if getattr(val, "is_Boolean", False):
        return True

    # 如果val是sympy.Expr类型
    if isinstance(val, sympy.Expr):
        assert val.is_number
        # 如果val是整数或布尔类型，转换为整数；否则转换为浮点数
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)

    # 如果val是浮点数，判断其是否在32位浮点数的范围内
    if isinstance(val, float):
        return val <= (2**24) and val >= -(2**24)

    # 如果val是整数，使用torch.iinfo获取int32的最大最小值范围，判断val是否在范围内
    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min

    # 如果val不是以上类型，则抛出TypeError异常
    raise TypeError(f"Unexpected value {val}")


# 定义函数range_expressable_in_32_bits，判断范围是否可以用32位表示
def range_expressable_in_32_bits(range):
    # 调用val_expressable_in_32_bits函数判断range.lower和range.upper是否可以用32位表示
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


# 定义函数try_to_reduce_precision，尝试减少节点的精度
def try_to_reduce_precision(node, bounds, indirect_vars, indices, replacement_vals):
    # 定义内部函数skip_filter，用于过滤掉已经明确转换为int32或特定浮点数类型的节点
    def skip_filter(node):
        return node.target == "to_dtype" and node.args[2] in (
            torch.int32,
            torch.float32,
            torch.float64,
        )

    # 遍历dominated_nodes([node], skip_filter)生成的节点列表
    # skip_filter函数用于过滤不需要考虑精度的节点
    for dominated in dominated_nodes([node], skip_filter):
        # 如果dominated的target是"store"或"output"，跳过处理
        if dominated.target in ["store", "output"]:
            continue

        # 如果dominated的target是字符串且以"set_indirect"开头
        if isinstance(dominated.target, str) and "set_indirect" in dominated.target:
            idx = int(dominated.target[len("set_indirect"):])
            indirect_var = indirect_vars[idx]

            # 检查所有涉及的索引是否可以用int32表示
            for index, expr in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]

                    # 如果index_val的下界或上界为无穷大，则返回
                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return

                    # 将index_val转换为整数表示
                    index_val_int = ValueRanges[sympy.Expr](
                        int(index_val.lower), int(index_val.upper)
                    )
                    # 检查index_val_int是否可以用32位表示
                    if not range_expressable_in_32_bits(index_val_int):
                        return

        # 检查bounds[dominated]是否可以用32位表示
        if not range_expressable_in_32_bits(bounds[dominated]):
            return

    # 将node.args转换为列表，并将第三个参数设置为torch.int32
    args = list(node.args)
    args[2] = torch.int32
    # 将参数列表 args 转换为元组，并赋值给节点的 args 属性
    node.args = tuple(args)
# 对循环体进行数据类型强度降级索引
def indexing_dtype_strength_reduction(loop_body: LoopBody):
    """
    对 LoopBody 的 fx 图执行值范围分析，将 int64 类型的中间变量精度降级为 int32
    """
    # 获取循环体的边界值
    bv = loop_body.bounds()

    # 找到所有目标为 "to_dtype"，且第三个参数为 torch.int64，并且不在未界定变量中的节点
    int64_dtype_nodes = [
        node
        for node in loop_body.get_nodes()
        if (
            node.target == "to_dtype"
            and node.args[2] == torch.int64
            and node not in bv.unbounded_vars
        )
    ]
    if not int64_dtype_nodes:
        return

    # 获取变量的边界
    bounds = bv.get_bounds()

    # TODO - 如果某个 to_dtype 节点的支配节点不能用 int32 表达，应该短路另一个 to_dtype 节点，如果该节点也支配了
    # 遍历所有 int64 类型的节点，尝试降低精度
    for node in int64_dtype_nodes:
        try_to_reduce_precision(
            node,
            bounds,
            loop_body.indirect_vars,
            loop_body.indexing_exprs,
            bv.replacement_vals,
        )
```