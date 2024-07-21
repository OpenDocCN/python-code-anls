# `.\pytorch\torch\fx\traceback.py`

```py
# mypy: allow-untyped-defs
# 引入用于异常跟踪的模块
import traceback
# 引入上下文管理器相关模块
from contextlib import contextmanager
# 引入类型提示
from typing import List, Any, Dict
# 引入兼容性相关模块
from ._compatibility import compatibility

# 将以下函数添加到模块的公开接口中
__all__ = ['preserve_node_meta', 'has_preserved_node_meta',
           'set_stack_trace', 'set_grad_fn_seq_nr', 'reset_grad_fn_seq_nr',
           'format_stack', 'set_current_meta', 'get_current_meta']

# 当前节点的元数据字典，初始化为空
current_meta: Dict[str, Any] = {}
# 是否应保留节点元数据的标志，默认为False
should_preserve_node_meta = False


@compatibility(is_backward_compatible=False)
@contextmanager
# 保留节点元数据的上下文管理器
def preserve_node_meta():
    global should_preserve_node_meta
    global current_meta

    # 保存当前的should_preserve_node_meta和current_meta状态
    saved_should_preserve_node_meta = should_preserve_node_meta
    saved_current_meta = current_meta.copy()
    try:
        # 设置should_preserve_node_meta为True，表示需要保留节点元数据
        should_preserve_node_meta = True
        # yield语句之前的代码块执行完毕后，进入with语句块
        yield
    finally:
        # 在finally块中恢复之前保存的should_preserve_node_meta和current_meta状态
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@compatibility(is_backward_compatible=False)
# 设置堆栈跟踪信息的函数
def set_stack_trace(stack : List[str]):
    global current_meta

    # 如果应保留节点元数据且堆栈不为空，则将堆栈跟踪信息存入current_meta字典
    if should_preserve_node_meta and stack:
        current_meta["stack_trace"] = "".join(stack)


@compatibility(is_backward_compatible=False)
# 设置梯度函数序号的函数
def set_grad_fn_seq_nr(seq_nr):
    global current_meta

    # 如果应保留节点元数据，则将梯度函数序号和计数存入current_meta字典
    if should_preserve_node_meta:
        current_meta["grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", []) + [seq_nr]
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
# 重置梯度函数序号的函数
def reset_grad_fn_seq_nr():
    # 注意：正确地重置状态，这对支持可重新进入的自动微分非常有帮助
    global current_meta
    if should_preserve_node_meta:
        current_level = current_meta.get("in_grad_fn", 0)
        assert current_level > 0
        if current_level == 1:
            del current_meta["in_grad_fn"]
            del current_meta["grad_fn_seq_nr"]
        else:
            current_meta["in_grad_fn"] = current_level - 1
            current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
# 格式化堆栈信息的函数，返回一个字符串列表
def format_stack() -> List[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # 如果不应保留节点元数据，则回退到traceback.format_stack()函数
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
# 检查是否保留了节点元数据的函数，返回布尔值
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
# 设置当前元数据的上下文管理器，接收一个节点作为参数
def set_current_meta(node):
    global current_meta
    # 如果应保留节点元数据且节点有元数据存在，则执行以下操作
    if should_preserve_node_meta and node.meta:
        # 保存当前的元数据以便后续恢复
        saved_meta = current_meta
        try:
            # 将当前节点的元数据复制给current_meta，确保在操作期间不影响原始数据
            current_meta = node.meta.copy()

            # 如果当前元数据中没有"from_node"字段，则将(node.name, node.target)添加到"from_node"中以跟踪数据来源
            if "from_node" not in current_meta:
                current_meta["from_node"] = [(node.name, node.target)]
            elif current_meta["from_node"][-1][0] != node.name:
                # 如果"from_node"中最后一个元素的第一个值不等于当前节点的名称，则追加新的元组到"from_node"中
                current_meta["from_node"] = current_meta["from_node"] + [(node.name, node.target)]

            # 使用生成器语法，yield用于产生当前的执行结果
            yield
        finally:
            # 无论try块中的代码是否成功执行，都将恢复保存的元数据到current_meta中
            current_meta = saved_meta
    else:
        # 如果不需要保留节点元数据或节点没有元数据，则直接yield空结果
        yield
# 使用装饰器 @compatibility(is_backward_compatible=False) 标记该函数的兼容性属性为非向后兼容
@compatibility(is_backward_compatible=False)
# 定义一个函数 get_current_meta，其返回类型为字典，包含字符串键和任意类型的值
def get_current_meta() -> Dict[str, Any]:
    # 返回当前的元数据 current_meta
    return current_meta
```