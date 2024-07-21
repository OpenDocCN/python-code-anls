# `.\pytorch\torch\_inductor\fx_passes\dedupe_symint_uses.py`

```
# 设置类型检查时允许未标记的函数定义
# 导入必要的模块和类型
from dataclasses import dataclass
from typing import Union

import torch
# 从 torch.fx.experimental.proxy_tensor 模块导入特定的符号类型
from torch.fx.experimental.proxy_tensor import py_sym_types, SymBool, SymFloat, SymInt


@dataclass
class _SymExprHash:
    """
    用于对 py_sym_types 进行哈希的类，将使用底层的 sympy 表达式
    """

    sym_obj: Union[SymInt, SymFloat, SymBool]

    def __hash__(self) -> int:
        # 返回基于符号对象类型和其节点表达式的哈希值
        return hash((type(self.sym_obj), self.sym_obj.node.expr))

    def __eq__(self, value) -> bool:
        # 检查两个 _SymExprHash 实例是否相等
        if not isinstance(value, _SymExprHash):
            return False
        return self.sym_obj.node.expr == value.sym_obj.node.expr


class _SymHashingDict:
    """
    包装了一个字典，将符号类型转换为使用 _SymExprHash 进行哈希，并重复使用现有的符号代理。

    由于 SymPy 的哈希不总是可靠的，因此乐观地使用 sympy 表达式进行哈希，如果失败，则回退到 symnodes。
    """

    def __init__(self):
        # 初始化空的符号哈希字典
        self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        # 将键值对存入字典，键会被包装成 _SymExprHash 进行存储
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        # 获取键对应的值
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        # 检查键是否在字典中
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def get(self, key, default=None):
        # 获取键对应的值，不存在时返回默认值
        return self.sym_hash_dict.get(self._wrap_to_sym_expr_hash(key), default)

    def _wrap_to_sym_expr_hash(self, key):
        # 如果键是 py_sym_types 类型，则包装成 _SymExprHash，否则直接返回键
        return _SymExprHash(key) if isinstance(key, py_sym_types) else key


def dedupe_symints(graph: torch.fx.Graph):
    """
    在图中去重符号整数，以便节点可以解析为符号整数的图输入。

    我们只从图输入处进行去重，以避免在正向传播中添加潜在的依赖关系。
    """

    # 创建符号哈希字典实例
    sym_dict = _SymHashingDict()
    # 可从图输入解析为符号整数的节点集合
    resolvable_from_input_symints = set()

    # 遍历图中的所有节点
    for node in graph.nodes:
        # 获取节点的元数据中的值，这里应该是一个符号类型的对象
        val = node.meta.get("val", None)
        # 如果值为空或者不是符号类型，则继续下一个节点
        if val is None or not isinstance(val, py_sym_types):
            continue

        # 如果节点操作是 "placeholder"
        if node.op == "placeholder":
            # 将该节点添加到可从图输入解析为符号整数的节点集合中
            resolvable_from_input_symints.add(node)
            # 将符号对象映射到该节点
            sym_dict[val] = node
        # 如果存在对应的节点
        elif existing_node := sym_dict.get(val):
            # 将当前节点的所有使用替换为已存在节点
            node.replace_all_uses_with(existing_node)
            # 从图中删除当前节点
            graph.erase_node(node)
        # 如果当前节点的所有输入节点都可以从图输入解析为符号整数
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            # 将符号对象映射到当前节点
            sym_dict[val] = node
            # 将当前节点添加到可从图输入解析为符号整数的节点集合中
            resolvable_from_input_symints.add(node)
```