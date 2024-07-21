# `.\pytorch\tools\code_analyzer\gen_op_registration_allowlist.py`

```
"""
This util is invoked from cmake to produce the op registration allowlist param
for `ATen/gen.py` for custom mobile build.
For custom build with dynamic dispatch, it takes the op dependency graph of ATen
and the list of root ops, and outputs all transitive dependencies of the root
ops as the allowlist.
For custom build with static dispatch, the op dependency graph will be omitted,
and it will directly output root ops as the allowlist.
"""

from __future__ import annotations  # 使用未来版本的注解特性

import argparse  # 导入解析命令行参数的模块
from collections import defaultdict  # 导入默认字典，支持默认值的字典
from typing import Dict, Set  # 导入类型提示相关的模块

import yaml  # 导入 YAML 文件解析库


DepGraph = Dict[str, Set[str]]  # 定义依赖图类型为字典，键为字符串，值为集合


def canonical_name(opname: str) -> str:
    # 返回操作名称的规范形式，去掉可能存在的重载部分
    return opname.split(".", 1)[0]


def load_op_dep_graph(fname: str) -> DepGraph:
    with open(fname) as stream:
        result = defaultdict(set)  # 创建一个默认字典，值默认为空集合
        for op in yaml.safe_load(stream):  # 从 YAML 文件加载数据
            op_name = canonical_name(op["name"])  # 获取操作名称的规范形式
            for dep in op.get("depends", []):  # 遍历操作的依赖列表
                dep_name = canonical_name(dep["name"])  # 获取依赖操作名称的规范形式
                result[op_name].add(dep_name)  # 将依赖关系添加到依赖图中
        return dict(result)  # 返回标准字典形式的依赖图


def load_root_ops(fname: str) -> list[str]:
    result = []
    with open(fname) as stream:
        for op in yaml.safe_load(stream):  # 从 YAML 文件加载数据
            result.append(canonical_name(op))  # 添加规范形式的操作名称到结果列表
    return result  # 返回规范形式的操作名称列表


def gen_transitive_closure(
    dep_graph: DepGraph,
    root_ops: list[str],
    train: bool = False,
) -> list[str]:
    result = set(root_ops)  # 初始化结果为根操作集合的副本
    queue = root_ops.copy()  # 初始化队列为根操作列表的副本

    # 依赖图中可能包含特殊条目 '__BASE__'，其值为要始终包含在定制构建中的基本操作集合
    queue.append("__BASE__")

    # 如果训练标志为真，依赖图中可能包含特殊条目 '__ROOT__'，其值为从 C++ 函数可达的操作集合
    # 将 '__ROOT__' 添加到队列中，以便包含可以直接从 C++ 代码调用的操作
    if train:
        queue.append("__ROOT__")

    while queue:  # 遍历队列直到为空
        cur = queue.pop()  # 弹出当前操作
        for dep in dep_graph.get(cur, []):  # 遍历当前操作的依赖
            if dep not in result:  # 如果依赖不在结果集合中
                result.add(dep)  # 将依赖添加到结果集合中
                queue.append(dep)  # 将依赖添加到队列中继续处理

    return sorted(result)  # 返回排序后的结果集合


def gen_transitive_closure_str(dep_graph: DepGraph, root_ops: list[str]) -> str:
    return " ".join(gen_transitive_closure(dep_graph, root_ops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Util to produce transitive dependencies for custom build"
    )
    parser.add_argument(
        "--op-dependency",
        help="input yaml file of op dependency graph "
        "- can be omitted for custom build with static dispatch",
    )
    # 添加一个命令行参数，用于指定根操作符的配置文件路径，参数名为 "--root-ops"
    parser.add_argument(
        "--root-ops",
        required=True,  # 必须指定该参数
        help="input yaml file of root (directly used) operators",  # 参数的帮助信息，描述参数用途
    )
    
    # 解析命令行参数，并将结果存储在 args 变量中
    args = parser.parse_args()
    
    # 如果命令行参数中指定了操作依赖文件 args.op_dependency，则加载操作依赖关系图，否则使用空字典
    deps = load_op_dep_graph(args.op_dependency) if args.op_dependency else {}
    
    # 加载根操作符配置文件 args.root_ops，返回根操作符的数据结构
    root_ops = load_root_ops(args.root_ops)
    
    # 根据加载的操作依赖关系图 deps 和根操作符 root_ops，生成传递闭包的字符串表示，并打印输出
    print(gen_transitive_closure_str(deps, root_ops))
```