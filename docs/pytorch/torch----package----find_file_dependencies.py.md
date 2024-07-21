# `.\pytorch\torch\package\find_file_dependencies.py`

```
# mypy: allow-untyped-defs
# 导入ast模块，用于抽象语法树的解析
import ast
# 导入List、Optional和Tuple类型，用于类型提示
from typing import List, Optional, Tuple
# 从._importlib模块导入_resolve_name函数
from ._importlib import _resolve_name


class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    # 类方法，接收源代码和包名作为参数，返回一个元组列表
    def run(cls, src: str, package: str) -> List[Tuple[str, Optional[str]]]:
        visitor = cls(package)
        # 解析源代码生成抽象语法树
        tree = ast.parse(src)
        # 调用访问者对象处理抽象语法树
        visitor.visit(tree)
        # 返回访问者对象中收集的引用列表
        return list(visitor.references.keys())

    # 初始化方法，接收包名参数，并初始化references字典
    def __init__(self, package):
        super().__init__()
        self.package = package
        self.references = {}

    # 返回绝对模块名的方法，考虑了模块级别
    def _absmodule(self, module_name: str, level: int) -> str:
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    # 处理import语句的方法
    def visit_Import(self, node):
        # 遍历import语句中的别名列表，将每个别名加入references字典
        for alias in node.names:
            self.references[(alias.name, None)] = True

    # 处理from ... import ...语句的方法
    def visit_ImportFrom(self, node):
        # 计算模块的绝对名称
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        # 遍历from ... import ...语句中的别名列表
        for alias in node.names:
            # 如果别名不是"*"，则将模块和别名加入references字典
            if alias.name != "*":
                self.references[(name, alias.name)] = True
            else:
                # 否则，将模块加入references字典
                self.references[(name, None)] = True

    # 辅助方法，提取节点中整数值
    def _grab_node_int(self, node):
        return node.value

    # 辅助方法，提取节点中字符串值
    def _grab_node_str(self, node):
        return node.value
    # 处理对 `__import__` 的调用，不会路由到 visit_Import/From 节点
    if hasattr(node.func, "id") and node.func.id == "__import__":
        try:
            # 获取导入的模块名
            name = self._grab_node_str(node.args[0])
            # 初始化 fromlist 列表为空
            fromlist = []
            # 初始化 level 为 0
            level = 0
            # 如果 node.args 中超过3个元素，则从第4个参数开始遍历添加到 fromlist 中
            if len(node.args) > 3:
                for v in node.args[3].elts:
                    fromlist.append(self._grab_node_str(v))
            # 如果有关键字参数 keywords，则遍历关键字参数寻找名为 "fromlist" 的参数
            elif hasattr(node, "keywords"):
                for keyword in node.keywords:
                    if keyword.arg == "fromlist":
                        for v in keyword.value.elts:
                            fromlist.append(self._grab_node_str(v))
            # 如果 node.args 中超过4个元素，则从第5个参数开始获取 level 的值
            if len(node.args) > 4:
                level = self._grab_node_int(node.args[4])
            # 如果有关键字参数 keywords，则遍历关键字参数寻找名为 "level" 的参数
            elif hasattr(node, "keywords"):
                for keyword in node.keywords:
                    if keyword.arg == "level":
                        level = self._grab_node_int(keyword.value)
            # 如果 fromlist 为空列表，则表示导入的是顶级包
            if fromlist == []:
                # 将顶级包名（第一个点之前的部分）添加到引用字典中
                self.references[(name, None)] = True
                # 获取顶级包名的绝对路径
                top_name = name.rsplit(".", maxsplit=1)[0]
                # 如果顶级包名与原始名字不同，则继续处理顶级包名
                if top_name != name:
                    top_name = self._absmodule(top_name, level)
                    # 将顶级包名的绝对路径添加到引用字典中
                    self.references[(top_name, None)] = True
            else:
                # 否则，获取绝对路径的模块名
                name = self._absmodule(name, level)
                # 遍历 fromlist 列表中的每个别名，并添加到引用字典中
                for alias in fromlist:
                    # 如果别名不是 "*"，则将模块名与别名添加到引用字典中
                    if alias != "*":
                        self.references[(name, alias)] = True
                    else:
                        # 否则，将模块名与空别名添加到引用字典中
                        self.references[(name, None)] = True
        except Exception as e:
            # 发生异常时返回，不做任何处理
            return
# 将函数 _ExtractModuleReferences.run 赋值给变量 find_files_source_depends_on
find_files_source_depends_on = _ExtractModuleReferences.run
```