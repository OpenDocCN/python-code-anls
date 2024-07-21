# `.\pytorch\torch\jit\_check.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块
import ast  # 提供对Python代码的抽象语法树的解析和操作
import inspect  # 提供检查活跃对象源代码的函数
import textwrap  # 提供文本包装和填充功能
import warnings  # 提供警告相关功能

import torch  # PyTorch深度学习框架


class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    """检查给定的 nn.Module 的 __init__ 方法。

    确保所有实例级别的属性可以正确初始化。

    具体来说，我们根据属性值进行类型推断......即使属性已经使用 Python3 风格的注释或 torch.jit.annotate 进行了类型标注。
    这意味着，将实例级别的属性设置为 []（对应 List）、{}（对应 Dict）或 None（对应 Optional）并不足以为我们提供足够的信息来正确初始化该属性。

    此类的对象可以遍历给定 nn.Module 的 AST，并确定是否符合我们的要求。

    已知限制：
    1. 我们只能检查AST节点的某些结构；我们不能对任意表达式进行“eval”评估。
       这意味着函数调用、类实例化以及解析为上述任一“空”值的复杂表达式都不会被标记为问题。
    2. 我们匹配字符串字面量，因此如果用户决定使用非标准导入（例如`from typing import List as foo`），我们将无法捕获它。

    示例：
        .. code-block:: python

            class M(torch.nn.Module):
                def fn(self):
                    return []

                def __init__(self):
                    super().__init__()
                    self.x: List[int] = []

                def forward(self, x: List[int]):
                    self.x = x
                    return 1

        上述代码将通过 AttributeTypeIsSupportedChecker 检查，因为我们在 __init__ 中有一个函数调用。
        但是，稍后会由于“RuntimeError”而失败：“尝试设置不存在的属性：x。您是否忘记在 __init__() 中初始化它？”。

    Args:
        nn_module - 我们希望检查其 __init__ 方法的 torch.nn.Module 实例
    """

    def check(self, nn_module: torch.nn.Module) -> None:
        # 获取 nn.Module.__init__ 方法的源代码行
        source_lines = inspect.getsource(nn_module.__class__.__init__)

        # 忽略任何缩进的注释行
        def is_useless_comment(line):
            line = line.strip()
            return line.startswith("#") and not line.startswith("# type:")

        # 过滤掉无用的注释行
        source_lines = "\n".join(
            [l for l in source_lines.split("\n") if not is_useless_comment(l)]
        )

        # 解析 `__init__` 方法的抽象语法树
        init_ast = ast.parse(textwrap.dedent(source_lines))

        # 获取类体中注解的项
        self.class_level_annotations = list(nn_module.__annotations__.keys())

        # 用于标记后续处理
        self.visiting_class_level_ann = False

        # 访问并处理 `__init__` 方法的抽象语法树
        self.visit(init_ast)
    def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool:
        # 如果注解类型为 "List"
        if ann_type == "List":
            # 检查节点是否为 ast.List 类型
            if not isinstance(node, ast.List):
                return False
            # 检查列表元素是否为空
            if node.elts:
                return False
        # 如果注解类型为 "Dict"
        elif ann_type == "Dict":
            # 检查节点是否为 ast.Dict 类型
            if not isinstance(node, ast.Dict):
                return False
            # 检查字典的键是否为空
            if node.keys:
                return False
        # 如果注解类型为 "Optional"
        elif ann_type == "Optional":
            # 检查节点是否为 ast.Constant 类型
            if not isinstance(node, ast.Constant):
                return False
            # 检查常量的值是否为 None
            if node.value:  # type: ignore[attr-defined]
                return False

        # 如果上述条件均不满足，则返回 True 表示容器为空
        return True

    def visit_Assign(self, node):
        """Store assignment state when assigning to a Call Node.

        If we're visiting a Call Node (the right-hand side of an
        assignment statement), we won't be able to check the variable
        that we're assigning to (the left-hand side of an assignment).
        Because of this, we need to store this state in visitAssign.
        (Luckily, we only have to do this if we're assigning to a Call
        Node, i.e. ``torch.jit.annotate``. If we're using normal Python
        annotations, we'll be visiting an AnnAssign Node, which has its
        target built in.)
        """
        try:
            # 如果节点的值是 ast.Call，并且目标是类级别注解中的属性
            if (
                isinstance(node.value, ast.Call)
                and node.targets[0].attr in self.class_level_annotations
            ):
                # 设置正在访问类级别注解标志为 True
                self.visiting_class_level_ann = True
        except AttributeError:
            return
        # 继续遍历节点的子节点
        self.generic_visit(node)
        # 在退出后重置正在访问类级别注解标志为 False
        self.visiting_class_level_ann = False
    # 定义一个方法，用于访问 `AnnAssign` 节点，这通常在 `nn.Module` 的 `__init__` 方法中发生
    def visit_AnnAssign(self, node):
        """Visit an AnnAssign node in an ``nn.Module``'s ``__init__`` method.

        It checks if it conforms to our attribute annotation rules."""
        
        # 如果我们有一个局部变量
        try:
            # 检查目标节点是否为 `self`
            if node.target.value.id != "self":
                return
        except AttributeError:
            return

        # 如果我们已经在类级别注释过这个属性
        if node.target.attr in self.class_level_annotations:
            return

        # TODO @ansley: add `Union` once landed

        # NB: 即使 `Tuple` 是一个容器，我们也不在这里检查它。`Tuple` 函数作为一种具有“无限”子类型的类型，
        # 意味着你可以有 `Tuple[()]`, `Tuple[T1]`, `Tuple[T2]`, `Tuple[T1, T2]`, `Tuple[T2, T1]` 等等，
        # 这些子类型不能互换使用。因此，在 `__init__` 中赋予空元组意味着该变量不能在稍后被重新赋值为非空元组。
        # `NamedTuple` 也是同样的情况

        containers = {"List", "list", "Dict", "dict", "Optional"}

        # 如果我们没有评估到指定的问题类型之一
        try:
            # 检查注解的值是否不在容器集合中
            if node.annotation.value.id not in containers:
                return
        except AttributeError:
            # 要评估基本类型（如 `str`、`int` 等），我们需要通过 `node.annotation.id` 而不是
            # `node.annotation.value.id` 获取名称。看起来我们没有评估到我们的“容器”之一
            return

        # 检查赋值的变量是否为空
        ann_type = node.annotation.value.id
        if not self._is_empty_container(node.value, ann_type):
            return

        # 发出警告，TorchScript 类型系统不支持在 `__init__` 中对空的非基本类型进行实例级别的注释。
        # 替代方式是：1) 在类体中使用类型注释，或者 2) 将类型包装在 `torch.jit.Attribute` 中
        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`."
        )
    # 定义方法 visit_Call(self, node)，用于处理 AST 中的 Call 节点
    def visit_Call(self, node):
        """Determine if a Call node is 'torch.jit.annotate' in __init__.

        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        """
        # 如果正在访问类级别的注解，则直接返回，不处理
        if self.visiting_class_level_ann:
            return

        # 如果这不是对 `torch.jit.annotate` 的调用
        try:
            if (
                node.func.value.value.id != "torch"
                or node.func.value.attr != "jit"
                or node.func.attr != "annotate"
            ):
                # 如果不符合 `torch.jit.annotate` 的调用形式，则进行通用访问
                self.generic_visit(node)
            elif (
                node.func.value.value.id != "jit" or node.func.value.attr != "annotate"
            ):
                # 如果 `torch.jit.annotate` 的调用形式不正确，则进行通用访问
                self.generic_visit(node)
        except AttributeError:
            # 如果节点结构不正确，无法检查是否是 `torch.jit.annotate` 的调用
            self.generic_visit(node)

        # 不变条件：我们有一个 `torch.jit.annotate` 或 `torch.annotate` 的调用

        # `torch.jit.annotate` 的调用节点应该有一个长度为 2 的 `args` 列表，
        # 其中 args[0] 表示注解，args[1] 表示实际值
        if len(node.args) != 2:
            return

        # 如果 `args[0]` 不是 `ast.Subscript` 类型，则直接返回
        if not isinstance(node.args[0], ast.Subscript):
            return

        # 在 `visit_AnnAssign` 中有关容器的注释

        # 可能的容器类型
        containers = {"List", "Dict", "Optional"}

        try:
            # 获取注解类型
            ann_type = node.args[0].value.id  # type: ignore[attr-defined]
        except AttributeError:
            return

        # 如果注解类型不在容器类型集合中，则直接返回
        if ann_type not in containers:
            return

        # 检查分配的变量是否为空
        if not self._is_empty_container(node.args[1], ann_type):
            return

        # 发出警告信息，说明 TorchScript 类型系统不支持在 `__init__` 中对空的非基本类型进行实例级别的注解
        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`."
        )
```