# `.\pytorch\torch\fx\experimental\rewriter.py`

```
# mypy: allow-untyped-defs
# 引入 AST 模块，用于处理抽象语法树
import ast
# inspect 模块用于获取源代码并检查其结构
import inspect
# textwrap 模块用于格式化源代码
import textwrap
# copy 模块用于复制对象
import copy
# functools 模块提供了一些函数修饰器，用于操作函数对象
import functools
# 从 types 模块中导入 FunctionType 类型，表示函数对象
from types import FunctionType
# 从 typing 模块中导入各种类型提示
from typing import cast, Union, Callable, Dict, Optional, Any
# 从 torch.fx._symbolic_trace 模块中导入 Tracer 类
from torch.fx._symbolic_trace import Tracer
# 从 torch.fx.graph 模块中导入 Graph 类
from torch.fx.graph import Graph
# 从 torch._sources 模块中导入 normalize_source_lines 函数
from torch._sources import normalize_source_lines
# 导入 torch 库
import torch

# 定义 AST_Rewriter 类，继承自 ast.NodeTransformer 类
class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    # 此装饰器函数用于禁用 TorchDynamo 对象的动态特性
    @torch._dynamo.disable
    def rewrite(self, fn: FunctionType):
        # 获取函数 fn 的源代码行，并进行标准化处理
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = ''.join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # 解析标准化后的源代码，生成 AST
        source_ast = ast.parse(normalized_str)
        # 对 AST 进行重写操作，将不可跟踪的节点替换为 FX 替代调用
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # 编译新生成的 AST，并执行以获取编译后的函数对象
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) == 1
        fn_compiled = globals_dict[new_keys[0]]

        # 定义一个函数，用于更改函数对象的全局变量
        def change_func_globals(f, globals):
            """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
            # 复制函数对象 f 的所有成员，除了 __globals__ 属性
            g = FunctionType(
                f.__code__,
                globals,
                name=f.__name__,
                argdefs=f.__defaults__,
                closure=f.__closure__,
            )
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type:ignore[attr-defined]
            return g

        # 返回修改后的函数对象，保持其原始全局变量
        return change_func_globals(fn_compiled, globals=fn.__globals__)
    # 处理 Assert 节点的访问，将 Python 的 `assert` 替换为对可符号跟踪的 torch._assert 函数的调用
    def visit_Assert(self, node):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # 创建调用节点
        n = ast.parse('torch._assert()', mode='eval')
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        # 确定消息内容，如果不存在则创建空的常量节点
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        # 设置调用节点的参数为原 assert 节点的测试条件和消息
        call_node.args = [node.test, msg]

        # 确保新节点符合 Python AST 语法
        expr_wrapper = ast.Expr(value=call_node)

        # 返回更新位置信息后的新调用节点，表示我们要将其用作原始 _assert 节点的替代
        return ast.copy_location(expr_wrapper, node)

    # 处理 AnnAssign 节点的访问，将 Python 的 AnnAssign 替换为调用注解函数的 Assign 节点
    def visit_AnnAssign(self, node):
        """
        Swap out Python's AnnAssign with an Assign node where the annotation function is called.
        Example:
             Original:
             y: Tensor_Type(1,2,3, Dyn) = f2(x)
            Output:
             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))
        """
        # 返回一个新的 Assign 节点，其中的值为调用注解函数 annotate 的 Call 节点
        return ast.Assign(targets=[node.target], value=ast.Call(
            func=ast.Name(id='annotate', ctx=ast.Load()),
            args=[node.value, node.annotation], keywords=[]))
class RewritingTracer(Tracer):
    # 定义一个 RewritingTracer 类，继承自 Tracer 类
    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        # 重写 trace 方法，接受一个根节点（可以是 torch.nn.Module 或 Callable 类型）和可选的具体参数字典，返回一个图对象
        return super().trace(_rewrite(root), concrete_args)

# 定义一个内部函数 _rewrite，接受一个 torch.nn.Module 或 Callable 对象，返回一个同类型对象
def _rewrite(fn: Union[torch.nn.Module, Callable]) -> Union[torch.nn.Module, Callable]:
    if isinstance(fn, torch.nn.Module):
        # 如果输入是 torch.nn.Module 类型
        # 重写此模块的 `forward` 方法以及所有递归后代模块的 `forward` 方法。返回新的重写模块层次结构。
        def rewrite_module(m : torch.nn.Module):
            # 定义一个内部函数 rewrite_module，接受一个 torch.nn.Module 对象 m
            class RewrittenModule(torch.nn.Module):
                # 定义一个内部类 RewrittenModule，继承自 torch.nn.Module
                def __init__(self, orig):
                    super().__init__()
                    # 初始化方法，复制原始模块的所有属性到新模块中
                    for k, v in orig.__dict__.items():
                        if isinstance(v, torch.nn.Module):
                            self.__dict__[k] = copy.copy(rewrite_module(v))
                            # 如果属性值是 torch.nn.Module 类型，则递归调用 rewrite_module 复制
                        else:
                            self.__dict__[k] = copy.copy(v)
                            # 否则直接复制属性值

            # 为 RewrittenModule 类动态添加 forward 方法，通过 AST_Rewriter().rewrite 重写模块的前向方法
            RewrittenModule.forward = AST_Rewriter().rewrite(cast(FunctionType, m.forward))
            return RewrittenModule(m)
        
        # 返回调用 rewrite_module 处理后的 fn 对象
        return rewrite_module(fn)
    else:
        # 如果输入是 Callable 类型
        # 重写这个单独的自由函数
        return AST_Rewriter().rewrite(cast(FunctionType, fn))
        # 通过 AST_Rewriter().rewrite 重写函数类型对象的 AST 结构
```