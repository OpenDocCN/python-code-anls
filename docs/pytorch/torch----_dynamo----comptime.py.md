# `.\pytorch\torch\_dynamo\comptime.py`

```
# 引入必要的模块和库
import builtins  # 引入内置模块
import dis  # 引入 dis 模块，用于反汇编 Python 字节码
import traceback  # 引入 traceback 模块，用于打印异常堆栈信息
from typing import Optional, Union  # 引入类型提示模块中的 Optional 和 Union 类型

import torch  # 引入 PyTorch 库
from torch.fx.experimental.symbolic_shapes import free_symbols  # 从 torch.fx.experimental.symbolic_shapes 中引入 free_symbols 函数

# 引入自定义异常模块
from .exc import unimplemented
# 引入自定义变量模块
from .variables import NewCellVariable
# 引入常量变量类
from .variables.constant import ConstantVariable
# 引入闭包变量类
from .variables.misc import ClosureVariable
# 引入符号节点变量类
from .variables.tensor import SymNodeVariable

class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """

    def __init__(self, v):
        # 初始化 ComptimeVar 类，传入参数 v 作为私有变量 __variable
        self.__variable = v

    def as_proxy(self):
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
        # 返回当前变量的 fx.Proxy 或者代表此变量的 fx.Proxy 列表/元组
        return self.__variable.as_proxy()

    def is_proxy(self):
        """
        Returns True if as_proxy() would succeed.
        """
        # 如果 as_proxy() 方法可以成功调用，则返回 True
        return self.__variable.is_proxy()

    def as_fake(self):
        """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
        # 返回表示当前变量的“虚拟”值，可以是 FakeTensor 或 SymInt
        return self.__variable.as_proxy().node.meta["example_value"]

    def size(self, dim: Optional[int] = None) -> Union[int, torch.SymInt]:
        """
        Returns the size of the tensor (if dim is None) or the size
        at the dimension dim.  The returned size may be a SymInt.
        """
        # 返回张量的大小（如果 dim 为 None），或者指定维度 dim 的大小，返回值可能是 SymInt
        return self.as_fake().size(dim)

    def python_type(self):
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
        # 返回变量在编译时的类型
        return self.__variable.python_type()
    def as_python_constant(self):
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
        # 调用私有变量 __variable 的 as_python_constant 方法，返回其编译时的常量值
        return self.__variable.as_python_constant()

    def is_python_constant(self):
        """
        Returns True if as_python_constant would succeed.
        """
        # 返回是否可以成功调用 as_python_constant 方法，即变量是否是编译时常量
        return self.__variable.is_python_constant()

    def is_dynamic(self):
        """
        Determines if the variable is dynamic (not fully known at compile-time).
        """
        if isinstance(self.__variable, SymNodeVariable):
            # 如果变量是 SymNodeVariable 类型，则获取其自由符号集合
            fs = free_symbols(self.__variable.sym_num)
            return bool(fs)  # 返回自由符号集合是否为空的布尔值
        return False  # 对于其他类型的变量，默认为静态（非动态）

    def force_static(self):
        """
        Forces that a value is static, inducing a guard on its specific value
        """
        if isinstance(self.__variable, SymNodeVariable):
            # 如果变量是 SymNodeVariable 类型，则评估其表达式以确定静态值
            self.__variable.evaluate_expr()
        elif isinstance(self.__variable, ConstantVariable):
            # 如果变量是 ConstantVariable 类型，则不需要强制静态化处理
            pass
        else:
            # 如果变量类型不在预期范围内，抛出断言错误
            raise AssertionError(
                f"cannot force {self.__variable} ({type(self.__variable)}) static"
            )

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
        # 返回私有变量 __variable，用于表示编译时的变量跟踪器
        return self.__variable

    def __repr__(self):
        # 返回调试用的字符串表示，调用私有变量 __variable 的 debug_repr 方法
        return self.__variable.debug_repr()

    # TODO: API for adding a custom guard
class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """

    def __init__(self, tx):
        self.__tx = tx

    def get_local(self, name: str, *, stacklevel=0) -> ComptimeVar:
        """
        Retrieve the compile-time known information about a local.
        """
        # 获取当前栈级别对应的 transaction 对象
        tx = self.__get_tx(stacklevel)

        # 这类似于 LOAD_DEREF 操作
        if hasattr(tx, "closure_cells") and name in tx.closure_cells:
            # 如果存在闭包变量且名称在闭包单元中
            cell = tx.closure_cells[name]
            if isinstance(cell, ClosureVariable):
                # 返回与闭包变量对应的符号化局部变量
                return ComptimeVar(tx.output.root_tx.symbolic_locals[cell.name])
            else:
                # 返回加载闭包单元的副作用
                return ComptimeVar(tx.output.side_effects.load_cell(cell))
        else:
            # 否则，返回符号化局部变量中对应名称的结果
            r = tx.symbolic_locals[name]
            if isinstance(r, NewCellVariable):
                # 返回加载新的单元变量的副作用
                return ComptimeVar(tx.output.side_effects.load_cell(r))
            else:
                # 返回符号化局部变量中对应名称的结果
                return ComptimeVar(r)

    def graph_break(self, msg="ComptimeContext.graph_break"):
        """
        Manually trigger a graph break
        """
        # 触发图断点，但未实现
        unimplemented(msg)

    def graph(self):
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
        # 返回部分构建的 FX 图，该图将在编译后传递给用户编译器
        return self.__tx.output.graph

    def assert_static(self, val):
        """
        Asserts that the int is static (and not dynamic, per dynamic shapes)
        """
        # 断言整数是静态的，而不是动态的（根据动态形状）
        assert (
            not val.is_dynamic()
        ), "expected static but got dynamic (run with TORCH_LOGS=dynamic for more info)"

    def print_graph(self, *, verbose=True, file=None):
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
        # 打印部分构建的 FX 图，该图将在编译后传递给用户编译器
        print(
            self.__tx.output.graph.python_code("self", verbose=verbose).src, file=file
        )

    def parent(self):
        """
        Returns a new ComptimeContext instance with parent transaction.
        """
        return ComptimeContext(self.__tx.parent)

    def __get_tx(self, stacklevel):
        """
        Retrieve the transaction object at the specified stack level.
        """
        tx = self.__tx
        for _ in range(stacklevel):
            tx = tx.parent
        return tx

    def print(self, val, *, file=None):
        """
        Print a representation of the value.
        """
        print(repr(val), file=file)

    def print_disas(self, *, file=None, stacklevel=0):
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
        # 获取指定栈级别的 transaction 对象，并打印当前执行的操作码序列
        tx = self.__get_tx(stacklevel)
        print(
            dis.Bytecode(
                tx.f_code,
                current_offset=tx.instructions[tx.instruction_pointer].offset,
            ).dis(),
            file=file,
        )
    def print_value_stack(self, *, file=None, stacklevel=0):
        """
        Print the current Python value stack.  Note that this is NOT the same
        as the traceback; use print_bt() to print that.  Note that at
        stacklevel=0, this will typically be empty, as comptime cannot
        currently be used in an expression context where there would be
        intermediates on the stack.  If you would find this useful, please
        file a bug at https://github.com/pytorch/pytorch/

        NB: Stack grows downwards in our print
        """
        # TODO: improve printing
        # 获取当前调用栈信息
        tx = self.__get_tx(stacklevel)
        # 遍历并打印当前值栈的内容
        for s in tx.stack:
            print(f"- {s}", file=file)

    def print_locals(self, *, file=None, stacklevel=0):
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
        # TODO: improve by improving the VariableTracker printing
        # 获取当前调用栈信息
        tx = self.__get_tx(stacklevel)
        # 遍历并打印当前上下文中的局部变量和其值
        for k, v in tx.symbolic_locals.items():
            print(f"{k} = {v}", file=file)

    def print_bt(self, *, file=None, stacklevel=0):
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
        # 初始化一个空列表来存储调用栈信息
        stack = []
        # 获取当前调用栈信息
        tx = self.__get_tx(stacklevel)
        # 从当前调用栈一直向上追溯到顶部，并将每一帧的摘要信息加入到stack列表中
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        # 将逆序的调用栈信息格式化输出
        print(
            "".join(traceback.StackSummary.from_list(reversed(stack)).format()),
            file=file,
        )

    def print_guards(self, *, file=None):
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
        # TODO: improve print format, current guard format is extremely
        # verbose
        # 打印当前上下文中已安装的保护器的详细信息
        print(
            "\n".join(f"{repr(guard)}" for guard in sorted(self.__tx.output.guards)),
            file=file,
        )

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
        # 返回用于跟踪符号评估状态的内部数据结构InstructionTranslator
        return self.__tx
class _Comptime:
    # 将给定的函数在编译时调用，如果失败则调用 fallback_fn
    @staticmethod
    def __call__(fn, fallback_fn=lambda: None):
        """fn gets called at compile time in TorchDynamo, calls fallback_fn otherwise"""
        fallback_fn()

    # 更加简洁的便捷包装器

    # 调用 ctx.graph_break() 方法
    @staticmethod
    def graph_break():
        comptime(lambda ctx: ctx.graph_break())

    # 打印表达式 e 的本地变量值
    @staticmethod
    def print(e):
        comptime(lambda ctx: ctx.print(ctx.get_local("e")), lambda: print(e))

    # 打印编译时的计算图
    @staticmethod
    def print_graph():
        comptime(lambda ctx: ctx.print_graph())

    # 打印当前的指令序列，带有调用堆栈信息
    @staticmethod
    def print_disas(*, stacklevel=0):
        comptime(
            lambda ctx: ctx.print_disas(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # 打印当前值堆栈的内容，带有调用堆栈信息
    @staticmethod
    def print_value_stack(*, stacklevel=0):
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # 打印当前值堆栈的内容并返回表达式 e，带有调用堆栈信息
    @staticmethod
    def print_value_stack_and_return(e, *, stacklevel=0):
        comptime(
            lambda ctx: ctx.print_value_stack(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )
        return e

    # 打印当前本地变量的值，带有调用堆栈信息
    @staticmethod
    def print_locals(*, stacklevel=0):
        comptime(
            lambda ctx: ctx.print_locals(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # 打印当前调用堆栈的信息，带有调用堆栈信息
    @staticmethod
    def print_bt(*, stacklevel=0):
        comptime(
            lambda ctx: ctx.print_bt(
                stacklevel=ctx.get_local("stacklevel").as_python_constant() + 1
            )
        )

    # 打印当前的保护器信息
    @staticmethod
    def print_guards():
        comptime(lambda ctx: ctx.print_guards())

    # 在编译时断点，相当于 pdb 的 breakpoint()
    @staticmethod
    def assert_static(val):
        comptime(lambda ctx: ctx.assert_static(ctx.get_local("val")))

    # 强制将 val 定义为静态值
    @staticmethod
    def force_static(val):
        comptime(lambda ctx: ctx.get_local("val").force_static())

    # 在编译时设置断点，调用 builtins.breakpoint() 进入调试器
    @staticmethod
    def breakpoint():
        """
        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime
            comptime.breakpoint()

        And then, inside pdb, you can access 'ctx' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        """

        def inner(inner_ctx):
            ctx = inner_ctx.parent()
            builtins.breakpoint()

        comptime(inner)


comptime = _Comptime()
```