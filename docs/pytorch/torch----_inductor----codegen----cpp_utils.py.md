# `.\pytorch\torch\_inductor\codegen\cpp_utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import contextlib                    # 上下文管理工具
import copy                          # 复制操作相关
import math                          # 数学函数

from collections import namedtuple   # 命名元组
from typing import Dict, List, Tuple # 类型注解
from unittest.mock import patch     # 单元测试的模拟装饰器

import sympy                         # 符号计算库

import torch                         # PyTorch 深度学习库
from torch.utils._sympy.symbol import symbol_is_type, SymT # 符号计算相关模块
from .. import ir                    # 从上层模块导入 IR 相关内容
from ..utils import IndentedBuffer, sympy_index_symbol_with_prefix # 实用工具和符号计算函数
from ..virtualized import V          # 虚拟化相关模块

from .common import CSEVariable, ExprPrinter, Kernel # 从当前目录的 common 模块导入特定类和函数

# 定义从 Torch 数据类型到 C++ 数据类型的映射
DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "int64_t",
    torch.int32: "int32_t",
    torch.int16: "int16_t",
    torch.int8: "int8_t",
    torch.uint64: "uint64_t",
    torch.uint32: "uint32_t",
    torch.uint16: "uint16_t",
    torch.uint8: "uint8_t",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
    torch.complex64: "complex64",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
}

# 定义从 Torch 数据类型到 ATen 数据类型的映射
DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint64: "at::kUInt64",
    torch.uint32: "at::kUInt32",
    torch.uint16: "at::kUInt16",
    torch.uint8: "at::kByte",
    torch.uint32: "at::kUInt32",  # 注意：重复的键
    torch.uint64: "at::kUInt64",  # 注意：重复的键
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
    torch.complex32: "at::kComplexHalf",
    torch.complex64: "at::kComplexFloat",
    torch.complex128: "at::kComplexDouble",
    torch.float8_e4m3fn: "at::kFloat8_e4m3fn",
    torch.float8_e5m2: "at::kFloat8_e5m2",
    torch.float8_e4m3fnuz: "at::kFloat8_e4m3fnuz",
    torch.float8_e5m2fnuz: "at::kFloat8_e5m2fnuz",
}

# 定义设备类型到 ATen 设备类型的映射
DEVICE_TO_ATEN = {
    "cpu": "at::kCPU",
    "cuda": "at::kCUDA",
}

# 定义布局类型到 ATen 布局类型的映射
LAYOUT_TO_ATEN = {
    torch.strided: "at::kStrided",
    torch._mkldnn: "at::kMkldnn",  # type: ignore[attr-defined]
}

# 定义索引类型
INDEX_TYPE = "long"

# 定义用于矩阵乘法块的命名元组
GemmBlocking = namedtuple("GemmBlocking", ["block_m", "block_n", "block_k"])

# 定义 CppPrinter 类，继承自 ExprPrinter
class CppPrinter(ExprPrinter):
    # 定义私有方法，用于打印整数
    def _print_Integer(self, expr):
        return f"{int(expr)}L"

    # 定义私有方法，用于打印 Where 表达式
    def _print_Where(self, expr):
        c = self.paren(self.doprint(expr.args[0]))
        p = self.paren(self.doprint(expr.args[1]))
        q = self.paren(self.doprint(expr.args[2]))
        return f"{c} ? {p} : {q}"

    # 定义私有方法，用于打印 ModularIndexing 表达式
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        if div != 1:
            div = self.paren(self.doprint(div))
            if expr.is_integer:
                x = f"c10::div_floor_integer({x}, {div})"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.paren(self.doprint(mod))
        return f"static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})"
    # 定义一个方法，用于打印整数除法表达式
    def _print_FloorDiv(self, expr):
        # 解构表达式中的参数
        x, div = expr.args
        # 对 x 和 div 进行打印，并使用括号包裹
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        # 如果表达式表示整数除法，返回对应的 C++ 代码
        if expr.is_integer:
            return f"c10::div_floor_integer({x}, {div})"
        # 如果是浮点数除法，返回对应的 C++ 代码
        return f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"

    # 定义一个方法，用于打印取整函数表达式
    def _print_floor(self, expr):
        # 断言表达式只有一个参数
        assert len(expr.args) == 1
        # 将参数打印成 std::floor 函数调用的形式
        r = f"std::floor({self._print(expr.args[0])})"
        # 如果表达式表示整数，将结果转换成 INDEX_TYPE 类型
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    # 定义一个方法，用于打印向下取整到整数表达式
    def _print_FloorToInt(self, expr):
        # 断言表达式只有一个参数
        assert len(expr.args) == 1
        # 将参数打印成 std::floor 函数调用的形式
        r = f"std::floor({self._print(expr.args[0])})"
        # 如果表达式表示整数，将结果转换成 INDEX_TYPE 类型
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    # 定义一个方法，用于打印截断到整数表达式
    def _print_TruncToInt(self, expr):
        # 断言表达式只有一个参数
        assert len(expr.args) == 1
        # 将参数打印成 std::trunc 函数调用的形式
        r = f"std::trunc({self._print(expr.args[0])})"
        # 将结果转换成 INDEX_TYPE 类型
        return f"static_cast<{INDEX_TYPE}>({r})"

    # 定义一个方法，用于打印截断到浮点数表达式
    def _print_TruncToFloat(self, expr):
        # 断言表达式只有一个参数
        assert len(expr.args) == 1
        # 将参数打印成 std::trunc 函数调用的形式
        return f"std::trunc({self._print(expr.args[0])})"

    # 定义一个方法，用于打印转换为浮点数表达式
    def _print_ToFloat(self, expr):
        # 断言表达式只有一个参数
        assert len(expr.args) == 1
        # 将参数打印成 static_cast<double> 函数调用的形式
        return f"static_cast<double>({self._print(expr.args[0])})"

    # TODO: 如果输入值有一个是负数，这段代码会出错。然而，我们通常能保证输入值是正数，
    # 因此我们会使用 Mod 函数，这时这段代码是正确的代码生成。
    def _print_PythonMod(self, expr):
        # 返回参数表达式中每个参数打印后用 "%" 连接的字符串
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # 定义一个方法，用于打印 C 语言风格取模表达式
    def _print_CMod(self, expr):
        # 返回参数表达式中每个参数打印后用 "%" 连接的字符串
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # 定义一个方法，用于打印整数真除表达式
    def _print_IntTrueDiv(self, expr):
        # 解构表达式中的左右参数
        lhs, rhs = expr.args
        # 返回左右参数打印后用 "/" 连接的 C++ 表达式
        return f"static_cast<double>({self._print(lhs)}) / static_cast<double>({self._print(rhs)})"

    # TODO: PowByNatural：我们需要实现自己的整数幂函数。不要使用操作浮点数的 std::pow 函数。
    def _print_PowByNatural(self, expr):
        # 抛出未实现错误，显示该函数未对当前类型实现
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    # 定义一个方法，用于打印浮点数真除表达式
    def _print_FloatTrueDiv(self, expr):
        # 解构表达式中的左右参数
        lhs, rhs = expr.args
        # 返回左右参数打印后用 "/" 连接的字符串
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    # 定义一个方法，用于打印浮点数幂表达式
    def _print_FloatPow(self, expr):
        # 解构表达式中的底数和指数
        base, exp = expr.args
        # 返回 std::pow 函数调用的字符串形式
        return f"std::pow({self._print(base)}, {self._print(exp)})"
    def _print_Pow(self, expr):
        # 对幂表达式进行打印处理
        base, exp = expr.args
        base = self._print(base)

        # 处理指数为 0.5 或 -0.5 的情况
        if exp == 0.5 or exp == -0.5:
            return f"std::sqrt({base})" if exp == 0.5 else f"1.0/std::sqrt({base})"
        
        # 处理整数指数的情况
        if exp.is_integer:
            exp = int(exp)
            if exp > 0:
                r = "*".join([self.paren(base)] * exp)
            elif exp < 0:
                r = "1.0/" + self.paren("*".join([self.paren(base)] * abs(exp)))
            else:  # exp == 0
                r = "1.0"

            # 返回静态转换类型后的结果
            return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r
        else:
            # 处理浮点数指数的情况
            # TODO: float vs double
            return f"std::pow({base}, {float(exp)})"

    def _print_Rational(self, expr):
        # 对有理数表达式进行打印处理，使用浮点常量进行浮点数除法
        if expr.q == 1:
            r = f"{expr.p}"
        else:
            r = f"{expr.p}.0/{expr.q}.0"
        # 返回静态转换类型后的结果
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_ceiling(self, expr):
        # 对天花板函数进行打印处理，向上取整
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        # 返回静态转换类型后的结果
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_CeilToInt(self, expr):
        # 对向上取整转为整数进行打印处理
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        # 返回静态转换类型后的结果
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_Min(self, expr):
        # 对最小值函数进行打印处理
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::min({args[0]}, {args[1]})"
        else:
            # 初始化列表重载
            il = "{" + ", ".join(args) + "}"
            return f"std::min({il})"

    def _print_Max(self, expr):
        # 对最大值函数进行打印处理
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::max({args[0]}, {args[1]})"
        else:
            # 初始化列表重载
            il = "{" + ", ".join(args) + "}"
            return f"std::max({il})"

    def _print_Abs(self, expr):
        # 对绝对值函数进行打印处理
        assert len(expr.args) == 1
        return f"std::abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr):
        # 对余弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        # 对双曲余弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr):
        # 对反余弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr):
        # 对正弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        # 对双曲正弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr):
        # 对反正弦函数进行打印处理
        assert len(expr.args) == 1
        return f"std::asin({self._print(expr.args[0])})"
    # 打印包含单参数的 tan 函数表达式
    def _print_OpaqueUnaryFn_tan(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回以 std::tan 包装表达式参数的字符串
        return f"std::tan({self._print(expr.args[0])})"

    # 打印包含单参数的 tanh 函数表达式
    def _print_OpaqueUnaryFn_tanh(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回以 std::tanh 包装表达式参数的字符串
        return f"std::tanh({self._print(expr.args[0])})"

    # 打印包含单参数的 atan 函数表达式
    def _print_OpaqueUnaryFn_atan(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # 返回以 std::atan 包装表达式参数的字符串
        return f"std::atan({self._print(expr.args[0])})"

    # 打印包含单参数的 sqrt 函数表达式
    def _print_OpaqueUnaryFn_sqrt(self, expr):
        # 返回以 std::sqrt 包装表达式参数的字符串
        return f"std::sqrt({self._print(expr.args[0])})"

    # 打印将表达式四舍五入到最接近的整数
    def _print_RoundToInt(self, expr):
        # 断言表达式参数只有一个
        assert len(expr.args) == 1
        # TODO: 根据索引类型调度到 llrint
        # 返回以 std::lrint 包装表达式参数的字符串
        return f"std::lrint({self._print(expr.args[0])})"

    # 打印将表达式四舍五入到最接近的小数点后指定位数的十进制数
    def _print_RoundDecimal(self, expr):
        # 断言表达式参数有两个
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # 对于整数输入，不支持负的小数位数 ndigits
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        # 返回以 static_cast<double>(std::nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits}) 包装表达式参数的字符串
        return f"static_cast<double>(std::nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits})"

    # 打印布尔值 true
    def _print_BooleanTrue(self, expr):
        return "true"

    # 打印布尔值 false
    def _print_BooleanFalse(self, expr):
        return "false"
# A function to print, useful for printing sympy symbols.
cexpr = CppPrinter().doprint

# Function to generate a C++ static cast expression with a specific index type
def cexpr_index(index):
    return f"static_cast<{INDEX_TYPE}>({cexpr(index)})"

# Function to convert Python value to C++ representation based on type
def value_to_cpp(value, cpp_type):
    if value == float("-inf"):
        return f"-std::numeric_limits<{cpp_type}>::infinity()"
    elif value == float("inf"):
        return f"std::numeric_limits<{cpp_type}>::infinity()"
    elif isinstance(value, bool):
        return f"static_cast<{cpp_type}>({str(value).lower()})"
    elif math.isnan(value):
        return f"std::numeric_limits<{cpp_type}>::quiet_NaN()"
    else:
        return f"static_cast<{cpp_type}>({repr(value)})"

# Context manager class to handle local buffers in code generation
class LocalBufferScope:
    """
    This class creates a context that helps to generate code involving Inductor IR with
    function local buffers. These buffers are constructed during the codegen process and
    are used to store intermediate results such as local accumulators. We do not want to
    add them to `V.graph` since they are not global and we do not want to add them as
    function arguments either. So we patch the codegen processes under this scope to support
    these buffers without exposure to the outside world.
    """

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.exit_stack = contextlib.ExitStack()
        self.local_buffers: Dict[str, ir.Buffer] = {}

    def __enter__(self):
        # Enter the context and patch the `get_dtype` function of `V.graph`
        self.exit_stack.__enter__()
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name):
            if name in self.local_buffers:
                return self.local_buffers[name].get_dtype()
            return original_get_dtype(name)

        self.exit_stack.enter_context(patch.object(V.graph, "get_dtype", get_dtype))

        # Patch the `input` function of `self.kernel.args`
        original_input = self.kernel.args.input

        def input(name):
            if name in self.local_buffers:
                return name
            return original_input(name)

        self.exit_stack.enter_context(patch.object(self.kernel.args, "input", input))

        # Patch the `output` function of `self.kernel.args`
        original_output = self.kernel.args.output

        def output(name):
            if name in self.local_buffers:
                return name
            return original_output(name)

        self.exit_stack.enter_context(patch.object(self.kernel.args, "output", output))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear local buffers and exit the context
        self.local_buffers.clear()
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def add_local_buffer(self, buffer: ir.Buffer):
        # Add a local buffer to the scope, ensuring it does not already exist
        assert buffer.get_name() not in self.local_buffers
        self.local_buffers[buffer.get_name()] = buffer

    def localize_buffer(
        self, global_buf: ir.Buffer, local_buf: ir.Buffer, nodes: List[ir.IRNode]
    ):
        # Method to localize a global buffer into a local buffer within the scope
        ...
    ) -> List[ir.IRNode]:
        """
        Localizes the buffer `global_buf` to `local_buf` in the given `nodes` and returns
        a new list of IR nodes that work on `local_buf` instead of `global_buf`, i.e., all
        the loads and stores are redirected to `local_buf`. This helps the fused loops to
        work on smaller-sized local buffers for better data locality.

        The `local_buf` should already be registered in the local scope and the data access
        is assumed to be contiguous with the same order as the `global_buf`.
        """
        assert local_buf.get_name() in self.local_buffers  # 确保 `local_buf` 在本地缓冲区中注册
        assert len(global_buf.get_size()) == len(local_buf.get_size())  # 确保 `global_buf` 和 `local_buf` 的尺寸维度相同
        assert len(nodes) > 0  # 确保节点列表不为空

        class LocalizeBufferHandler(V.WrapperHandler):  # type: ignore[name-defined]
            def __init__(self, inner):
                super().__init__(inner)

            def localize(self, name: str, index: sympy.Expr):
                if name == global_buf.get_name():
                    name = local_buf.get_name()
                    used_vars = {
                        s for s in index.free_symbols if symbol_is_type(s, SymT.INDEX)
                    }
                    index_vars = []
                    for i in range(len(local_buf.get_size())):
                        var = sympy_index_symbol_with_prefix(SymT.INDEX, i)
                        index_vars.append(var if var in used_vars else 0)
                    index = local_buf.layout.make_indexer()(index_vars)
                return name, index

            def load(self, name: str, index: sympy.Expr):
                return self._inner.load(*self.localize(name, index))  # 调用父类方法加载本地化后的数据

            def store(self, name, index, value, mode=None):
                return self._inner.store(*self.localize(name, index), value, mode)  # 调用父类方法存储本地化后的数据

            def store_reduction(self, name, index, value):
                return self._inner.store_reduction(*self.localize(name, index), value)  # 调用父类方法进行本地化存储的归约操作

        def wrap_inner_fn_for_node(node: ir.IRNode, inner_fn_wrapper):
            loops = node.data if isinstance(node, ir.ComputedBuffer) else node
            assert isinstance(loops, ir.Loops)
            new_loops = copy.copy(loops)
            if isinstance(node, ir.ComputedBuffer):
                new_node = ir.ComputedBuffer(
                    node.get_name(), node.get_layout(), new_loops
                )
            else:
                new_node = new_loops  # type: ignore[assignment]

            new_loops.inner_fn = inner_fn_wrapper(new_loops.inner_fn)
            return new_node

        def inner_fn_wrapper(inner_fn):
            def inner(index):
                with V.set_ops_handler(LocalizeBufferHandler(V.get_ops_handler())):
                    return inner_fn(index)

            return inner

        return [wrap_inner_fn_for_node(node, inner_fn_wrapper) for node in nodes]
def unify_mask_base_type(
    buffer: IndentedBuffer,
    vars: Tuple[CSEVariable, ...],
    dtype=torch.float,
):
    """
    统一掩码基础类型
    
    给定一个缓冲区和一组CSE变量，
    将每个变量转换为新的掩码基础数据类型并返回转换后的CSE变量。
    """
    # 使用生成器表达式，对每个变量进行处理
    new_vars = (
        V.kernel.cse.generate(
            buffer,
            f"{V.kernel._get_mask_cast(var, dtype)}",
        )
        for var in vars
    )
    # 返回处理后的变量生成器
    return new_vars
```