# `.\pytorch\torch\_inductor\codegen\cuda\cutlass_epilogue_gen.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型注解
from typing import Dict, List
# 使用 unittest.mock 模块中的 patch 函数
from unittest.mock import patch

# 引入 sympy 符号计算库
import sympy

# 导入 torch 库中的特定模块和类
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str

# 用作特殊字符串的魔术字符串，用于指示不支持的 sympy 表达式
_MAGIC_SYMPY_ERROR_STRING = "[!sympy: unsupported expr!]"


def _arg_str(a):
    # 如果参数 a 是 sympy.Expr 类型
    if isinstance(a, sympy.Expr):
        # 返回包含 _MAGIC_SYMPY_ERROR_STRING 的字符串，以及 sympy 表达式的字符串表示
        # 如果这个返回值被用作最终生成的 C++ 代码的一部分，
        # 会引发 CUTLASSEVTOpNotImplementedError，表示无法将操作转换为有效的 EVT 表达式。
        return f"{_MAGIC_SYMPY_ERROR_STRING}('{sympy_str(a)}')"
    # 如果不是 sympy.Expr 类型，直接返回其字符串表示
    return str(a)


class CUTLASSEVTOpNotImplementedError(NotImplementedError):
    # 表示 CUTLASS EVT 操作未实现的异常类
    pass


class CutlassEVTEpilogueTypeFormatter:
    """
    代码生成类，提供生成 Cutlass "Epilogue Visitor Tree" (EVT) 函数声明的入口点。

    详见 https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    关于 EVT 及其如何声明和用于生成的更多信息。

    注意：
        * 被 CUTLASSGemmTemplate 使用。
        * 用户不应该实例化这个类，而是通过调用 CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(...)
          来实例化这个类作为 virtualized.V.ops.[op-name] 的操作处理器。
        * 可以通过扩展 _op_<whatever> 节点来添加对新 pointwise 操作的支持。

    """

    def __init__(self, accumulator_node_name, evt_type_name):
        """
        初始化 CutlassEVTEpilogueTypeFormatter 的实例。

        参数：
        - accumulator_node_name (str): GEMM 操作在原始（未融合）IR图中的输出缓冲区的名称。
        - evt_type_name (str): EVT 类型的输出名称。

        """
        self.accumulator_node_name = accumulator_node_name
        self.output = IndentedBuffer(0)  # 初始化缓冲区对象，初始缩进为0
        self.var_counter = 0  # 变量计数器，用于生成唯一的变量名
        self.evt_type_name = evt_type_name
        self.aliases = dict()  # 别名字典，用于存储变量名和别名的映射关系

    @staticmethod
    def ir_to_evt_string(
        template_output_node_name: str,
        evt_type_name: str,
        epilogue_nodes: List[IRNode],
        # 以下省略了函数的其余部分，不在当前代码段中
    ):
        """
        Formats IR nodes into a string representation compatible with Cutlass EVT format.

        Args:
            template_output_node_name (str): The name of the template output node.
            evt_type_name (str): The name of the EVT type.
            epilogue_nodes (List[IRNode]): A list of IR nodes representing the epilogue nodes. As of now, these must be
                ComputedBuffer nodes wrapping Pointwise nodes.

        Returns:
            A string representation of the IR nodes formatted according to the Cutlass EVT format.
        """
        # 创建 CutlassEVTEpilogueTypeFormatter 对象，用于格式化 IR 节点
        formatter = CutlassEVTEpilogueTypeFormatter(
            template_output_node_name, evt_type_name
        )

        # 设置 V.set_ops_handler 为 formatter，同时修改 FlexibleLayout 类的 allow_indexing 属性
        with virtualized.V.set_ops_handler(formatter), patch.object(
            FlexibleLayout, "allow_indexing", True
        ):
            # 遍历后处理节点列表
            for node in epilogue_nodes:
                if isinstance(node, ComputedBuffer):
                    # 获取计算缓冲区的数据节点
                    pnode = node.data
                else:
                    # 如果节点不是计算缓冲区，抛出运行时错误
                    raise RuntimeError(
                        "Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer"
                    )
                # 确保 pnode 是 Pointwise 类型
                assert isinstance(pnode, Pointwise)
                # 使用 pnode 的索引方法计算索引
                index = pnode._index(pnode.ranges)
                # 调用 pnode 的内部函数生成结果
                result = pnode.inner_fn(index)
                # 每个后处理节点产生一个单独的 "using" 语句，并可能引用先前步骤的名称
                formatter.aliases[node.name] = result
            # 获取 formatter 的字符串表示，可能是未定义类型
            res = formatter.getvalue(result)  # type: ignore[possibly-undefined]
            # 如果结果包含特定字符串，则引发 CUTLASSEVTOpNotImplementedError
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError(
                    "sympy / indexing expressions not yet supported in EVT fusion"
                )
            else:
                # 否则返回格式化后的结果字符串
                return res

    def __getattr__(self, name):
        """
        Resolve V.ops.<whatever> calls, after this instance has been installed as V.ops handler.
        """

        def inner(*args, **kwargs):
            # 将参数转换为字符串
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            # 获取名为 _op_<name> 的方法对象
            fn = getattr(self, f"_op_{name}")
            # 调用方法对象并得到结果
            line = fn(*fargs, **fkwargs)
            # 自增变量计数器
            self.var_counter += 1
            # 生成新的变量名并替换 line
            varname = f"EVT_expr_{self.var_counter}"
            # 将新的变量声明写入输出
            self.output.writeline(f"using {varname} = {line};")
            return varname

        # 如果方法名以 '_' 开头，则抛出未实现异常
        if name.startswith("_"):
            raise CUTLASSEVTOpNotImplementedError(name)
        # 如果实例具有名为 _op_<name> 的属性，则返回 inner 函数
        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            # 否则抛出未实现异常
            raise CUTLASSEVTOpNotImplementedError(name)
    def _op_load(self, name, index_expr):
        # Load an input to an operation. Might be the output of the matmul, the result
        # of a previous epilogue node, a constant or (TODO) an auxiliary input.

        # 如果名称等于累加器节点名称，则返回与该名称相关的信息
        if name == self.accumulator_node_name:
            return f"cutlass::epilogue::fusion::Sm90AccFetch /* :={name} (matmul output in accumulator) */"
        elif name in self.aliases:
            # 如果名称在别名字典中，则返回其对应的别名
            return self.aliases[name]
        else:
            # 否则，抛出未实现的操作异常，说明操作数未找到
            raise CUTLASSEVTOpNotImplementedError(
                f"Operand {name} not found. Auxiliary inputs not supported yet."
            )

    def _op_constant(self, value, dtype):
        # Load a constant

        # 如果 dtype 是 torch.float16 或 torch.float32，则返回相应的常量信息
        if str(dtype) in ("torch.float16", "torch.float32"):
            return f"cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAcc> /* value={value}, dtype={dtype} */"
        else:
            # 否则，抛出未实现的操作异常，说明不支持该常量的数据类型
            raise CUTLASSEVTOpNotImplementedError(
                f"Unsupported dtype for constant: {dtype}"
            )

    def _cutlass_binary_functional_op(self, op, a, b):
        # Perform a named operation on two inputs
        # see https://github.com/NVIDIA/cutlass/blob/6407bcdf0a24097b7b016ee105937693c62f9923/include/cutlass/functional.h for ops

        # 返回特定操作和输入数据的 Cutlass 操作字符串
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::{op}, ElementAcc, ElementAcc, RoundStyle>,{a},{b}>"  # noqa: B950

    def _convert_to_output_dtype(self, a):
        # Convert the final output to the dtype of the output buffer

        # 将最终输出转换为输出缓冲区的数据类型的 Cutlass 操作字符串
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<identity_op, ElementD, ElementAcc, RoundStyle>,{a}>"  # noqa: B950

    def _op_to_dtype(self, a, *args, **kwargs):
        # no-op in our case, since we convert to the output dtype at the end and convert everything to the accumulator
        # dtype.
        # Is is asserted ( and ascertained during can_fuse decision ) that the dtype remains compatible
        # throughout the fusion chain.

        # 在我们的情况下，这是一个空操作，因为我们最终将输出转换为输出数据类型，并将所有内容转换为累加器数据类型。
        # 在 can_fuse 决策期间断言（并且在其间验证），数据类型在整个融合链中保持兼容。
        return a  # noqa: B950

    def _op_mul(self, a, b):
        # Perform a multiplication operation

        # 执行乘法操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_div(self, a, b):
        # Perform a division operation

        # 执行除法操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_truediv(self, a, b):
        # Perform a true division operation

        # 执行真实除法操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_ge(self, a, b):
        # Perform a greater than or equal comparison operation

        # 执行大于或等于比较操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        # Perform an addition operation

        # 执行加法操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        # Perform a subtraction operation

        # 执行减法操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("minus", a, b)

    def _op_minimum(self, a, b):
        # Perform a minimum value selection operation

        # 执行最小值选择操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("minimum", a, b)

    def _op_maximum(self, a, b):
        # Perform a maximum value selection operation

        # 执行最大值选择操作的 Cutlass 操作字符串
        return self._cutlass_binary_functional_op("maximum", a, b)
    # 定义私有方法 `_op_relu`，接受参数 `a`，返回一个字符串表达式
    def _op_relu(self, a):
        # 创建常量零值，类型为 `torch.float32`
        const_zero = self._op_constant(0.0, "torch.float32")
        # 返回一个字符串，使用 `cutlass::epilogue::fusion::Sm90EVT` 模板，进行某种计算（最大值？），参数包括 `a` 和 `const_zero`
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::maximum, ElementAcc, ElementAcc, RoundStyle>,{a}, {const_zero}>"  # noqa: B950

    # 定义 `reduction` 方法，接受参数 `dtype`, `src_dtype`, `reduction_type`, `value`，抛出 `CUTLASSEVTOpNotImplementedError` 异常
    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise CUTLASSEVTOpNotImplementedError

    # 添加更多操作...

    # 定义 `getvalue` 方法，接受参数 `result`，返回一个字符串
    def getvalue(self, result) -> str:
        # 根据内部变量 `var_counter` 创建表达式名称
        dtype_converted_expr = self._convert_to_output_dtype(
            f"EVT_expr_{self.var_counter}"
        )
        # 将使用 `self.evt_type_name` 和 `dtype_converted_expr` 创建的语句写入输出
        self.output.writeline(f"using {self.evt_type_name} = {dtype_converted_expr};")
        # 返回输出内容
        return self.output.getvalue()
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) Argument initializers

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str):
        """
        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string static method.

        Args:
            accumulator_node_name (str): The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
        """
        self.accumulator_node_name: str = accumulator_node_name  # 累加器节点的名称，用于存储矩阵乘法结果，在 IR 图中融合之前
        self.output: IndentedBuffer = IndentedBuffer(0)  # 用于代码生成的输出缓冲区
        self.var_counter: int = (
            0  # 用于生成变量名的计数器，每创建一个新变量会递增
        )
        self.aliases: Dict[str, str] = dict()  # 子表达式函数符的别名字典

    @staticmethod
    def ir_to_evt_argument_string(
        template_output_node_name: str,
        epilogue_nodes: List[IRNode],
    ) -> str:
        """
        Converts IR nodes to EVT argument string format using a formatter instance.

        Args:
            template_output_node_name (str): Name of the template output node.
            epilogue_nodes (List[IRNode]): List of IR nodes representing epilogue operations.

        Returns:
            str: EVT argument string generated from the given IR nodes.

        Raises:
            CUTLASSEVTOpNotImplementedError: If sympy / indexing expressions are encountered,
                                              which are not yet supported in EVT fusion.
        """
        formatter = CutlassEVTEpilogueArgumentFormatter(
            template_output_node_name,
        )

        with virtualized.V.set_ops_handler(formatter), patch.object(
            FlexibleLayout, "allow_indexing", True
        ):
            for node in epilogue_nodes:
                assert isinstance(node, ComputedBuffer)
                pnode = node.data
                assert isinstance(pnode, Pointwise)
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                # 每个后处理节点会生成一个 "using" 语句，并可能引用之前步骤中的名称
                if node.name is not None:
                    formatter.aliases[node.name] = result

            res: str = formatter.getvalue(result)  # 获取最终生成的代码字符串
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError(
                    "sympy / indexing expressions not yet supported in EVT fusion"
                )
            else:
                return res
    def __getattr__(self, name):
        # 定义内部函数 inner，用于动态调用操作函数并返回结果
        def inner(*args, **kwargs):
            # 将所有位置参数转换成字符串列表
            fargs = [_arg_str(a) for a in args]
            # 将所有关键字参数转换成字符串字典
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            # 获取名为 "_op_{name}" 的属性，即操作函数
            fn = getattr(self, f"_op_{name}")
            # 调用操作函数 fn，并传入转换后的参数和关键字参数，获取结果
            line = fn(*fargs, **fkwargs)
            # 返回操作函数的执行结果
            return line

        # 如果请求的属性名以 "_" 开头，抛出未实现错误
        if name.startswith("_"):
            raise CUTLASSEVTOpNotImplementedError(name)

        # 如果存在名为 "_op_{name}" 的属性（操作函数），返回内部函数 inner
        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            # 否则抛出未实现错误
            raise CUTLASSEVTOpNotImplementedError(name)

    def _op_load(self, name, index_expr):
        # 如果请求加载的名字是累加器节点名字，返回空对象字符串
        if name == self.accumulator_node_name:
            return "{}"
        # 如果名字在别名字典中，返回对应的别名字符串
        elif name in self.aliases:
            return self.aliases[name]
        else:
            # 否则抛出未实现错误，提示该操作不支持辅助输入
            raise CUTLASSEVTOpNotImplementedError(
                f"Operand {name} not found. Auxiliary inputs not supported yet."
            )

    def _op_constant(self, value, dtype):
        # 如果数据类型是 torch.float16 或 torch.float32，返回静态转换为 ElementAcc 类型的常量字符串
        if str(dtype) in ("torch.float16", "torch.float32"):
            return "{ static_cast<ElementAcc>(" + str(value) + ") }"
        else:
            # 否则抛出未实现错误，提示不支持的数据类型
            raise CUTLASSEVTOpNotImplementedError(
                f"Unsupported dtype for constant: {dtype}"
            )

    def _cutlass_binary_functional_op(self, op, a, b):
        # 返回格式化后的二元函数操作字符串，包含操作符和操作数 a、b
        return f"{{ /*{op}: */ {a}, {b} }}"

    def _op_mul(self, a, b):
        # 返回乘法操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_div(self, a, b):
        # 返回除法操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_truediv(self, a, b):
        # 返回真除法操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_ge(self, a, b):
        # 返回大于等于操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        # 返回加法操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        # 返回减法操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("minus", a, b)

    def _op_minimum(self, a, b):
        # 返回最小值操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("minimum", a, b)

    def _op_maximum(self, a, b):
        # 返回最大值操作的二元函数操作字符串
        return self._cutlass_binary_functional_op("maximum", a, b)

    def _op_relu(self, a):
        # 获取零值常量字符串
        const_zero = self._op_constant(0.0, "torch.float32")
        # 返回 ReLU 操作的二元函数操作字符串
        return "{" + str(a) + ", " + const_zero + "}"

    def _op_to_dtype(self, a, dtype, src_dtype=None):
        # 断言目标数据类型和源数据类型在支持的范围内
        assert dtype in (
            "torch.float32",
            "torch.float16",
        ), f"Unsupported dtype: {dtype}"
        assert src_dtype in (
            None,
            "torch.float32",
            "torch.float16",
        ), f"Unsupported source dtype: {src_dtype}"
        # 返回输入参数 a，不做任何操作
        return a

    def reduction(self, dtype, src_dtype, reduction_type, value):
        # 抛出未实现错误，表示该方法尚未实现
        raise CUTLASSEVTOpNotImplementedError

    def getvalue(self, result) -> str:
        # 返回格式化后的结果字符串
        return "{" + str(result) + "}"
```