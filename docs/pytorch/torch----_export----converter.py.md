# `.\pytorch\torch\_export\converter.py`

```py
# mypy: allow-untyped-defs
# 引入日志、运算符和警告模块
import logging
import operator
import warnings

# 引入类型提示模块
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 引入PyTorch相关模块
import torch
import torch.export._trace

# 引入导出的程序和图签名相关模块
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
# 引入子图重写器
from torch.fx import subgraph_rewriter

# 引入ONNX相关工具
from torch.onnx.utils import _create_jit_graph

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def inplace_optimize_sym_size_div(gm: torch.fx.GraphModule):
    # 定义模式匹配函数，用于在计算图中查找特定模式
    def pattern(im, dim, scale):
        # 调用ATen函数获取符号尺寸的整数表示
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        # 创建标量张量表示的值
        scalar_tensor = torch.ops.aten.scalar_tensor(sym_size_int)
        # 使用ATen函数进行除法操作，指定截断模式
        div_scalar_mode = torch.ops.aten.div.Scalar_mode(
            scalar_tensor, scale, rounding_mode="trunc"
        )
        # 将结果转换为整数张量
        int_tensor = torch.ops.aten.Int.Tensor(div_scalar_mode)
        return int_tensor

    # 定义替换函数，用于替换匹配到的模式
    def replacement(im, dim, scale):
        # 直接计算符号尺寸整数除以指定的标量值
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        return sym_size_int // scale

    # 使用子图重写器替换计算图中匹配到的模式
    replaced_patterns = subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def normalize_name(name: str) -> str:
    # 将名称中的点号替换为下划线
    return name.replace(".", "_")


def ir_name_to_func_name(name: str) -> str:
    """prim::If -> convert_prim_If"""
    # 将名称按双冒号分割，加上前缀"convert_"并用下划线连接
    name_list = name.split("::")
    return "convert_" + "_".join(name_list)


def get_node_for_param_and_buffer(fx_graph, name, is_top_level_graph):
    # 根据是否为顶层图决定获取节点的方式
    if is_top_level_graph:
        return fx_graph.get_attr(name)
    return fx_graph.placeholder(name)


_TORCH_DTYPE_TO_ENUM = {
    # 定义PyTorch数据类型到枚举值的映射字典
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def get_dtype_as_int(tensor):
    """
    prim::dtype has the signature "Tensor a) -> int", where it gets the dtype of
    the tensor and returns the integer corresponding to this dtype based on the
    enum in ScalarType.h
    """
    # 获取张量的数据类型并返回其对应的枚举整数值
    dtype = tensor.dtype
    if dtype not in _TORCH_DTYPE_TO_ENUM:
        raise RuntimeError(f"Unsupported dtype {dtype}")
    return _TORCH_DTYPE_TO_ENUM[dtype]


# 这些操作将自动添加到TS2FXGraphConverter实例方法中，方法名格式为convert_<namespace>_<opname>()
# 请查看__init__方法以了解方法添加的具体实现。
kind_to_standard_operators = {
    # 将不同操作的名称映射到对应的Python标准运算符或自定义函数
    "prim::TupleIndex": operator.getitem,
    "aten::__is__": operator.is_,
    "aten::__isnot__": operator.is_not,
    "aten::__not__": operator.not_,
    "aten::__contains__": operator.contains,
    "prim::dtype": get_dtype_as_int,
    "aten::len": len,
}


def get_ir_value_parent_name_and_attr_name(node):
    # 获取节点关联的输入值名称、父名称和属性名称
    irv_parent_name, irv_name = node.input().debugName(), node.output().debugName()
    attr_name = node.s("name")
    return irv_name, irv_parent_name, attr_name
    # 定义一个函数，根据输入的 IR（指令流）、引用映射和名称映射，构造完全限定名称（FQN）
    def construct_fqn(ir, ref_map, name_map):
        # 用于存储构造的名称列表
        name_list = []
        # 循环直到 IR 存在于引用映射中
        while ir in ref_map:
            # 将当前 IR 对应的名称添加到名称列表中
            name_list.append(name_map[ir])
            # 更新当前 IR 为其父级 IR
            ir = ref_map[ir]
        # 将名称列表逆序并用点号连接成完全限定名称字符串
        return ".".join(reversed(name_list))


    # 定义一个函数，获取将块映射到其提升属性 FQN 集合的映射
    def get_block_to_lifted_attrs(graph: torch._C.Graph) -> Dict[torch._C.Block, Set[str]]:
        """
        执行两次遍历，获取将块映射到其提升属性 FQN 集合的映射。
        当图具有控制流时，图将被分为多个块。我们希望将每个块转换为将传递给 torch.cond 的图。
        torch.cond 有一个限制，即模型参数/缓冲区应作为子图的输入提升。在转换模型之前，
        我们将运行此步骤来：
            1. 通过追踪 GetAttr 调用来确定在块内使用哪些参数/缓冲区。
            2. 自底向上处理图，通过取当前块中使用的属性与其所有子块的提升属性的并集来找出每个块的提升属性。

        返回：
        将块映射到其提升属性 FQN 集合的映射。
        """

        # 一个将块映射到预期提升参数的字典映射
        blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]] = dict()

        # 引用映射存储 GetAttr 节点的输入（即源）和输出（即目标）IR。
        # 通过遍历此引用映射，我们可以找出完整的 IR 别名传递，并找出属性的 FQN。
        node_to_parent_map: Dict[str, str] = dict()

        # 用于基于引用映射重建属性的 FQN。
        # 简言之，对于每个 GetAttr 调用，GetAttr（输入 IR，属性名）-> 输出 IR。
        # 此名称映射存储了针对某个源 IR --> 目标 IR 操作调用了哪个属性名。
        node_to_attr_name: Dict[str, str] = dict()

        def _dfs_get_attr_dependency(entry):
            """
            第一个深度优先搜索路径，用于构造引用映射和名称映射。
            """
            for node in entry.nodes():
                if node.kind() == "prim::GetAttr":
                    (
                        irv_name,
                        irv_parent_name,
                        attr_name,
                    ) = get_ir_value_parent_name_and_attr_name(node)
                    node_to_parent_map[irv_name] = irv_parent_name
                    node_to_attr_name[irv_name] = attr_name
                for block in node.blocks():
                    _dfs_get_attr_dependency(block)
    def _map_blocks_to_lifted_attrs(entry):
        """
        Walk the graph in a bottom-up fashion to build the expected to be
        lifted arguments for each block.
        """
        arguments: Set[str] = set()  # 创建一个空集合用于存储参数名
        for node in entry.nodes():  # 遍历入口节点的所有节点
            for block in node.blocks():  # 遍历当前节点的所有块
                # 递归构建参数集合
                arguments = arguments.union(_map_blocks_to_lifted_attrs(block))
            if node.kind() == "prim::GetAttr":  # 如果节点类型是 prim::GetAttr
                irv_name = node.output().debugName()  # 获取节点输出的调试名称
                # 跳过中间 GetAttr，因为它们不会生成完整限定名称（FQN）
                # 例如，node_to_parent_name: {"%3": "%2", "%2": "%1"}
                #       node_to_attr_name: {"%3": "weight", "%2": "linear", "%1": "self"}
                #       只有一个完整的 FQN %3-->%2-->%1: self.linear.weight
                #       %2-->%1 不是完整的 FQN: self.linear
                if irv_name not in set(node_to_parent_map.values()):  # 如果调试名称不在父节点映射的值集合中
                    arguments.add(
                        construct_fqn(irv_name, node_to_parent_map, node_to_attr_name)
                    )  # 构建完整限定名称并添加到参数集合中
        if not isinstance(entry, torch._C.Graph):  # 如果 entry 不是 torch._C.Graph 类型（跳过顶层）
            blocks_to_lifted_attrs[entry] = arguments  # 将参数集合关联到对应的 entry
        return arguments  # 返回参数集合

    _dfs_get_attr_dependency(graph)  # 调用另一个函数处理图的属性依赖
    _map_blocks_to_lifted_attrs(graph)  # 调用当前函数处理图的块属性依赖

    return blocks_to_lifted_attrs  # 返回处理后的块到提升属性的映射
    def get_attribute_fqn_from_ts_node(
        name_to_attribute_fqn: Dict[str, str], node: torch._C.Node
    ) -> str:
        # 定义内部函数，用于根据名称获取完全限定属性名
        def get_attr(name: str):
            # 如果名称在映射中，则返回对应的完全限定属性名
            if name in name_to_attribute_fqn:
                return name_to_attribute_fqn[name]
            else:
                # 如果名称未在映射中，则抛出值错误异常
                raise ValueError(f"Attribute {name} not found")

        # 根据节点的类型进行分支处理，获取节点的输入名称
        if node.kind() == "prim::SetAttr":
            input_name = next(node.inputs()).debugName()
        elif node.kind() == "prim::GetAttr":
            input_name = node.input().debugName()
        else:
            # 如果节点类型不是预期的设置或获取属性类型，则引发运行时错误
            raise RuntimeError(
                f"Unexpected node kind when getting attribute fqn. node: {node} "
            )

        # 从节点中获取属性名称
        attr_name = node.s("name")
        # 使用内部函数获取输入名称对应的完全限定根属性名称
        root_attr_name = get_attr(input_name)
        # 组合属性的完全限定名称
        attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name

        return attr_fqn


    def get_op_overload(node: torch._C.Node):
        # 从节点中获取字符串格式的函数模式
        schema_str = node.schema()
        # 将字符串模式解析为函数模式对象
        schema: torch._C.FunctionSchema = torch._C.parse_schema(schema_str)
        # 将函数命名空间和操作名称分割开来
        ns, op_name = str(schema.name).split("::")
        # 获取函数模式中的重载名称
        override = schema.overload_name

        try:
            # 尝试获取指定命名空间下的操作模块
            op_overload_mod = getattr(torch.ops, ns)
            # 获取操作模块中的指定操作名称
            op_overload_packet = getattr(op_overload_mod, op_name)
            # 如果有重载名称，则获取对应的重载函数；否则使用默认函数
            if override:
                op_overload = getattr(op_overload_packet, override)
            else:
                op_overload = op_overload_packet.default
        except Exception as e:
            # 如果出现异常，抛出运行时错误，指明无法找到对应的操作符
            raise RuntimeError(
                f"Unable to find operator {node.kind()} with schema {node.schema}"
            ) from e

        return op_overload


    class TS2FXGraphConverter:
        def __init__(
            self,
            ts_graph: Union[torch._C.Graph, torch._C.Block],
            name_to_param_map: Dict[str, torch.Tensor],
            name_to_buffer_map: Dict[str, torch.Tensor],
            blocks_to_lifted_attrs: Dict[torch._C.Block, Set[str]],
            name_to_non_tensor_attribute: Dict[str, Any],
        ):
            # 初始化方法，接受转换器的各种输入映射和配置
            self.ts_graph = ts_graph
            self.name_to_param_map = name_to_param_map
            self.name_to_buffer_map = name_to_buffer_map
            self.blocks_to_lifted_attrs = blocks_to_lifted_attrs
            self.name_to_non_tensor_attribute = name_to_non_tensor_attribute
        ):
        self.ts_graph = ts_graph  # 将传入的 TorchScript 图赋值给对象的 ts_graph 属性
        self.name_to_param_map = name_to_param_map  # 将传入的名称到参数映射赋值给对象的 name_to_param_map 属性
        self.name_to_buffer_map = name_to_buffer_map  # 将传入的名称到缓冲区映射赋值给对象的 name_to_buffer_map 属性

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()  # 初始化一个空的 Torch FX 图，并赋给对象的 fx_graph 属性
        self.input_specs: List[InputSpec] = []  # 初始化一个空的输入规格列表，赋给对象的 input_specs 属性
        self.output_specs: List[OutputSpec] = []  # 初始化一个空的输出规格列表，赋给对象的 output_specs 属性

        self.name_to_node: Dict[
            str, Union[torch.fx.Node, List[torch.fx.Node], Dict[Any, torch.fx.Node]]
        ] = {}  # 初始化一个空的字典，用于存储节点名称到节点或节点列表或字典的映射，赋给对象的 name_to_node 属性
        self.constant_map: Dict[str, Any] = {}  # 初始化一个空的常量映射字典，赋给对象的 constant_map 属性

        # Mapping from torchscript node output name to attribute fully qualified name
        self.name_to_attribute_fqn: Dict[str, str] = {}  # 初始化一个空的字典，用于将 TorchScript 节点输出名称映射到属性完全限定名，赋给对象的 name_to_attribute_fqn 属性

        self.name_to_tensor_constants: Dict[str, torch.Tensor] = {}  # 初始化一个空的字典，用于存储名称到 Torch 张量常量的映射，赋给对象的 name_to_tensor_constants 属性

        # Mapping from fully qualified name to real values or a fx graph node
        # During convert, this represents the current value of a non-tensor attribute
        # One use case is:
        #   def forward(self, x):
        #        c1 = self.count
        #        self.count += 1
        #        c2 = self.count
        #        return x + c1 + c2
        self.name_to_non_tensor_attribute_node: Dict[str, Any] = {}  # 初始化一个空的字典，用于将完全限定名称映射到真实值或 FX 图节点，赋给对象的 name_to_non_tensor_attribute_node 属性

        # Mapping from fully qualified name to initial real values inputs
        # We separate it from self.name_to_non_tensor_attribute_node since
        # we need initial real value input when we construct fx.GraphModule
        self.name_to_non_tensor_attribute: Dict[str, Any] = name_to_non_tensor_attribute  # 将传入的名称到非张量属性的初始真实值映射赋值给对象的 name_to_non_tensor_attribute 属性

        self.subgraphs: Dict[str, torch.fx.GraphModule] = {}  # 初始化一个空的子图字典，赋给对象的 subgraphs 属性

        self.blocks_to_lifted_attrs = blocks_to_lifted_attrs  # 将传入的块到提升属性映射赋值给对象的 blocks_to_lifted_attrs 属性

        # Populate methods for the standard operators.
        for k in kind_to_standard_operators.keys():
            handler_func_name = ir_name_to_func_name(k)
            # Create an indirect function call:
            # convert_<namespace>_<opname> --> lambda node: _convert_standard_operator(node)
            setattr(
                self,
                handler_func_name,
                lambda node: self._convert_standard_operators(node),
            )  # 对于标准运算符，为每个运算符创建方法，并将方法绑定到对象上，方法的实现为调用 self._convert_standard_operators(node)

    def is_top_level_graph(self):
        return isinstance(self.ts_graph, torch._C.Graph)  # 检查对象的 ts_graph 属性是否是 torch._C.Graph 类型，判断当前图是否为顶层图

    def add_subgraph(self, subgraph) -> str:
        name = f"subgraph_{len(self.subgraphs)}"  # 根据当前子图数量生成一个唯一的名称
        self.subgraphs[name] = subgraph  # 将传入的子图对象存储在对象的 subgraphs 属性中
        return name  # 返回新添加的子图的名称作为字符串

    def get_args_kwargs(self, node: torch._C.Node, schema):
        args = []
        kwargs = {}
        for input, schema_arg in zip(node.inputs(), schema.arguments):
            if schema_arg.kwarg_only:
                kwargs[schema_arg.name] = self.get_fx_value(input)
            else:
                args.append(self.get_fx_value(input))
        return tuple(args), kwargs  # 根据节点的输入和模式返回参数元组和关键字参数字典
    # 获取给定的 torch._C.Value 对象的调试名称
    def get_fx_value(self, value: torch._C.Value):
        value_name = value.debugName()
        
        # 如果名称存在于 self.name_to_node 中，则返回对应的节点
        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        
        # 如果名称存在于 self.constant_map 中，则返回对应的常量值
        elif value_name in self.constant_map:
            return self.constant_map[value_name]
        
        # 如果都不存在，则抛出 ValueError 异常
        else:
            raise ValueError(f"Input {value_name} not found")

    # 将转换后的图形表示为 torch.fx.GraphModule 对象
    def convert(self) -> torch.fx.GraphModule:
        # 转换图形的输入参数
        self.convert_graph_inputs()

        # 遍历所有图中的节点，并对每个节点进行转换
        for node in self.ts_graph.nodes():
            self.convert_node(node)

        # 转换图形的输出
        self.convert_graph_outputs()

        # 创建 torch.fx.GraphModule 对象，并传入相关映射和图形表示
        gm = torch.fx.GraphModule(
            {
                **self.subgraphs,
                **self.name_to_param_map,
                **self.name_to_buffer_map,
                **self.name_to_tensor_constants,
                **self.name_to_non_tensor_attribute,
            },
            self.fx_graph,
        )

        # 对 gm 进行原地优化以减少符号大小除法操作
        inplace_optimize_sym_size_div(gm)

        # 对 gm 的图形进行 lint 检查
        gm.graph.lint()

        # 返回转换后的 GraphModule 对象
        return gm

    # 转换图形的输入参数
    def convert_graph_inputs(self):
        # 遍历原始图形的输入参数
        for graph_input in self.ts_graph.inputs():
            # 获取输入参数的调试名称
            name = graph_input.debugName()
            
            # 标准化输入参数的名称
            normalized_name = normalize_name(name)

            # 如果输入参数名称存在于 self.name_to_param_map 中，则将其标记为参数类型
            if name in self.name_to_param_map:
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                # 获取与参数和缓冲区相关的节点
                fx_node = get_node_for_param_and_buffer(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            
            # 如果输入参数名称存在于 self.name_to_buffer_map 中，则将其标记为缓冲区类型
            elif name in self.name_to_buffer_map:
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
                # 获取与参数和缓冲区相关的节点
                fx_node = get_node_for_param_and_buffer(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            
            # 如果都不存在，则将其标记为用户输入类型
            else:
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                # 创建占位符节点，并将其作为输入节点
                fx_node = self.fx_graph.placeholder(normalized_name)

            # 将输入参数名称与其节点映射关系保存到 self.name_to_node 中
            self.name_to_node[name] = fx_node
    def convert_aten_tensor(self, node: torch._C.Node):
        """Convert aten::tensor operation to a constant tensor and handle attributes."""
        # 获取操作的参数和关键字参数，使用默认的 schema: torch.ops.aten.tensor.default._schema
        args, kwargs = self.get_args_kwargs(node, torch.ops.aten.tensor.default._schema)
        
        # 处理关键字参数中的 requires_grad 属性，将其转换为布尔值
        for k in kwargs:
            if k == "requires_grad":
                kwargs[k] = bool(kwargs[k])  # 0 -> False, 1 -> True
        
        # 创建一个 PyTorch 的 tensor 对象
        tensor = torch.tensor(*args, **kwargs)

        # 获取输出节点的调试名称
        output_name = node.output().debugName()
        
        # 构造一个别名，用于在 FX 图中标识这个常量张量
        alias_name = f"lifted_tensor_{output_name}"
        
        # 从 FX 图中获取与别名对应的节点，并将创建的 tensor 存储到常量映射中
        fx_node = self.fx_graph.get_attr(alias_name)
        self.name_to_node[output_name] = fx_node
        self.name_to_tensor_constants[alias_name] = tensor

    def convert_prim_Constant(self, node: torch._C.Node):
        """Convert primitive constant to appropriate Python type and store in constant map."""
        # 获取节点的输出名称
        name = node.output().debugName()

        # 初始化值为 None
        value: Any = None
        
        # 检查节点是否有 value 属性
        if node.hasAttribute("value"):
            # 获取常量的类型
            constant_kind = node.kindOf("value")
            
            # 根据类型不同获取相应的值
            if constant_kind == "i":
                value = node.i("value")  # 整数类型
            elif constant_kind == "f":
                value = node.f("value")  # 浮点数类型
            elif constant_kind == "s":
                value = node.s("value")  # 字符串类型
            elif constant_kind == "t":
                alias_name = f"lifted_tensor_{name}"  # 遵循 EP 跟踪的命名惯例
                fx_node = self.fx_graph.get_attr(alias_name)
                self.name_to_tensor_constants[alias_name] = node.t("value")  # 存储常量张量
                value = fx_node  # 将 FX 节点赋给值
            elif constant_kind == "ival":
                value = node.ival("value")  # ival 类型
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")  # 抛出错误，不支持的常量类型
        else:
            value = None

        # 将常量值存储到常量映射中
        self.constant_map[name] = value

    def convert_prim_device(self, node: torch._C.Node):
        """Convert device primitive operation to device type and store in constant map."""
        # 获取输入节点的类型
        input_type = node.input().type()
        
        # 检查输入类型是否是 TensorType
        if input_type.isSubtypeOf(torch._C.TensorType.get()):
            # 获取设备信息
            device = input_type.device()  # type: ignore[attr-defined]
            
            # 获取输出节点的调试名称，并将设备信息存储到常量映射中
            output_name = node.output().debugName()
            self.constant_map[output_name] = device
        else:
            # 如果不是 TensorType，抛出错误
            raise ValueError(f"Unsupported JitType ({input_type}) when get device")
    def convert_prim_GetAttr(self, node: torch._C.Node):
        # 构建完全限定名
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        # 获取节点输出的调试名称
        output_name = node.output().debugName()
        # 将完全限定名映射到输出名称
        self.name_to_attribute_fqn[output_name] = attr_fqn

        # 获取属性值节点
        attr_value = node.output()
        if self.is_top_level_graph():
            if attr_value.type().annotation_str == "Tensor":
                # 如果是顶层图并且属性值是张量，则插入一个 get_attr 节点
                # 原因是：
                # 1. ts 图不会将张量常量作为输入节点提升。因此张量常量可能会在 convert_graph_inputs() 中被忽略。
                # 2. attr_fqn 可能已经通过 SetAttr 写入。两个 GetAttr 可能会给出不同的值。
                self.name_to_node[output_name] = self.fx_graph.get_attr(attr_fqn)
            else:
                # 如果属性值不是张量，则根据 attr_fqn 获取非张量属性节点
                if attr_fqn not in self.name_to_non_tensor_attribute_node:
                    self.name_to_non_tensor_attribute_node[
                        attr_fqn
                    ] = self.name_to_non_tensor_attribute[attr_fqn]
                self.name_to_node[output_name] = self.name_to_non_tensor_attribute_node[
                    attr_fqn
                ]
        else:
            # 对于不允许 SetAttr TorchScript 节点和 get_attr FX 图节点的 if 块，提供特殊支持
            if attr_value.type().annotation_str == "Tensor":
                self.name_to_node[output_name] = self.name_to_node[attr_fqn]

    def convert_prim_SetAttr(self, node: torch._C.Node):
        # 获取属性完全限定名
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        # 获取属性值
        attr_value = tuple(node.inputs())[1]
        # 获取 FX 图中的属性值输入
        ts_graph_tensor_input = self.get_fx_value(attr_value)
        if attr_value.type().annotation_str == "Tensor":
            # 如果属性值是张量，则获取 FX 图中的属性节点，并调用 torch.Tensor.copy_ 函数复制张量数据
            fx_attr_node = self.fx_graph.get_attr(attr_fqn)
            self.fx_graph.call_function(
                torch.Tensor.copy_, (fx_attr_node, ts_graph_tensor_input)
            )
        else:
            # 如果属性值不是张量，则将其存储为非张量属性节点
            self.name_to_non_tensor_attribute_node[attr_fqn] = ts_graph_tensor_input

    def convert_call_function_op(self, node: torch._C.Node):
        # 获取操作的目标函数
        target = get_op_overload(node)

        if target is torch.ops.aten.size.int:
            # 特殊处理 torch.ops.aten.size.int 的情况
            target = torch.ops.aten.sym_size.int

        # 获取调用函数的参数和关键字参数
        args, kwargs = self.get_args_kwargs(node, target._schema)

        # 在 FX 图中调用目标函数，并获取返回的 FX 节点
        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: 将 sourceRange() 转换为 stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        # 将 FX 节点映射到输出名称
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_TupleConstruct(self, node: torch._C.Node):
        # 转换元组构造节点，调用内部方法处理
        self._convert_prim_iterator(node)

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        # 转换列表构造节点，调用内部方法处理
        self._convert_prim_iterator(node)
    def _convert_prim_iterator(self, node: torch._C.Node):
        # 初始化空列表用于存储输出值
        output_list = []
        # 遍历节点的输入
        for inp in node.inputs():
            # 调用方法获取输入值的效果图表示，并添加到输出列表中
            output_list.append(self.get_fx_value(inp))

        # 获取节点输出的调试名称作为输出名
        output_name = node.output().debugName()
        # 将输出列表存储到名称到节点映射中
        self.name_to_node[output_name] = output_list

    def convert_prim_DictConstruct(self, node: torch._C.Node):
        # 初始化空字典用于存储输出字典
        output_dict = {}
        k, v = None, None
        # 遍历节点的输入
        for i, inp in enumerate(node.inputs()):
            # 假设键值对存储在DictConstruct中
            # 第一个元素是键，后续元素是值
            if i % 2 == 0:
                k = self.get_fx_value(inp)
            else:
                v = self.get_fx_value(inp)
                # 断言确保键和值都不为空
                assert (
                    k is not None and v is not None
                ), "DictConstruct包含空键值对."
                # 将键值对添加到输出字典中
                output_dict[k] = v
                k, v = None, None

        # 断言确保偶数个元素，即键值对数目匹配
        assert (
            k is None and v is None
        ), "DictConstruct包含奇数个元素（违反我们的假设）."

        # 获取节点输出的调试名称作为输出名
        output_name = node.output().debugName()
        # 将输出字典存储到名称到节点映射中
        self.name_to_node[output_name] = output_dict

    def convert_prim_ListUnpack(self, node: torch._C.Node):
        # 调用内部方法处理解包迭代器
        self._convert_prim_unpack_iterator(node)

    def convert_prim_TupleUnpack(self, node: torch._C.Node):
        # 调用内部方法处理解包迭代器
        self._convert_prim_unpack_iterator(node)

    def _convert_prim_unpack_iterator(self, node: torch._C.Node):
        # 单个输入和多个输出用于解包
        for i, outp in enumerate(node.outputs()):
            # 获取输出节点的调试名称
            outp_name = outp.debugName()
            # 获取输入值的效果图表示
            inp = self.get_fx_value(node.input())
            # 调用函数操作符获取指定项并创建节点
            fx_node = self.fx_graph.call_function(operator.getitem, (inp, i))
            # 将创建的节点存储到名称到节点映射中
            self.name_to_node[outp_name] = fx_node

    def convert_aten_Int(self, node: torch._C.Node):
        # 将 aten::Int 转换为 aten._to_copy + aten::_local_scalar_dense
        # 设置目标函数为 torch.ops.aten._to_copy.default
        target = torch.ops.aten._to_copy.default
        # 获取所有输入值的效果图表示，并作为参数传递给函数调用
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        # 调用函数创建 _to_copy 节点
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        # 调用函数创建 _local_scalar_dense 节点
        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: 将 sourceRange() 转换为 stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        # 获取节点输出的调试名称作为输出名
        output_name = node.output().debugName()
        # 将创建的节点存储到名称到节点映射中
        self.name_to_node[output_name] = fx_node
    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # 将 prim::NumToTensor 转换为 aten.scalar_tensor.
        # prim::NumToTensor IRs 当前由以下触发：
        # .size() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L950
        # .numel() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L971
        # 对于这两个 API，torch.jit.trace 隐式设置输出张量类型为 LongTensor.

        # 设置目标操作为 torch.ops.aten.scalar_tensor
        target = torch.ops.aten.scalar_tensor
        # 获取节点输入的参数
        args = tuple(self.get_fx_value(input) for input in node.inputs())

        # 在 fx_graph 中调用目标函数，传入参数和 dtype 参数设为 torch.long
        fx_node = self.fx_graph.call_function(target, args, {"dtype": torch.long})
        # 获取输出节点的调试名称
        output_name = node.output().debugName()
        # 将节点名称映射到 fx_node
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        # 获取输出节点的调试名称
        output_name = node.output().debugName()
        # 将节点名称映射到空字符串
        self.name_to_attribute_fqn[output_name] = ""

    def convert_aten__convolution(self, node: torch._C.Node):
        # 将 aten::_convolution 转换为 aten.convolution，因为 aten::_convolution 没有元函数
        # 设置目标操作为 torch.ops.aten.convolution.default
        target = torch.ops.aten.convolution.default
        # 根据节点和目标操作的模式获取参数和关键字参数
        args, kwargs = self.get_args_kwargs(node, target._schema)

        # 在 fx_graph 中调用目标函数，传入参数和关键字参数
        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # 获取输出节点的调试名称
        output_name = node.output().debugName()
        # 将节点名称映射到 fx_node
        self.name_to_node[output_name] = fx_node

    def convert_aten_div(self, node: torch._C.Node):
        # 获取操作的重载函数
        target = get_op_overload(node)
        # 获取目标操作的模式
        schema = target._schema

        # 根据节点和模式获取参数和关键字参数
        args, kwargs = self.get_args_kwargs(node, schema)

        # 将 aten::div.Tensor_mode(x, tensor_constant)
        # 转换为 aten.div.Scalar_mode(x, tensor_constant.item())
        if schema.overload_name == "Tensor_mode":
            arg1_name = args[1].name
            if arg1_name in self.name_to_tensor_constants:
                tensor_constant = self.name_to_tensor_constants[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.name_to_tensor_constants[arg1_name].item()

                    # 在 fx_graph 中调用目标函数，传入更新后的参数和关键字参数
                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: 将 sourceRange() 转换为 stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    # 获取输出节点的调试名称
                    output_name = node.output().debugName()
                    # 将节点名称映射到 fx_node
                    self.name_to_node[output_name] = fx_node
                    return

        # 如果不符合上述条件，则调用默认的转换函数
        self.convert_call_function_op(node)
    # 定义方法，用于将输入节点在 ATen 下标运算（getitem）的转换过程
    def convert_aten___getitem__(self, node: torch._C.Node):
        # 解析节点的输入容器和下标，获取其转换后的值
        input_container, index = tuple(
            self.get_fx_value(input) for input in node.inputs()
        )
        # 在 FX 图中调用 ATen 的下标运算函数，并将结果保存在 FX 节点中
        fx_node = self.fx_graph.call_function(
            operator.getitem, (input_container, index)
        )
        # 获取节点的输出名称并将 FX 节点保存到名称到节点的映射中
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    # 定义方法，用于检查在 if 节点块中设置属性的情况
    def _check_set_attr_in_if_block(self, if_node: torch._C.Node):
        # 遍历 if 节点的所有块
        for block in if_node.blocks():
            # 遍历每个块中的节点
            for node in block.nodes():
                # 如果节点的类型为 "prim::SetAttr"，则抛出运行时错误
                if node.kind() == "prim::SetAttr":
                    raise RuntimeError(
                        "During converting prim::If to torch.cond, found prim::SetAttr op"
                        " which is not supported yet. Please file an issue if you come "
                        "across this error."
                    )
    # 将节点转换为条件判断节点的表示
    def convert_prim_If(self, node: torch._C.Node):
        # 在条件节点块中检查并设置属性
        self._check_set_attr_in_if_block(node)

        # 获取节点的输入，应该只有一个输入作为条件判断
        inputs = list(node.inputs())
        assert len(inputs) == 1
        predicate = self.get_fx_value(inputs[0])

        # 获取所有块中输入参数的并集
        arguments = set()
        for block in node.blocks():
            block_args = set()

            # 遍历块中的节点
            for block_node in block.nodes():
                # 遍历节点的输入
                for block_node_in in block_node.inputs():
                    # 如果输入在名字映射中存在，则将其加入块参数集合中
                    if block_node_in.debugName() in self.name_to_node:
                        block_args.add(block_node_in.debugName())

            arguments.update(block_args)

        # 将块中需要提升为输入的参数加入到总参数集合中
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)

        # 将块转换为子图
        subgraph_nodes = []
        for block in node.blocks():
            # 使用转换器将块转换为 FX 图
            subgraph_converter = TS2FXGraphConverter(
                block, dict(), dict(), self.blocks_to_lifted_attrs, dict()
            )
            subgraph_converter.constant_map = self.constant_map
            subgraph_converter.name_to_attribute_fqn = self.name_to_attribute_fqn

            # 为子图中的每个参数创建占位符节点
            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            # 执行块的转换，并将生成的子图添加到主图中
            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))

        assert len(subgraph_nodes) == 2

        # 将块参数映射为 FX 图中的节点
        fx_block_args = []
        for arg_name in arguments:
            if arg_name in self.name_to_node:
                arg_node = self.name_to_node[arg_name]
                fx_block_args.append(arg_node)
            elif arg_name in self.name_to_non_tensor_attribute_node:
                arg_node = self.name_to_non_tensor_attribute_node[arg_name]
                fx_block_args.append(arg_node)
            elif arg_name in self.name_to_non_tensor_attribute:
                arg_value = self.name_to_non_tensor_attribute[arg_name]
                fx_block_args.append(arg_value)
            else:
                raise ValueError(f"Attribute {arg_name} not found")

        # 构造 torch.cond 函数调用的参数
        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        # 在 FX 图中调用 torch.cond 函数
        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        # 将条件节点的输出名映射为生成的条件节点
        output_name = node.output().debugName()
        self.name_to_node[output_name] = cond_node

    # 将节点转换为逻辑运算节点（在此处是作为空操作）
    def convert_aten_Bool(self, node: torch._C.Node):
        self._convert_as_noop(node)
    def _convert_as_noop(self, node: torch._C.Node):
        # 将节点转换为无操作，将其输出节点映射为 arg[0]
        
        # 获取操作重载的目标函数
        target = get_op_overload(node)
        # 获取目标函数的模式
        schema = target._schema
        
        # 获取节点的参数和关键字参数
        args, kwargs = self.get_args_kwargs(node, schema)
        
        # 获取节点的输出名称
        output_name = node.output().debugName()
        # 将输出节点映射到 args[0] 中
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_enter_new(self, node: torch._C.Node):
        # 将节点转换为 profiler._record_function_enter_new 方法的调用
        
        # 获取操作的目标函数
        target = torch.ops.profiler._record_function_enter_new
        # 获取节点的所有输入并转换为对应的 fx_value
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        # 在 fx_graph 中调用目标函数并获取返回的节点
        fx_node = self.fx_graph.call_function(target, args)
        # 获取节点的输出名称
        output_name = node.output().debugName()
        # 将输出节点映射到 fx_node 中
        self.name_to_node[output_name] = fx_node

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # 将节点转换为 profiler._record_function_exit 方法的调用
        # _record_function_exit 具有副作用，因此我们在 fx.graph 中保留它
        # 目前，在 `retrace_as_exported_program` 过程中丢弃 _record_function_enter_new 和 _record_function_exit
        
        # 获取操作的目标函数
        target = torch.ops.profiler._record_function_exit
        # 获取节点的所有输入并转换为对应的 fx_value
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        # 在 fx_graph 中调用目标函数
        self.fx_graph.call_function(target, args)

    def convert_prim_tolist(self, node: torch._C.Node):
        # 无法通过 `_convert_standard_operators` 支持 prim::tolist，因为它需要调用 call_method 而不是 call_function
        
        # 设置目标为 "tolist"
        target = "tolist"
        # 获取输入节点的 fx_value
        args = (self.get_fx_value(next(node.inputs())),)
        # 在 fx_graph 中调用方法 "tolist"
        fx_node = self.fx_graph.call_method(target, args)
        # 获取节点的输出名称
        output_name = node.output().debugName()
        # 将输出节点映射到 fx_node 中
        self.name_to_node[output_name] = fx_node

    def _convert_standard_operators(self, node: torch._C.Node):
        # 将节点转换为其对应的标准运算符函数调用
        
        # 根据节点类型获取标准运算符函数
        target = kind_to_standard_operators[node.kind()]
        # 获取节点的所有输入并转换为对应的 fx_value
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        # 在 fx_graph 中调用目标函数
        fx_node = self.fx_graph.call_function(target, args)
        # 获取节点的输出名称
        output_name = node.output().debugName()
        # 将输出节点映射到 fx_node 中
        self.name_to_node[output_name] = fx_node

    def convert_node(self, node: torch._C.Node):
        # 将节点转换为相应的处理函数
        
        # 获取节点的类型
        node_kind = node.kind()
        
        # 根据命名空间和操作符名称获取处理函数
        handler_func_name = ir_name_to_func_name(node_kind)
        handler_func = getattr(self, handler_func_name, self.convert_call_function_op)
        
        # 获取节点的字符串表示，只保留第一行以避免重复逻辑
        node_str = "".join(str(node).split("\n")[:1])
        log.debug(f"[{handler_func.__name__}] converts [{node_str}]")  # noqa: G004
        
        try:
            # 调用相应的处理函数处理节点
            handler_func(node)
        except Exception as e:
            # 如果处理失败，抛出运行时错误
            raise RuntimeError(f"TS2EPConverter failed for node {node_kind}") from e
    def convert_graph_outputs(self):
        args = []  # 创建一个空列表，用于存储处理后的输出节点
        for graph_output in self.ts_graph.outputs():
            output_name = graph_output.debugName()  # 获取当前输出节点的调试名称
            if output_name in self.name_to_node:
                args.append(self.name_to_node[output_name])  # 如果名称在节点映射中存在，将节点添加到args列表中
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output_name),  # 创建一个张量参数的输出规范
                        target=output_name,  # 设置输出目标名称
                    )
                )
            elif output_name in self.constant_map:
                args.append(self.constant_map[output_name])  # 如果名称在常量映射中存在，将常量添加到args列表中
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=ConstantArgument(
                            name=output_name, value=self.constant_map[output_name]  # 创建一个常量参数的输出规范
                        ),
                        target=output_name,  # 设置输出目标名称
                    )
                )
            else:
                raise ValueError(f"Output {output_name} not found")  # 如果输出名称既不在节点映射中也不在常量映射中，则引发值错误异常

        self.fx_graph.output(
            args[0]
        )  # 将处理后的第一个参数作为最终输出，解除最终输出周围的额外列表包装
class TS2EPConverter:
    # TorchScript model to ExportedProgram converter

    def __init__(
        self,
        ts_model: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction],
        sample_args: Tuple[Any, ...],
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 初始化函数，用于初始化转换器对象
        log.info(
            """
TS2EPConverter logging starts from here.

INFO: (TORCH_LOGS="export" <cmd>)
    * Log TorchScript IR.

DEBUG: (TORCH_LOGS="+export" <cmd>), additionaly
    * Log conversion IR by IR in a format of [<conversion handler name>] converts [<IR>].
        """
        )

        self.ts_model = ts_model  # 保存输入的 TorchScript 模型
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)
        log.info(f"TorchScript graph\n\n{self.ts_graph}\n")  # noqa: G004

        self.sample_args = sample_args  # 保存样本参数
        self.sample_kwargs = sample_kwargs  # 保存可选的样本关键字参数

        # 根据模型类型创建名称到参数张量的映射
        self.name_to_param_map: Dict[str, torch.Tensor] = (
            dict(ts_model.named_parameters())
            if isinstance(ts_model, torch.jit.ScriptModule)
            else dict()
        )
        # 根据模型类型创建名称到缓冲张量的映射
        self.name_to_buffer_map: Dict[str, torch.Tensor] = (
            dict(ts_model.named_buffers())
            if isinstance(ts_model, torch.jit.ScriptModule)
            else dict()
        )
        self.name_to_non_tensor_attributes: Dict[str, Any] = dict()

        # 提升常量张量到缓冲区
        self.lift_tensor_constants_to_buffer()

    def convert(self) -> ExportedProgram:
        # 执行模型转换为 ExportedProgram 的操作
        blocks_to_lifted_attrs = get_block_to_lifted_attrs(self.ts_graph)
        # 创建 TS2FXGraphConverter 实例并进行图形转换
        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param_map,
            self.name_to_buffer_map,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
        )
        # 转换图形模块为 ExportedProgram 实例
        gm = graph_converter.convert()
        ep = self.retrace_as_exported_program(
            gm, graph_converter.name_to_tensor_constants
        )
        return ep

    def retrace_as_exported_program(
        self, gm: torch.fx.GraphModule, tensor_constants: Dict[str, torch.Tensor]
    ):
        # 重新追溯为导出的程序（ExportedProgram）
        # TODO: 调整输入顺序以匹配 GraphSignature 的约定
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            strict=False,
            pre_dispatch=True,
        )

        # 后处理确保导出的程序状态正确
        # 因为在转换过程中，我们将张量常量设置为 GetAttr，
        # 重新追溯无法识别它们为张量常量，而是将它们视为缓冲区。需要在这里再次设置它们。
        ep._constants = tensor_constants
        for k in tensor_constants:
            ep.state_dict.pop(k, None)
        for spec in ep.graph_signature.input_specs:
            # 将错误追溯为缓冲区的标记为常量张量
            if spec.kind == InputKind.BUFFER and spec.target in tensor_constants:
                spec.kind = InputKind.CONSTANT_TENSOR
        ep.verifier().check(ep)

        return ep
    def lift_tensor_constants_to_buffer(self):
        # 此函数将张量常量属性（例如 self.data = torch.tensor([2,3])）提升为缓冲区。
        # 目前，在存在张量常量时，导出操作会出错，并要求用户将张量常量注册为缓冲区。
        # 由于对于 TorchScript 模型很难手动完成此操作（例如，源代码丢失），此函数自动将张量常量提升为缓冲区。
        # 此函数应该在 TS2EPConverter 中执行，而不是在 TS2FXGraphConverter 中执行，因为它从 self.ts_model 获取属性，
        # 而在 TS2FXGraphConverter 中无法访问 self.ts_model。这类似于我们收集 self.name_to_param_map 和
        # self.name_to_buffer_map 的地方。

        name_to_attribute_fqn: Dict[str, str] = {}

        def get_attr(fqn: str):
            # 根据完全限定名（fqn）获取属性值
            name = fqn.split(".")
            v = self.ts_model
            for n in name:
                v = getattr(v, n)
            return v

        def get_fqn(node: torch._C.Node):
            # 获取节点的完全限定名
            attr_name = node.s("name")
            input_name = node.input().debugName()
            root_attr_name = name_to_attribute_fqn[input_name]
            attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
            return attr_fqn

        def _dfs_get_attr(block):
            # 深度优先遍历图中的节点，获取属性
            for node in block.nodes():
                if node.kind() == "prim::CreateObject":
                    # 如果节点是创建对象的操作，则将其输出名字映射为空字符串
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = ""

                if node.kind() == "prim::GetAttr":
                    # 如果节点是获取属性的操作，则获取属性的完全限定名并检查其值
                    attr_fqn = get_fqn(node)
                    value = get_attr(attr_fqn)
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = attr_fqn
                    if isinstance(value, torch.Tensor):
                        if attr_fqn not in self.name_to_buffer_map:
                            # 将张量常量提升为缓冲区，并发出警告
                            warnings.warn(
                                f"ts converter lifted tensor constant {attr_fqn} to be a buffer"
                            )
                            self.name_to_buffer_map[attr_fqn] = value
                    else:
                        self.name_to_non_tensor_attributes[attr_fqn] = value

                for subblock in node.blocks():
                    _dfs_get_attr(subblock)

        # 调用深度优先遍历函数，从 self.ts_graph 开始遍历属性获取操作
        _dfs_get_attr(self.ts_graph)
```