# `.\pytorch\torch\export\graph_signature.py`

```
# 导入必要的模块和类型定义
# mypy: allow-untyped-defs
import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union

from torch._library.fake_class_registry import FakeScriptObject

# 导出的符号列表
__all__ = [
    "ConstantArgument",
    "CustomObjArgument",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "InputKind",
    "InputSpec",
    "OutputKind",
    "OutputSpec",
    "SymIntArgument",
    "TensorArgument",
]

# 定义 TensorArgument 类，包含一个名称属性
@dataclasses.dataclass
class TensorArgument:
    name: str

# 定义 TokenArgument 类，包含一个名称属性
@dataclasses.dataclass
class TokenArgument:
    name: str

# 定义 SymIntArgument 类，包含一个名称属性
@dataclasses.dataclass
class SymIntArgument:
    name: str

# 定义 CustomObjArgument 类，包含名称、类全限定名和可选的 FakeScriptObject 属性
@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: Optional[FakeScriptObject] = None

# 定义 ConstantArgument 类，包含名称和值的属性
@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: Union[int, float, bool, str, None]

# ArgumentSpec 可以是 TensorArgument、SymIntArgument、ConstantArgument、CustomObjArgument、TokenArgument 中的一种
ArgumentSpec = Union[
    TensorArgument,
    SymIntArgument,
    ConstantArgument,
    CustomObjArgument,
    TokenArgument,
]

# 输入类型枚举定义
class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()
    CUSTOM_OBJ = auto()
    TOKEN = auto()

# 定义输入规格类，包含类型、参数、目标和可选的持久标志
@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: Optional[str]
    persistent: Optional[bool] = None

    def __post_init__(self):
        # 如果类型是 BUFFER，则必须指定持久标志
        if self.kind == InputKind.BUFFER:
            assert (
                self.persistent is not None
            ), "Failed to specify persistent flag on BUFFER."
        # 确保参数类型是预期中的几种之一
        assert isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                ConstantArgument,
                CustomObjArgument,
                TokenArgument,
            ),
        ), f"got {type(self.arg)}"

# 输出类型枚举定义
class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
    TOKEN = auto()

# 定义输出规格类，包含类型、参数和目标
@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: Optional[str]

    def __post_init__(self):
        # 确保参数类型是预期中的几种之一
        assert isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                ConstantArgument,
                TokenArgument,
                CustomObjArgument,
            ),
        ), self.arg

# 将输入和输出列表转换为对应的规格列表的函数
def _sig_to_specs(
    *,
    user_inputs: Set[str],
    inputs_to_parameters: Mapping[str, str],
    inputs_to_buffers: Mapping[str, str],
    user_outputs: Set[str],
    buffer_mutations: Mapping[str, str],
    user_input_mutations: Mapping[str, str],
    grad_params: Mapping[str, str],
    grad_user_inputs: Mapping[str, str],
    loss_output: Optional[str],
    inputs: List[ArgumentSpec],
    outputs: List[ArgumentSpec],
    input_tokens: List[str],
    output_tokens: List[str],
) -> Tuple[List[InputSpec], List[OutputSpec]]:
    # 将输入参数（inp）转换为输入规格（InputSpec）
    def to_input_spec(inp: ArgumentSpec) -> InputSpec:
        # 如果输入参数是 TokenArgument 类型，则返回 TOKEN 类型的输入规格，目标为 None
        if isinstance(inp, TokenArgument):
            return InputSpec(kind=InputKind.TOKEN, arg=inp, target=None)

        # 如果输入参数不是 TensorArgument 类型，则返回 USER_INPUT 类型的输入规格，目标为 None
        if not isinstance(inp, TensorArgument):
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)

        # 获取输入参数的名称
        name = inp.name
        
        # 如果名称存在于 user_inputs 中，则返回 USER_INPUT 类型的输入规格，目标为 None
        if name in user_inputs:
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)
        
        # 如果名称存在于 inputs_to_parameters 中，则返回 PARAMETER 类型的输入规格，
        # 目标为 inputs_to_parameters[name]
        elif name in inputs_to_parameters:
            return InputSpec(
                kind=InputKind.PARAMETER,
                arg=inp,
                target=inputs_to_parameters[name],
            )
        
        # 如果名称存在于 inputs_to_buffers 中，则返回 BUFFER 类型的输入规格，
        # 目标为 inputs_to_buffers[name]，并且标记为持久的（persistent=True）
        elif name in inputs_to_buffers:
            return InputSpec(
                kind=InputKind.BUFFER,
                arg=inp,
                target=inputs_to_buffers[name],
                persistent=True,  # 暂时标记为持久，稍后在跟踪中将区分持久和非持久
            )
        
        # 如果以上条件都不满足，则抛出异常，指示未知的张量输入类型
        else:
            raise AssertionError(f"Unknown tensor input kind: {name}")

    # 将输出参数（o）转换为输出规格（OutputSpec）
    def to_output_spec(idx: int, o: ArgumentSpec) -> OutputSpec:
        # 如果输出参数是 TokenArgument 类型，则返回 TOKEN 类型的输出规格，目标为 None
        if isinstance(o, TokenArgument):
            return OutputSpec(kind=OutputKind.TOKEN, arg=o, target=None)

        # 如果输出参数不是 TensorArgument 类型，则返回 USER_OUTPUT 类型的输出规格，目标为 None
        if not isinstance(o, TensorArgument):
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)

        # 获取输出参数的名称
        name = o.name
        
        # 根据索引（idx）判断输出是属于哪种类型：BUFFER_MUTATION、USER_INPUT_MUTATION、USER_OUTPUT
        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            if name in buffer_mutations:
                return OutputSpec(
                    kind=OutputKind.BUFFER_MUTATION,
                    arg=o,
                    target=buffer_mutations[name],
                )
            elif name in user_input_mutations:
                return OutputSpec(
                    kind=OutputKind.USER_INPUT_MUTATION,
                    arg=o,
                    target=user_input_mutations[name],
                )
            else:
                raise AssertionError(f"Unknown tensor mutation kind: {name}")
        else:
            # 如果输出不属于上述变异类型，则判断其属于哪种类型：GRADIENT_TO_PARAMETER、GRADIENT_TO_USER_INPUT、LOSS_OUTPUT、USER_OUTPUT
            if name in user_outputs:
                return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)

            elif name in grad_params:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_PARAMETER,
                    arg=o,
                    target=grad_params[name],
                )
            elif name in grad_user_inputs:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_USER_INPUT,
                    arg=o,
                    target=grad_user_inputs[name],
                )
            elif name == loss_output:
                return OutputSpec(kind=OutputKind.LOSS_OUTPUT, arg=o, target=None)

            else:
                raise AssertionError(f"Unknown tensor output kind: {name}")
    # 将输入列表中的每个输入转换为输入规格列表
    input_specs = [to_input_spec(inp) for inp in inputs]
    
    # 使用 enumerate 函数遍历输出列表，为每个输出生成输出规格，返回输出规格列表
    output_specs = [to_output_spec(idx, o) for idx, o in enumerate(outputs)]
    
    # 返回转换后的输入规格列表和输出规格列表作为结果
    return input_specs, output_specs
# 使用 dataclasses 模块的 dataclass 装饰器创建 ExportBackwardSignature 类，表示导出反向签名
@dataclasses.dataclass
class ExportBackwardSignature:
    # 梯度到参数的映射字典，键为梯度名称，值为参数名称
    gradients_to_parameters: Dict[str, str]
    # 梯度到用户输入的映射字典，键为梯度名称，值为用户输入名称
    gradients_to_user_inputs: Dict[str, str]
    # 损失输出的字符串表示
    loss_output: str


# 使用 dataclasses 模块的 dataclass 装饰器创建 ExportGraphSignature 类，表示导出图的签名
@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    gurantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self):
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0) # In-place addition

                return output

    Resulting Graph would be::

        graph():
            %arg0_1 := placeholder[target=arg0_1]
            %arg1_1 := placeholder[target=arg1_1]
            %arg2_1 := placeholder[target=arg2_1]
            %arg3_1 := placeholder[target=arg3_1]
            %arg4_1 := placeholder[target=arg4_1]
            %add_tensor := call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %arg0_1), kwargs = {})
            %mul_tensor := call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %arg1_1), kwargs = {})
            %mul_tensor_1 := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %arg2_1), kwargs = {})
            %add_tensor_1 := call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
            %add_tensor_2 := call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1.0), kwargs = {})
            return (add_tensor_2, add_tensor_1)
    """

    # 此类模型导出图的输入输出签名
    # 参数、缓冲区和常量张量的扁平化列表作为输入
    # 被变异的输入和用户扁平化输出的列表作为输出
    pass
    Resulting ExportGraphSignature would be::

        ExportGraphSignature(
            input_specs=[
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='my_parameter'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg1_1'), target='my_buffer1'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg2_1'), target='my_buffer2'),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg4_1'), target=None)
            ],
            output_specs=[
                OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add_2'), target='my_buffer2'),
                OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add_1'), target=None)
            ]
        )
    """
    # 定义 ExportGraphSignature 类的文档字符串，描述其生成的签名对象结构

    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]

    # A list of parameters uniquely identified by mangled fully qualified name
    @property
    def parameters(self) -> Collection[str]:
        # TODO Make this tuple.
        # 返回所有参数的目标名称列表，仅包括类型为 PARAMETER 的 InputSpec
        return [
            s.target
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            if isinstance(s.target, str)
        ]

    # A list of buffers uniquely identified by mangled fully qualified name
    @property
    def buffers(self) -> Collection[str]:
        # TODO Make this tuple.
        # 返回所有缓冲区的目标名称列表，仅包括类型为 BUFFER 的 InputSpec
        return [
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if isinstance(s.target, str)
        ]

    @property
    def non_persistent_buffers(self) -> Collection[str]:
        # 返回所有非持久性缓冲区的目标名称列表，仅包括类型为 BUFFER 且 persistent 属性为 False 的 InputSpec
        return [
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if s.persistent is False
            if isinstance(s.target, str)
        ]

    # A list of lifted constant tensors
    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        # TODO Make this tuple.
        # 返回所有 lifted constant tensors 的目标名称列表，仅包括类型为 CONSTANT_TENSOR 的 InputSpec
        return [
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            if isinstance(s.target, str)
        ]

    @property
    def lifted_custom_objs(self) -> Collection[str]:
        # TODO Make this tuple.
        # 返回所有 lifted custom objects 的目标名称列表，仅包括类型为 CUSTOM_OBJ 的 InputSpec
        return [
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            if isinstance(s.target, str)
        ]

    # Graph node names of pytree-flattened inputs of original program
    @property
    # 返回一个由用户输入组成的列表，可能包括整数、浮点数、布尔值、None、字符串等类型
    def user_inputs(self) -> Collection[Union[int, float, bool, None, str]]:
        # 初始化一个空列表，用于存储用户输入
        user_inputs: List[Union[int, float, bool, None, str]] = []
        # 遍历输入规格列表
        for s in self.input_specs:
            # 如果输入类型不是用户输入，则跳过当前循环
            if s.kind != InputKind.USER_INPUT:
                continue

            # 根据不同类型的参数，将相应的参数名或值添加到用户输入列表中
            if isinstance(s.arg, (TensorArgument, SymIntArgument, CustomObjArgument)):
                user_inputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_inputs.append(s.arg.value)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user inputs")
        # 返回元组形式的用户输入列表
        return tuple(user_inputs)

    # 返回一个由用户输出组成的列表，可能包括整数、浮点数、布尔值、None、字符串等类型
    @property
    def user_outputs(self) -> Collection[Union[int, float, bool, None, str]]:
        # 初始化一个空列表，用于存储用户输出
        user_outputs: List[Union[int, float, bool, None, str]] = []
        # 遍历输出规格列表
        for s in self.output_specs:
            # 如果输出类型不是用户输出，则跳过当前循环
            if s.kind != OutputKind.USER_OUTPUT:
                continue

            # 根据不同类型的参数，将相应的参数名或值添加到用户输出列表中
            if isinstance(s.arg, (TensorArgument, SymIntArgument)):
                user_outputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_outputs.append(s.arg.value)
            elif isinstance(s.arg, CustomObjArgument):
                user_outputs.append(s.arg.name)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user output")
        # 返回元组形式的用户输出列表
        return tuple(user_outputs)

    # 返回一个映射，将图输入节点名称映射到参数名称，用于标记 lifted parameter
    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        return {
            s.arg.name: s.target
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        }

    # 返回一个映射，将图输入节点名称映射到缓冲区名称，用于标记 lifted buffer
    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        return {
            s.arg.name: s.target  # type: ignore[union-attr, misc]
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        }

    # 返回一个映射，将图输出节点名称映射到被原始程序中修改的缓冲区名称，用于标记 mutated buffer
    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        return {
            s.arg.name: s.target
            for s in self.output_specs
            if s.kind == OutputKind.BUFFER_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        }
    def user_inputs_to_mutate(self) -> Mapping[str, str]:
        # 返回一个字典，将输出规范中符合以下条件的项目映射为字典条目：
        # - OutputKind.USER_INPUT_MUTATION 类型
        # - s.arg 是 TensorArgument 类型
        # - s.target 是字符串类型
        return {
            s.arg.name: s.target
            for s in self.output_specs
            if s.kind == OutputKind.USER_INPUT_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        }

    # 一个字典，将图输入节点名称映射到升级的张量常量。
    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        # 返回一个字典，将输入规范中符合以下条件的项目映射为字典条目：
        # - InputKind.CONSTANT_TENSOR 类型
        # - s.arg 是 TensorArgument 类型
        # - s.target 是字符串类型
        return {
            s.arg.name: s.target
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        }

    @property
    def inputs_to_lifted_custom_objs(self) -> Mapping[str, str]:
        # 返回一个字典，将输入规范中符合以下条件的项目映射为字典条目：
        # - InputKind.CUSTOM_OBJ 类型
        # - s.arg 是 CustomObjArgument 类型
        # - s.target 是字符串类型
        return {
            s.arg.name: s.target
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            and isinstance(s.arg, CustomObjArgument)
            and isinstance(s.target, str)
        }

    @property
    def backward_signature(self) -> Optional[ExportBackwardSignature]:
        # 构建反向传播的签名信息对象，包括损失输出、梯度到参数的映射和梯度到用户输入的映射。
        loss_output = None
        gradients_to_parameters: Dict[str, str] = {}
        gradients_to_user_inputs: Dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                assert loss_output is None
                assert isinstance(spec.arg, TensorArgument)
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_user_inputs[spec.arg.name] = spec.target

        if loss_output is None:
            return None

        return ExportBackwardSignature(
            loss_output=loss_output,
            gradients_to_parameters=gradients_to_parameters,
            gradients_to_user_inputs=gradients_to_user_inputs,
        )

    # 映射断言依赖的令牌索引到输出中的断言依赖令牌名称。
    # 在经过 aot_autograd 处理后，输出的形状将是：(updated_inputs, user_outputs, dep_token)。
    @property
    def assertion_dep_token(self) -> Optional[Mapping[int, str]]:
        # 返回空值，表示可能没有断言依赖令牌映射。
        return None

    @property
    def input_tokens(self) -> List[str]:
        # 返回一个列表，其中包含所有输入规范中类型为 InputKind.TOKEN 的 TokenArgument 名称。
        input_tokens = []
        for s in self.input_specs:
            if s.kind == InputKind.TOKEN:
                assert isinstance(s.arg, TokenArgument)
                input_tokens.append(s.arg.name)
        return input_tokens

    @property
    # 返回一个包含所有输出 token 名称的列表
    def output_tokens(self) -> List[str]:
        output_tokens = []
        # 遍历所有输出规格
        for s in self.output_specs:
            # 如果规格类型是 TOKEN
            if s.kind == OutputKind.TOKEN:
                # 断言参数类型为 TokenArgument
                assert isinstance(s.arg, TokenArgument)
                # 将 token 名称添加到输出列表中
                output_tokens.append(s.arg.name)
        # 返回输出 token 名称列表
        return output_tokens

    # 对象初始化完成后执行的方法
    def __post_init__(self) -> None:
        # 获取断言依赖的 token
        assertion_dep_token = self.assertion_dep_token
        # 如果断言依赖的 token 为 None，则直接返回
        if assertion_dep_token is None:
            return
        # 断言断言依赖的 token 长度为 1
        assert len(assertion_dep_token) == 1
        # 获取断言依赖的 token 的索引值
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        # 断言用户输出和需要变异的缓冲区的总数等于断言依赖的 token 的索引值
        assert (
            len(self.user_outputs) + len(self.buffers_to_mutate)
            == assertion_dep_token_index
        )

    # 替换签名中所有旧名称为新名称的方法
    def replace_all_uses(self, old: str, new: str):
        """
        替换签名中所有使用旧名称的地方为新名称。
        """
        assert isinstance(old, str)
        assert isinstance(new, str)
        # 支持的参数类型
        arg_types = (TensorArgument, SymIntArgument, CustomObjArgument, TokenArgument)
        # 遍历所有输出规格
        for o in self.output_specs:
            # 如果参数类型属于支持的类型之一
            if isinstance(o.arg, arg_types):
                # 如果规格中的参数名称等于旧名称
                if o.arg.name == old:
                    # 替换为新名称
                    o.arg.name = new
        # 遍历所有输入规格
        for i in self.input_specs:
            # 如果参数类型属于支持的类型之一
            if isinstance(i.arg, arg_types):
                # 如果规格中的参数名称等于旧名称
                if i.arg.name == old:
                    # 替换为新名称
                    i.arg.name = new

    # 返回一个替换钩子函数
    def get_replace_hook(self):
        def _(old, new, user):
            # 如果用户操作是输出或输入
            if user.op in ("output", "input"):
                # 调用替换所有使用方法，替换旧名称为新名称
                self.replace_all_uses(old.name, new)

        return _
```