# `.\pytorch\torch\_functorch\_aot_autograd\schemas.py`

```py
"""
The various dataclasses, Enums, namedtuples etc used in AOTAutograd. This includes
input/output types, metadata, config, function signatures etc.
"""

# 引入必要的库和模块
import collections
import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union

import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# 引入外部依赖模块和包
from .. import config

# 引入自定义的功能性工具函数
from .functional_utils import (
    _check_if_mutation_can_be_in_graph,
    FunctionalTensorMetadataEq,
)
# 引入严格的 zip 函数
from .utils import strict_zip

# 将 strict_zip 赋值给 zip，使其在当前模块可直接使用
zip = strict_zip

# 定义枚举类型 OutputType，描述输出类型的不同情况
OutputType = Enum(
    "OutputType",
    (
        # output 不是别名
        "non_alias",
        # output 是输入的别名
        "alias_of_input",
        # output **是**一个输入张量
        "is_input",
        # output 有一个 ._base 张量，这是一个图中间结果。
        # 我们需要返回其 ._base 作为图输出，以正确填充其 requires_grad 信息。
        # 指示运行时代码从基张量生成当前输出，graph_intermediates[base_idx]
        "alias_of_intermediate_save_as_output",
        # 与上述相同；但我们不需要将其 ._base 明确添加为图输出，
        # 因为它已经 **是** 一个图输出。
        "alias_of_intermediate",
        # 与上述相同；但输出的 ._base **已经** 是用户输出。
        # 指示运行时代码从基张量 user_outputs[base_idx] 生成当前输出
        "alias_of_intermediate_base_is_user_output",
        # 参见注释 [Intermediate Bases Optimization]
        "unsafe_view_alias",
        # output 是一个别名，但具有自定义的 autograd.Function 反向函数。
        # 在这种情况下，我们不想进行视图重播，因为我们无法重播自定义函数。
        # 相反，我们将正常对待此输出，并将其反向跟踪到图中。
        "custom_function_view",
    ),
)


# 这个类存储有关每个用户输出的信息。
@dataclass(frozen=True)
class OutputAliasInfo:
    # 告诉我们这个输出是：
    # (1) 一个常规的（非别名）输出
    # (2) 一个前向输入的别名
    # (3) **是** 一个前向输入（"alias_of_input" 的特殊情况）
    # (4) 一个中间结果的别名（内部跟踪前向输出的别名）
    # (5) 一个中间结果的别名，显式要求将中间结果作为图输出返回
    output_type: OutputType
    # 输出的原始类型（torch.Tensor, SymInt 等）
    raw_type: type
    # 如果是上述的情况 (1)，则 base_idx 是 None
    # 如果是上述情况 (2) 或 (3)，则
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # 如果条件是(4)或(5)，则
    # - 告诉我们此别名的基础是 output_graph_intermediates[base_idx]
    #   这里，这指的是*直接*跟踪的索引
    # 如果条件是(6)，则：
    # - 告诉我们此别名的基础是 output_user_fwds[base_idx]
    #   这里，这指的是*直接*跟踪的索引
    base_idx: Optional[int]
    
    # 如果它是一个张量，指示动态维度是什么（否则为None）
    dynamic_dims: Optional[Set[int]]
    
    # 是否需要梯度
    requires_grad: bool
    
    # 表示此输出的 FunctionalTensorWrapper。
    #
    # 提供给我们重新播放其视图的方式。
    #
    # 我们需要用这个类包装实际的 FunctionalTensorWrapper，以便仅比较张量的元数据。
    # 这是因为在 AOTAutograd 中模型的转换过程中，ViewMeta 的序列和基础张量可能会发生变化。
    functional_tensor: Optional[FunctionalTensorMetadataEq] = None
# Define an enumeration for different mutation types
class MutationType(Enum):
    NOT_MUTATED = 1                # Indicates no mutation occurred
    MUTATED_IN_GRAPH = 2           # Indicates mutation occurred and was captured in the computation graph
    MUTATED_OUT_GRAPH = 3          # Indicates mutation occurred but was not captured in the computation graph

# Dataclass that holds information about user inputs related to mutations
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool                    # Flag indicating if the input is a leaf node
    mutates_data: bool               # Flag indicating if data mutation occurs
    mutates_metadata: bool           # Flag indicating if metadata mutation occurs
    mutations_hidden_from_autograd: bool  # Flag indicating if mutations are hidden from autograd
    mutations_under_no_grad_or_inference_mode: bool  # Flag indicating if mutations occur under no grad or inference mode
    mutation_inductor_storage_resize: bool  # Flag indicating if mutation inductor triggers storage resize
    mutates_storage_metadata: bool   # Flag indicating if storage metadata mutation occurs
    requires_grad: bool              # Flag indicating if gradients are required
    keep_input_mutations: bool       # Flag indicating if input mutations should be kept

    def __post_init__(self):
        if self.mutates_storage_metadata:
            # Ensure mutates_metadata is always true when mutates_storage_metadata is true
            # This is a runtime guarantee for tensor metadata consistency
            assert self.mutates_metadata

    @functools.cached_property
    def mutation_type(self) -> MutationType:
        # Determine the type of mutation based on various flags
        if (not self.mutates_data) and (not self.mutates_metadata) and not (self.mutation_inductor_storage_resize):
            return MutationType.NOT_MUTATED

        if _check_if_mutation_can_be_in_graph(
            self.keep_input_mutations,
            self.mutates_data,
            self.mutates_metadata,
            self.mutations_hidden_from_autograd,
            self.mutations_under_no_grad_or_inference_mode,
            self.mutates_storage_metadata,
            self.mutation_inductor_storage_resize,
            self.requires_grad,
        ):
            return MutationType.MUTATED_IN_GRAPH

        return MutationType.MUTATED_OUT_GRAPH


# Dataclass that provides metadata for creating subclasses
@dataclass
class SubclassCreationMeta:
    """
    Used for AOTDispatch.
    This dataclass provides information for reconstructing a tensor subclass from flat inputs.
    It deals with the translation between user's subclass inputs/outputs and backend compiler's flat tensor graph.
    
    Complications arise from subclass inputs potentially containing multiple inner tensors,
    requiring careful tracking of indices mapping to subclass tensors in the dense-tensor-only graph.
    """

    # Index in the flat tensor graph where subclass tensors start
    flat_tensor_start_idx: int
    # Total count of arguments including any inner tensor subclasses
    arg_count: int
    # Metadata and attributes produced by __tensor_flatten__ of the subclass
    # 在 __tensor_unflatten__ 方法中需要使用这些属性以及 outer_size / outer_stride，
    # 以便将它们传递给 __tensor_unflatten__ 方法
    attrs: Dict[str, Union["SubclassCreationMeta", None]]
    outer_size: List[int]
    outer_stride: List[int]
    meta: Any
    # 存储原始的子类本身。
    # 这是必要的，因为我们需要在原始子类上保留 autograd 元数据
    # （这保证了它是一个包含虚拟张量的包装子类，在运行时保持这个不会泄露内存）
    original_subclass: Any

    def creation_fn(self, all_args, *, is_runtime: bool):
        # 用于存储内部张量
        inner_tensors = {}

        # 当前起始索引从 flat_tensor_start_idx 开始
        curr_start_idx = self.flat_tensor_start_idx
        for attr, creation_meta in self.attrs.items():
            if creation_meta is None:
                # 如果 creation_meta 为 None，则从 all_args 直接取得子类
                subclass = all_args[curr_start_idx]
                curr_start_idx += 1
            else:
                # 否则，通过 creation_meta 的 creation_fn 方法递归创建子类
                subclass = creation_meta.creation_fn(all_args, is_runtime=is_runtime)
                curr_start_idx += creation_meta.arg_count
            inner_tensors[attr] = subclass

        # 使用 self.original_subclass 的类型调用 __tensor_unflatten__ 方法重建结构
        rebuilt = type(self.original_subclass).__tensor_unflatten__(
            inner_tensors, self.meta, self.outer_size, self.outer_stride
        )

        if not is_runtime:
            # 在将内部密集张量封装成子类后，需要确保新的包装子类具有正确的 autograd 元数据，
            # 因为我们将通过子类跟踪 autograd 引擎。
            # 但在运行时不需要跟踪 autograd 引擎，因此无需计算此额外的元数据！
            torch._mirror_autograd_meta_to(self.original_subclass, rebuilt)  # type: ignore[attr-defined]

        return rebuilt

    def __post_init__(self):
        # 断言以确保我们不会泄露内存
        assert is_fake(self.original_subclass)

        # 这保存了子类嵌套结构的类型，以便与运行时的切线输入进行比较。
        # 我们确实希望在 AOT 时间计算这个，因为它在热路径中被调用
        from .subclass_utils import get_types_for_subclass

        self.subclass_type = get_types_for_subclass(self.original_subclass)
# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
@dataclass(eq=False)
class ViewAndMutationMeta:
    # length = # user inputs
    # This gives us info about every input, and what sort of mutation happened to it (if any)
    input_info: List[InputAliasInfo]

    # length = # user outputs
    # This gives us info about every output (mostly around whether it aliases other tensors)
    output_info: List[OutputAliasInfo]

    # length = the number of intermediate bases appended as outputs to the end of the forward graph.
    # Note: this is not necessarily the same thing as:
    #   len([x for x in output_info if x.output_type == OutputType.alias_of_intermediate])
    # Because outputs might share a ._base, or an output's ._base might itself be
    # another user output (in both cases, we won't redundantly append bases to the end of the graph)
    num_intermediate_bases: int

    # For inference only: instructs us to keep data-only input mutations directly in the graph
    keep_input_mutations: bool

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # These are the FakeTensor (or potential SymInt) outputs that we traced from our
    # metadata pass of the user's forward function.
    # Their only use today is to pass them as a best-guess for tangents when tracing the joint.
    # Stashing them as part of our "metadata" makes it simpler if we want to run our analysis
    # pass once, and re-use the output throughout AOTAutograd
    traced_tangents: List[Any]

    # Each of these is a list telling us about subclasses for the inputs/outputs/grad_outs
    # They are used throughout AOTDispatch to tell us how to generate a list of subclass tensors,
    # Given a (potentially larger) list of plain torch tensors.

    # Taking subclass_inp_meta as an example:
    #   subclass_inp_meta[i] = j (an int) tells us:
    #     "The i'th user input is not a subclass, and corresponds to inputs[j] of the plain-tensor graph."
    #   subclass_inp_meta[i] = SubclassCreationMeta(flat_tensor_start_idx=3, arg_count=2)
    #     "The i'th user input is subclass holding two inner tensors, which are
    #      inputs[3] and inputs[4] of the plain-tensor graph".
    subclass_inp_meta: List[Union[int, SubclassCreationMeta]]
    # So, the full set of outputs to the forward graph looks something like:
    # (*mutated_inps, *user_outs, *intermediate_bases, *saved_for_bw_tensors)
    # where the first 3 of those 4 can be subclasses
    # (but not saved_for_bw tensors, since these are internal to the compiler
    # and not user visible, so there's no point in wrapping/unwrapping them at runtime).
    # This list contains subclass information on all of the fw graph outputs
    #python
    # except for saved_for_bw_tensors.
    # subclass_fw_graph_out_meta: List[Union[int, SubclassCreationMeta]]
    # length = # backward graph inputs
    # subclass_tangent_meta: List[Union[int, SubclassCreationMeta]]
    # TODO: we should kill this
    # (need to default it to not break internal)
    # is_train: bool = False

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # At runtime, we don't keep the traced_tangents around since they're not serializable.
    # Instead, we keep any necessary subclass metadata necessary about each traced_tangent.
    # This list is generated after calling make_runtime_safe().
    # traced_tangent_metas: Optional[List[Any]] = None

    # num_symints_saved_for_bw: Optional[int] = None

    # The grad_enabled mutation that will be emitted in the runtime_wrapper epilogue
    # NOTE: AOTAutograd will assume that the ambient `is_grad_enabled` is the grad mode
    # that is intended to be in effect prior to running the graph, in keeping with
    # equivalence to eager mode. It is the responsibility of upstream graph acquisition
    # to reset the grad mode to its pre-graph value prior to calling aot_autograd.
    # grad_enabled_mutation: Optional[bool] = None

    # Keeps track of whether `torch.use_deterministic_algorithms` was turned on
    # when the forward was run. If deterministic mode was turned off during the
    # forward, but is turned on during the backward call, then an error is
    # raised
    # deterministic: Optional[bool] = None

    # Keeps track of which input indices store parameters (which we will treat as static)
    # static_parameter_indices: List[int] = field(default_factory=list)

    # Map of effect type (ex. _EffectType.ORDERED) to token.  If there are
    # side-effectful operators, FunctionalTensorMode will populate this
    # dictionary telling us how many tokens we will need during tracing.
    # tokens: Dict[Any, torch.Tensor] = field(default_factory=dict)

    # Only filled in if/when we trace the joint function
    # If an input requires grad and is mutated in the backward, it is only safe to keep the mutation
    # in the graph if gradients are disabled while the backward runs
    # (grad mode is disabled by default when users run the backward, but can be turned on with create_graph=True)
    # At runtime during the backward, we use this list of indices to error properly if we find out
    # that it was not safe to include a backward mutation in the graph.
    # indices_of_inputs_that_requires_grad_with_mutations_in_bw: List[int] = field(
    #     default_factory=list
    # )
    def make_runtime_safe(self):
        """
        There are various fields in ViewAndMutationMeta that aren't serializable. This function is called after all tracing
        is completed to simplify certain fields in the metadata so that they can be safely cached.

        Doing so may lose information (in the case of traced_tangents), but none of the information is needed at runtime.
        """
        # TODO: This function is only a best effort: there are other fields that may not be cache safe
        # (i.e., there's no guarantee that tensor_flatten() returns a serializable result), or that
        # SubclassCreationMeta is cache safe.
        assert self.traced_tangent_metas is None  # 断言确保 self.traced_tangent_metas 为 None

        def extract_metadata(t):
            if isinstance(t, torch.Tensor) and is_traceable_wrapper_subclass(t):
                (inner_tensors, flatten_spec) = t.__tensor_flatten__()  # type: ignore[attr-defined]
                # Technically, we only need the flatten_spec, not the inner tensors.
                # However, some Tensor subclasses (like TwoTensor) may have flatten_spec = None.
                # And we want to be able to assert that this metadata is non-None,
                # to distinguish between "this was a tensor subclass with no metadata" vs.
                # "this wasn't a tensor subclass at all".
                return (inner_tensors, flatten_spec)
            else:
                return None

        self.traced_tangent_metas = [extract_metadata(t) for t in self.traced_tangents]
        # Clear traced tangents at runtime
        self.traced_tangents = []  # 清空 traced_tangents，以确保在运行时不包含跟踪信息

    @property
    def tensors_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None  # 断言确保 self.num_symints_saved_for_bw 不为 None
        if self.num_symints_saved_for_bw > 0:
            return slice(self.num_forward, -self.num_symints_saved_for_bw)
        else:
            return slice(self.num_forward, None)

    @property
    def symints_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None  # 断言确保 self.num_symints_saved_for_bw 不为 None
        if self.num_symints_saved_for_bw > 0:
            return slice(-self.num_symints_saved_for_bw, None)
        else:
            return slice(0, 0)  # 空切片，表示没有保存的符号整数

    def __eq__(self, other):
        if not isinstance(other, ViewAndMutationMeta):
            return NotImplemented
        return (
            self.input_info == other.input_info
            and self.output_info == other.output_info
            and self.num_intermediate_bases == other.num_intermediate_bases
            and self.keep_input_mutations == other.keep_input_mutations
            and self.is_rng_op_functionalized == other.is_rng_op_functionalized
            and self.num_outputs_rng_offset == other.num_outputs_rng_offset
            and len(self.traced_tangents) == len(other.traced_tangents)
            and all(
                x.shape == y.shape and x.dtype == y.dtype
                for x, y in zip(self.traced_tangents, other.traced_tangents)
            )
        )
@dataclass(eq=False)
class SubclassMeta:
    # A copy of all forward metadata, but computed on the *dense* tensor forward (after desugaring subclasses)
    # So for example, if the user had a model containing two `TwoTensor` inputs,
    # Then `SubclassMeta.fw_metadata.input_infos` would have length 4 here.
    # 存储所有前向元数据的副本，但是在密集张量前向计算后计算（子类展开后）
    # 例如，如果用户的模型包含两个 `TwoTensor` 输入，
    # 那么 `SubclassMeta.fw_metadata.input_infos` 在这里将有长度为 4。

    fw_metadata: ViewAndMutationMeta  # 存储视图和变异元数据的对象引用

    # Note: [Computing Subclass Metadata about grad_inputs]
    # Given a list of flattened, plain tensor grad_inputs, this tells us how to reconstruct the grad_input subclasses
    #
    # You might think: why not just assume that all grad_inputs will have the same subclass-ness as the original inputs?
    # (AOTAutograd generally assumes other properties, e.g. that grad_outputs are contiguous)
    #
    # This doesn't really work though. take this example:
    #
    # def f(DoubleTensor, DenseTensor):
    #     return DoubleTensor  * DenseTensor
    #
    # In the above example, the .grad field of *both* DoubleTensor and DenseTensor will be a DoubleTensor.
    # When we trace out a joint fw-bw graph, we'll end up returning two subclasses for the two grad_inputs.
    # This means that our backward graph will return 4 outputs (two dense tensors for each DoubleTensor grad_input)
    # and we need to properly store the metadata that tells us how to turn these 4 outputs back into DoubleTensors.
    #
    # Note that this info **cannot** easily be figured out from ViewAndMutationMeta.
    # We can only compute this info by tracing the entire joint and examining the grad_inputs that we computed.
    #
    # See Note: [AOTAutograd Backward Guards]
    # This will also eventually require us to install backward guards,
    # in case we made incorrect assumptions about the subclass-ness of our grad_outputs
    #
    # Optional field because we don't compute for inference graphs
    grad_input_metas: Optional[List[Union[int, SubclassCreationMeta]]] = None
    # 计算关于 grad_inputs 的子类元数据的说明
    # 给定一个扁平化的普通张量 grad_inputs 列表，这告诉我们如何重建 grad_input 的子类
    #
    # 你可能会想：为什么不假设所有 grad_inputs 都与原始输入具有相同的子类性质呢？
    # （AOTAutograd 通常假设其他属性，例如 grad_outputs 是连续的）
    #
    # 但这并不适用。举个例子：
    #
    # def f(DoubleTensor, DenseTensor):
    #     return DoubleTensor  * DenseTensor
    #
    # 在上面的例子中，DoubleTensor 和 DenseTensor 的 .grad 字段都将是一个 DoubleTensor。
    # 当我们跟踪联合前向-后向图时，我们将为两个 grad_input 返回两个子类。
    # 这意味着我们的后向图将返回 4 个输出（每个 DoubleTensor grad_input 对应两个密集张量），
    # 我们需要正确存储元数据，告诉我们如何将这 4 个输出转换回 DoubleTensors。
    #
    # 请注意，这些信息无法轻易地从 ViewAndMutationMeta 中推断出来。
    # 我们只能通过跟踪整个联合并检查我们计算的 grad_inputs 来计算这些信息。
    #
    # 参见注释：[AOTAutograd Backward Guards]
    # 这最终还将要求我们安装后向保护，以防我们对 grad_outputs 的子类性质做出错误假设

    def __init__(self):
        # The fields in this class get set after its construction.
        # 此类中的字段在构造后设置。

# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor
    # 此类的存在原因：
    # - AOT autograd 中的 autograd.Function.forward() 返回可能与输入别名的输出
    # - 我们只关心这些别名上的元数据，以便我们可以重新生成它们。
    #   我们不希望它们参与 autograd.Function。
    # 我们通过将它们包装在一个不透明的类中来实现这一点，这样 autograd.Function
    # 就不会将它们视为张量处理。

@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    """
    # 提供导出的联合前向-后向图的后向部分的信息。
    # 对于特定的 fx GraphModule，此类包含以下信息：
    # (1) 每个梯度（后向输出）到其对应的参数（前向输入）的映射
    # (2) 每个梯度（后向输出）到其对应的用户输入（前向输入）的映射
    # 定义一个变量，用于存储梯度与参数之间的映射关系，其中键为节点名称，值为对应的参数名称
    gradients_to_parameters: Dict[str, str]
    
    # 定义一个变量，用于存储梯度与用户输入之间的映射关系，其中键为节点名称，值为对应的用户输入名称
    gradients_to_user_inputs: Dict[str, str]
    
    # 定义一个变量，用于存储损失函数的输出节点名称，这是我们在反向传播过程中需要进行梯度计算的节点
    loss_output: str
@dataclass
class GraphSignature:
    """
    Provides information about an exported module.
    For a particular fx GraphModule, this class contains information on:
    (1) Which graph inputs are parameters, buffers, or user inputs
    (2) (for params/buffers) a mapping from the name of each graph argument
        to its parameter/buffer FQN in the original nn.Module.
    (3) If there are input mutations, these are represented as extra outputs
        in the fx GraphModule. We provide a mapping from these
        extra output names to the names of the actual inputs.
    (4) The pytree metadata on how to flatten/unflatten inputs and outputs.
        The corresponding FX GraphModule only accepts and returns
        pytree-flattened inputs/outputs.
    (5) (Optionally) if the FX is a joint forward-backward graph, we provide
        a signature on the backward section of the joint graph.
    """

    parameters: List[FQN]
    buffers: List[FQN]

    user_inputs: List[GraphInputName]
    user_outputs: List[GraphOutputName]
    inputs_to_parameters: Dict[GraphInputName, FQN]
    inputs_to_buffers: Dict[GraphInputName, FQN]

    # If the user's module mutates a buffer,
    # it's represented in the graph as an extra graph output.
    # This dict is a mapping from
    # "graph outputs that correspond to updated buffers"
    # to the FQN names of those mutated buffers.
    buffers_to_mutate: Dict[GraphOutputName, FQN]
    user_inputs_to_mutate: Dict[GraphOutputName, GraphInputName]

    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    backward_signature: Optional[BackwardSignature]

    input_tokens: List[GraphInputName]
    output_tokens: List[GraphOutputName]

    @classmethod
    def from_tracing_metadata(
        cls,
        *,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        graph_input_names: List[str],
        graph_output_names: List[str],
        view_mutation_metadata: ViewAndMutationMeta,
        named_parameters: List[str],
        named_buffers: List[str],
        num_user_inputs: int,
        num_user_outputs: int,
        loss_index: Optional[int],
        backward_signature: Optional[BackwardSignature],
    ):
        """
        Class method to construct a GraphSignature object from tracing metadata.

        Parameters:
        - in_spec: The pytree.TreeSpec for input metadata.
        - out_spec: The pytree.TreeSpec for output metadata.
        - graph_input_names: List of names of graph inputs.
        - graph_output_names: List of names of graph outputs.
        - view_mutation_metadata: Metadata for viewing and mutation.
        - named_parameters: List of named parameters.
        - named_buffers: List of named buffers.
        - num_user_inputs: Number of user inputs.
        - num_user_outputs: Number of user outputs.
        - loss_index: Index of loss (optional).
        - backward_signature: Signature for backward section (optional).

        Returns:
        - GraphSignature object constructed from the provided metadata.
        """
        pass


@dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: Dict[Callable, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: Optional[List[Source]] = None
    inference_compiler: Optional[Callable] = None
    enable_log: bool = True
    # this is always false outside of export.
    pre_dispatch: bool = False

    # Key to use for AOTAutogradCache
    cache_key: Optional[str] = None
    # 初始化对象后处理方法，用于在对象创建后进行额外的初始化操作
    def __post_init__(self):
        # 如果设置了预调度标志
        if self.pre_dispatch:
            # 如果不是导出状态，则抛出断言错误
            assert self.is_export, "Can only have pre_dispatch IR for export."
# 使用 collections 模块的 namedtuple 函数创建一个名为 SubclassTracingInfo 的命名元组
SubclassTracingInfo = collections.namedtuple(
    # 元组的名称为 "SubclassTracingInfo"
    "SubclassTracingInfo",
    # 元组的字段包括三个：plain_tensor_trace_fn、plain_tensor_args 和 maybe_subclass_meta
    ["plain_tensor_trace_fn", "plain_tensor_args", "maybe_subclass_meta"],
)
```