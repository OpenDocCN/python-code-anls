# `.\pytorch\torch\ao\quantization\quantize_fx.py`

```
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
    Scope,
    ScopeContextManager
)
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
    BackendConfig,
    get_tensorrt_backend_config,
)
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
    ConvertCustomConfig,
    FuseCustomConfig,
    PrepareCustomConfig,
)
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping

def attach_preserved_attrs_to_model(
    model: Union[GraphModule, torch.nn.Module],
    preserved_attrs: Dict[str, Any],
) -> None:
    """ Store preserved attributes to the model.meta so that it can be preserved during deepcopy
    """
    model.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = copy.copy(preserved_attrs)  # type: ignore[operator, index, assignment]
    # set the preserved attributes in the model so that user can call
    # model.attr as they do before calling fx graph mode quantization
    for attr_name, attr in model.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():  # type: ignore[index, union-attr]
        setattr(model, attr_name, attr)

def _check_is_graph_module(model: torch.nn.Module) -> None:
    """ Check if the input model is an instance of GraphModule.
        Raise ValueError if it's not.
    """
    if not isinstance(model, GraphModule):
        raise ValueError(
            "input model must be a GraphModule, "
            + "Got type:"
            + str(type(model))
            + " Please make "
            + "sure to follow the tutorials."
        )

def _attach_meta_to_node_if_not_exist(model: GraphModule) -> None:
    """ Attach meta field to all nodes of the graph if it does not exist,
    meta field is a field stores some meta information about the node, such
    as dtype and shape information for output of the node, this only exists
    if the program is captured by make_fx (used in quantize_pt2e flow), if
    the program is captured by torch.fx symbolic tracing, this field may not exist,
    so we add it here to avoid checking this all over the places
    """
    for node in model.graph.nodes:
        if not hasattr(node, "meta"):
            node.meta = {}

def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    r""" Swap FloatFunctional with FXFloatFunctional
    """
    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
            modules_to_swap.append(name)
        else:
            _swap_ff_with_fxff(module)

    for name in modules_to_swap:
        del model._modules[name]
        model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()

def _fuse_fx(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,



    # model: GraphModule 类型的参数，表示输入的模型对象
    model: GraphModule,
    # is_qat: bool 类型的参数，表示是否处于量化训练模式的标志
    is_qat: bool,
    # fuse_custom_config: 可选参数，可以是 FuseCustomConfig 类型、字典类型（键为字符串，值为任意类型），或者为 None
    fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None] = None,
    # backend_config: 可选参数，可以是 BackendConfig 类型、字典类型（键为字符串，值为任意类型），或者为 None
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,


These annotations describe each parameter's type and potential values accepted by the function or method.
`
) -> GraphModule:
    # 内部帮助函数，融合模块以为量化做准备
    r""" Internal helper function to fuse modules in preparation for quantization

    Args:
        model: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(model)  # 确保输入是 GraphModule 对象
    return fuse(
        model, is_qat, fuse_custom_config, backend_config)  # 执行模块融合，准备量化，忽略类型检查警告

def _prepare_fx(
    model: torch.nn.Module,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
    is_qat: bool,
    example_inputs: Tuple[Any, ...],
    prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
    _equalization_config: Optional[Union[QConfigMapping, Dict[str, Any]]] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
    is_standalone_module: bool = False,
) -> GraphModule:
    r""" Internal helper function for prepare_fx
    Args:
      `model`, `qconfig_mapping`, `prepare_custom_config`, `_equalization_config`:
      see docs for :func:`~torch.ao.quantization.prepare_fx`
      `is_standalone_module`: a boolean flag indicates whether we are
      quantizing a standalone module or not, a standalone module
      is a submodule of the parent module that is not inlined in the
      forward graph of the parent module,
      the way we quantize standalone module is described in:
      :func:`~torch.ao.quantization._prepare_standalone_module_fx`
    """
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()  # 如果没有提供 prepare_custom_config，使用默认配置
    if _equalization_config is None:
        _equalization_config = QConfigMapping()  # 如果没有提供 _equalization_config，使用默认配置

    if isinstance(prepare_custom_config, dict):
        warnings.warn(
            "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a PrepareCustomConfig instead.",
            FutureWarning,
            stacklevel=3,
        )
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)  # 将字典形式的 prepare_custom_config 转换为 PrepareCustomConfig 对象

    _swap_ff_with_fxff(model)  # 将模型中的 FloatFunctional 替换为 FXFloatFunctional

    skipped_module_names, skipped_module_classes = \
        get_skipped_module_name_and_classes(prepare_custom_config, is_standalone_module)  # 获取需要跳过的模块名称和类
    preserved_attr_names = prepare_custom_config.preserved_attributes  # 获取需要保留的属性名称
    preserved_attrs = {attr: getattr(model, attr) for attr in preserved_attr_names if hasattr(model, attr)}  # 获取模型中需要保留的属性

    tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)  # 创建量化跟踪器对象
    graph_module = GraphModule(model, tracer.trace(model))  # 对模型进行符号化跟踪，创建 GraphModule 对象
    _attach_meta_to_node_if_not_exist(graph_module)  # 如果节点没有元数据，附加元数据

    fuse_custom_config = FuseCustomConfig().set_preserved_attributes(prepare_custom_config.preserved_attributes)  # 创建 FuseCustomConfig 对象，设置保留的属性
    graph_module = _fuse_fx(
        graph_module,
        is_qat,
        fuse_custom_config,
        backend_config)  # 融合图形模块，准备量化
    # 使用 prepare 函数对模型进行预处理，准备模型以进行量化或训练。
    prepared = prepare(
        graph_module,                    # 图形模块，表示待处理的模型结构
        qconfig_mapping,                 # 量化配置映射，指定量化的方式和参数
        is_qat,                          # 是否为量化感知训练模式的标志
        tracer.node_name_to_scope,       # 跟踪器中节点名称到作用域的映射，用于定位模型中各部分的位置
        example_inputs=example_inputs,   # 示例输入，用于模型分析和准备
        prepare_custom_config=prepare_custom_config,  # 自定义的准备配置，用于特定的模型处理需求
        _equalization_config=_equalization_config,    # 均衡化配置，用于量化均衡化处理
        backend_config=backend_config,   # 后端配置，指定模型的运行环境和后端设置
        is_standalone_module=is_standalone_module,   # 是否为独立模块的标志，用于模型处理的上下文判断
    )  # type: ignore[operator]  # 类型注解，忽略操作符类型的类型检查提示

    # 将保留的属性附加到预处理后的模型中，以确保这些属性在后续处理中保持不变
    attach_preserved_attrs_to_model(prepared, preserved_attrs)
    # 返回预处理后的模型对象
    return prepared
# [Internal use only] 准备一个独立的模块，使其可以在量化父模块时使用。
# standalone_module 表示它是父模块中未内联的子模块，
# 将作为一个单独的单元进行独立量化。

# 准备定制配置中指定的独立模块如何被观察的输入量化索引和输出量化索引。
# 返回值:
# * model(GraphModule): 准备好的独立模块。它在 model.meta 中具有以下属性:
#   - `standalone_module_input_quantized_idxs(List[Int])`: 预期要量化的图输入索引列表，与为独立模块提供的 input_quantized_idxs 配置相同。
#   - `standalone_module_output_quantized_idxs(List[Int])`: 被量化的图输出索引列表，与为独立模块提供的 output_quantized_idxs 配置相同。
def _prepare_standalone_module_fx(
    model: torch.nn.Module,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
    is_qat: bool,
    example_inputs: Tuple[Any, ...],
    prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    return _prepare_fx(
        model,
        qconfig_mapping,
        is_qat,
        example_inputs,
        prepare_custom_config,
        backend_config=backend_config,
        is_standalone_module=True,
    )


# 融合模块如 conv+bn, conv+bn+relu 等，模型必须处于评估模式。
# 融合规则在 torch.ao.quantization.fx.fusion_pattern.py 中定义。
#
# 参数:
# * `model` (torch.nn.Module): 一个 torch.nn.Module 模型
# * `fuse_custom_config` (FuseCustomConfig): fuse_fx 的自定义配置。详见:class:`~torch.ao.quantization.fx.custom_config.FuseCustomConfig`
#
# 示例:
# ```
# from torch.ao.quantization import fuse_fx
# m = Model().eval()
# m = fuse_fx(m)
# ```
def fuse_fx(
    model: torch.nn.Module,
    fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    torch._C._log_api_usage_once("quantization_api.quantize_fx.fuse_fx")
    preserved_attr_names = fuse_custom_config.preserved_attributes
    # 根据给定的模型和属性名列表，创建一个字典，包含模型中存在的指定属性及其对应的值
    preserved_attrs = {attr: getattr(model, attr) for attr in preserved_attr_names if hasattr(model, attr)}
    
    # 使用 Torch FX 对模型进行符号化追踪，生成表示模型计算图的图模块对象
    graph_module = torch.fx.symbolic_trace(model)
    
    # 如果图模块中的节点不存在元数据，则附加元数据到每个节点
    _attach_meta_to_node_if_not_exist(graph_module)
    
    # 对符号化追踪后的图模块应用自定义的融合策略和后端配置，生成优化后的图模块对象
    graph_module = _fuse_fx(graph_module, False, fuse_custom_config, backend_config)
    
    # 将之前保存的模型属性附加到优化后的图模块中
    attach_preserved_attrs_to_model(graph_module, preserved_attrs)
    
    # 返回最终优化后的图模块对象
    return graph_module
# 准备一个模型以进行后训练量化
def prepare_fx(
    model: torch.nn.Module,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
    example_inputs: Tuple[Any, ...],
    prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
    _equalization_config: Optional[Union[QConfigMapping, Dict[str, Any]]] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    r""" Prepare a model for post training quantization

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model
         模型本身，需要进行量化的模型

      * `qconfig_mapping` (QConfigMapping): QConfigMapping object to configure how a model is
         quantized, see :class:`~torch.ao.quantization.qconfig_mapping.QConfigMapping`
         for more details
         QConfigMapping对象，用于配置模型量化的方式和参数

      * `example_inputs` (Tuple[Any, ...]): Example inputs for forward function of the model,
         Tuple of positional args (keyword args can be passed as positional args as well)
         模型前向传播函数的示例输入，以元组形式给出（关键字参数也可以作为位置参数传递）

      * `prepare_custom_config` (PrepareCustomConfig): customization configuration for quantization tool.
          See :class:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig` for more details
          自定义配置，用于量化工具

      * `_equalization_config`: config for specifying how to perform equalization on the model
         用于指定如何在模型上执行均衡化的配置

      * `backend_config` (BackendConfig): config that specifies how operators are quantized
         in a backend, this includes how the operators are observed,
         supported fusion patterns, how quantize/dequantize ops are
         inserted, supported dtypes etc. See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details
         指定在后端如何量化运算符的配置

    Return:
      返回一个包含观察器（由qconfig_mapping配置）的GraphModule，准备好进行校准
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_fx")
    调用内部C++ API，记录一次API使用情况，用于量化API日志
    return _prepare_fx(
        model,
        qconfig_mapping,
        False,  # is_qat
        example_inputs,
        prepare_custom_config,
        _equalization_config,
        backend_config,
    )


# 准备一个模型以进行量化感知训练
def prepare_qat_fx(
    model: torch.nn.Module,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
    example_inputs: Tuple[Any, ...],
    prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    r""" Prepare a model for quantization aware training

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model
         模型本身，用于量化感知训练的模型

      * `qconfig_mapping` (QConfigMapping): see :func:`~torch.ao.quantization.prepare_fx`
         QConfigMapping对象，用于配置模型量化的方式和参数，参考prepare_fx函数的说明

      * `example_inputs` (Tuple[Any, ...]): see :func:`~torch.ao.quantization.prepare_fx`
         模型前向传播函数的示例输入，用于量化感知训练

      * `prepare_custom_config` (PrepareCustomConfig): see :func:`~torch.ao.quantization.prepare_fx`
          自定义配置，用于量化感知训练工具

      * `backend_config` (BackendConfig): see :func:`~torch.ao.quantization.prepare_fx`
         后端配置，指定后端如何量化运算符的配置

    Return:
      返回一个包含假量化模块（由qconfig_mapping和backend_config配置）的GraphModule，准备好进行量化感知训练
    """
    # 调用 PyTorch 内部函数，用于记录量化 API 使用情况，仅记录一次
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_qat_fx")
    # 调用 _prepare_fx 函数，准备量化感知训练（QAT）的效果
    return _prepare_fx(
        model,                  # 待量化的模型
        qconfig_mapping,        # 量化配置映射
        True,                   # 表示进行量化感知训练（QAT）
        example_inputs,         # 示例输入数据
        prepare_custom_config,  # 准备定制化配置的函数
        backend_config=backend_config,  # 后端配置参数
    )
# 定义一个函数 `_convert_fx`，用于将给定的图模块转换为量化后的图模块
# `graph_module: GraphModule` - 待转换的图模块对象
# `is_reference: bool` - 是否作为参考模块进行转换
# `convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None` - 转换的自定义配置，可以是配置对象、字典或空
# `is_standalone_module: bool = False` - 是否作为独立模块进行准备
# `_remove_qconfig: bool = True` - 是否移除量化配置
# `qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None` - 量化配置映射，可以是映射对象、字典或空
# `backend_config: Union[BackendConfig, Dict[str, Any], None] = None` - 后端配置，可以是后端配置对象、字典或空
# `is_decomposed: bool = False` - 是否分解模块
# 返回转换后的量化图模块对象
def _convert_fx(
    graph_module: GraphModule,
    is_reference: bool,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
    is_standalone_module: bool = False,
    _remove_qconfig: bool = True,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
    is_decomposed: bool = False,
) -> GraphModule:
    """ `is_standalone_module`: see docs in :func:`~torch.ao.quantization.prepare_standalone_module_fx`
    """
    # 如果未提供转换的自定义配置，则使用默认配置对象 ConvertCustomConfig
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()

    # 如果传入的转换自定义配置是字典，发出警告并将其转换为 ConvertCustomConfig 对象
    if isinstance(convert_custom_config, dict):
        warnings.warn(
            "Passing a convert_custom_config_dict to convert is deprecated and will not be supported "
            "in a future version. Please pass in a ConvertCustomConfig instead.",
            FutureWarning,
            stacklevel=3,
        )
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)

    # 检查 graph_module 是否为图模块类型
    _check_is_graph_module(graph_module)

    # 根据转换函数 convert 将图模块进行量化转换，返回量化后的模块对象 quantized
    quantized = convert(
        graph_module,
        is_reference,
        convert_custom_config,
        is_standalone_module,
        _remove_qconfig_flag=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        is_decomposed=is_decomposed,
    )

    # 将保留的属性附加到量化后的模块对象 quantized 上
    attach_preserved_attrs_to_model(quantized, preserved_attrs)

    # 返回量化后的图模块对象 quantized
    return quantized


# 定义函数 convert_fx，用于将经过校准或训练的模型转换为量化模型
# `graph_module: GraphModule` - 待转换的图模块对象
# `convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None` - 转换的自定义配置，可以是配置对象、字典或空
# `_remove_qconfig: bool = True` - 是否移除量化配置
# `qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None` - 量化配置映射，可以是映射对象、字典或空
# `backend_config: Union[BackendConfig, Dict[str, Any], None] = None` - 后端配置，可以是后端配置对象、字典或空
# 返回转换后的量化图模块对象
def convert_fx(
    graph_module: GraphModule,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
    _remove_qconfig: bool = True,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    r""" Convert a calibrated or trained model to a quantized model
    Args:
        * `graph_module` (torch.fx.GraphModule): 一个准备好并经过校准/训练的模型（GraphModule）

        * `convert_custom_config` (ConvertCustomConfig): 用于转换函数的自定义配置。
            详见:class:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig`了解更多细节

        * `_remove_qconfig` (bool): 在转换后是否移除模型中的qconfig属性的选项。

        * `qconfig_mapping` (QConfigMapping): 指定如何为量化配置模型的配置。

           键必须包含传递给`prepare_fx`或`prepare_qat_fx`的qconfig_mapping中的键，值可以与其相同，也可以为`None`。
           可以通过将附加键设置为`None`来指定额外的键，以跳过在模型中量化该键的步骤::

                qconfig_mapping = QConfigMapping
                    .set_global(qconfig_from_prepare)
                    .set_object_type(torch.nn.functional.add, None)  # 跳过对torch.nn.functional.add的量化
                    .set_object_type(torch.nn.functional.linear, qconfig_from_prepare)
                    .set_module_name("foo.bar", None)  # 跳过对模块"foo.bar"的量化

         * `backend_config` (BackendConfig): 后端的配置，描述后端如何对操作员进行量化，
            包括量化模式支持（静态/动态/仅权重），数据类型支持（quint8/qint8等），
            每个操作员的观察者放置和融合操作员。
            详见:class:`~torch.ao.quantization.backend_config.BackendConfig`了解更多细节

    Return:
        量化模型（torch.nn.Module）

    Example::

        # prepared_model: 经过prepare_fx/prepare_qat_fx和校准/训练后的模型
        # convert_fx将经过校准/训练的模型转换为量化模型，用于目标硬件，
        # 这包括首先将模型转换为参考量化模型，然后将参考量化模型降低到后端
        # 目前支持的后端是fbgemm (onednn)、qnnpack (xnnpack)，它们共享相同的量化操作员集，因此我们使用相同的降低过程
        #
        # backend_config定义了模型中加权模块（例如nn.Linear）的相应参考量化模块
        # TODO: 在我们拆分fbgemm和qnnpack的backend_config后添加backend_config
        # 例如 backend_config = get_default_backend_config("fbgemm")
        quantized_model = convert_fx(prepared_model)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")
    return _convert_fx(
        graph_module,
        is_reference=False,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )
# 将经过校准或训练的模型转换为参考量化模型，使用 FX 图模式量化
# 参考量化模型是由 FX 图模式量化提供的标准量化模型表示，可以进一步优化以在目标硬件上运行，如加速器
def convert_to_reference_fx(
    graph_module: GraphModule,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
    _remove_qconfig: bool = True,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    r""" Convert a calibrated or trained model to a reference quantized model,
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = convert_to_reference_fx(prepared_model)

    """
    # 记录 API 使用情况，仅记录一次
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_to_reference_fx")
    # 调用内部函数 _convert_fx 完成实际的转换操作，返回参考量化模型
    return _convert_fx(
        graph_module,
        is_reference=True,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=_remove_qconfig,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )

# 将经过校准或训练的模型转换为使用分解表示的参考量化模型
# 见 https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md 获取更多详细信息
def _convert_to_reference_decomposed_fx(
    graph_module: GraphModule,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
    qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    r""" Convert a calibrated or trained model to a reference quantized model, with
    decomposed representation for quantized Tensor
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    """
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators
    
    Note: this is not public API
    
    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)
    
        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.
    
        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.
    
        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.
    
         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.
    
    Return:
        A reference quantized model (GraphModule) with operators working with decomposed quantized Tensor
    
    Example::
    
        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        reference_quantized_model = _convert_to_reference_decomposed_fx(prepared_model)
    
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx._convert_to_reference_decomposed_fx")
    return _convert_fx(
        graph_module,
        is_reference=True,
        convert_custom_config=convert_custom_config,
        _remove_qconfig=False,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        is_decomposed=True,
    )
# [仅限内部使用] 将由 :func:`~torch.ao.quantization.prepare_standalone_module_fx` 生成的模型转换为量化模型
#
# 返回一个量化的独立模块，输入/输出是否量化由 prepare_custom_config 指定，
# 输入量化索引、输出量化索引，请参阅 prepare_fx 文档获取详细信息
def _convert_standalone_module_fx(
    graph_module: GraphModule,
    is_reference: bool = False,
    convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    # 调用 _convert_fx 函数，将 graph_module 转换为量化的独立模块
    return _convert_fx(
        graph_module,
        is_reference,
        convert_custom_config,
        is_standalone_module=True,
    )
```