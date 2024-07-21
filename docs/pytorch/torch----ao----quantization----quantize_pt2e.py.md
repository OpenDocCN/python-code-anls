# `.\pytorch\torch\ao\quantization\quantize_pt2e.py`

```
# 导入PyTorch库中需要的模块
import torch
# 从torch.fx模块中导入GraphModule和Node类
from torch.fx import GraphModule
from torch.fx import Node

# 从当前目录下的pt2e.prepare模块中导入prepare函数
from .pt2e.prepare import prepare
# 从当前目录下的pt2e.qat_utils模块中导入_fuse_conv_bn_qat和_fold_conv_bn_qat函数
from .pt2e.qat_utils import (
    _fuse_conv_bn_qat,
    _fold_conv_bn_qat,
)
# 从当前目录下的pt2e.utils模块中导入_get_node_name_to_scope、_fuse_conv_bn_和_disallow_eval_train函数
from .pt2e.utils import (
    _get_node_name_to_scope,
    _fuse_conv_bn_,
    _disallow_eval_train,
)
# 从当前目录下的pt2e.representation模块中导入reference_representation_rewrite函数
from .pt2e.representation import reference_representation_rewrite
# 从当前目录下的quantize_fx模块中导入_convert_to_reference_decomposed_fx函数
from .quantize_fx import _convert_to_reference_decomposed_fx
# 从torch.ao.quantization.quantizer模块中导入多个类和函数
from torch.ao.quantization.quantizer import (
    Quantizer,
    QuantizationSpecBase,
    QuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
    DerivedQuantizationSpec,
    QuantizationAnnotation,
)
# 从torch.fx.passes.infra.pass_manager模块中导入PassManager类
from torch.fx.passes.infra.pass_manager import PassManager
# 从torch.ao.quantization.pt2e.duplicate_dq_pass模块中导入DuplicateDQPass类
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
# 从torch.ao.quantization.pt2e.port_metadata_pass模块中导入PortNodeMetaForQDQ类
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
# 从torch._export.passes.constant_folding模块中导入constant_fold函数
from torch._export.passes.constant_folding import constant_fold

# 定义公开的模块成员列表，用于模块导入时的显示
__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]


def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for post training quantization

    Args:
      * `model` (torch.fx.GraphModule): a model captured by `torch.export` API
        in the short term we are using `torch._export.capture_pre_autograd_graph`,
        in the long term we'll migrate to some `torch.export` API
      * `quantizer`: A backend specific quantizer that conveys how user want the
        model to be quantized. Tutorial for how to write a quantizer can be found here:
        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html

    Return:
      A GraphModule with observer (based on quantizer annotation), ready for calibration
    """
    # 准备模型以进行后训练量化的预处理
    # 返回一个带有观察者的GraphModule（基于量化器的注释），准备进行校准
    pass
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_pt2e")
    # 记录一次 API 使用，这里是用于量化准备的 API

    original_graph_meta = model.meta
    # 保存原始模型的元数据信息

    node_name_to_scope = _get_node_name_to_scope(model)
    # 获取模型中节点名称到作用域的映射

    # TODO: 检查 qconfig_mapping 确保在融合之前卷积和批归一化都配置为量化
    # TODO: (或许) 可以使用子图重写器重写这部分逻辑
    # 执行卷积和批归一化融合操作
    _fuse_conv_bn_(model)

    # 使用量化器进行模型的注释和转换
    quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)

    # 对模型进行准备，设置节点名称到作用域的映射，不处于量化训练状态
    model = prepare(model, node_name_to_scope, is_qat=False)

    # 恢复原始模型的元数据信息
    model.meta.update(original_graph_meta)

    # 禁止模型处于评估或训练状态
    model = _disallow_eval_train(model)

    # 返回经过量化准备后的模型
    return model
def prepare_qat_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for quantization aware training

    Args:
      * `model` (torch.fx.GraphModule): see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`
      * `quantizer`: see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`

    Return:
      A GraphModule with fake quant modules (based on quantizer annotation), ready for
      quantization aware training

    Example::
        import torch
        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
        from torch._export import capture_pre_autograd_graph
        from torch.ao.quantization.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result shoud mostly stay the same
        m = capture_pre_autograd_graph(m, *example_inputs)
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_qat_pt2e(m, quantizer)

        # run quantization aware training
        train_loop(prepared_model, train_loop)

    """
    # 记录 API 使用情况
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_qat_pt2e")
    # 保存原始图的元数据
    original_graph_meta = model.meta
    # 获取节点名称到作用域的映射
    node_name_to_scope = _get_node_name_to_scope(model)
    # 对模型进行注释转换
    quantizer.transform_for_annotation(model)
    # 对模型进行注释
    quantizer.annotate(model)
    # 验证模型
    quantizer.validate(model)
    # 在注释后执行融合以避免在新子图中量化不需要量化的操作
    # TODO: 只有在卷积和批归一化都配置为量化时才进行融合
    _fuse_conv_bn_qat(model)
    # 准备模型，用于量化感知训练
    model = prepare(model, node_name_to_scope, is_qat=True)
    # 更新模型的元数据
    model.meta.update(original_graph_meta)
    # 禁止模型处于评估或训练状态
    model = _disallow_eval_train(model)
    # 返回准备好的模型
    return model

_QUANT_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]
def _quant_node_constraint(n: Node) -> bool:
    """
    如果在 get_attr 和 quantize 操作之间存在纯操作，它们将被常量传播。
    例如：get_attr(weight) -> transpose -> quantize -> dequantize*
    （注意：dequantize 操作不会被常量传播）

    这个过滤器被添加是因为我们不希望对与量化无关的事物进行常量折叠。
    """
    # 检查节点 n 是否为函数调用，并且目标函数在 _QUANT_OPS 中
    return n.op == "call_function" and n.target in _QUANT_OPS
def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
    fold_quantize: bool = True,
) -> GraphModule:
    """Convert a calibrated/trained model to a quantized model

    Args:
      * `model` (torch.fx.GraphModule): calibrated/trained model
      * `use_reference_representation` (bool): boolean flag to indicate whether to produce reference representation or not
      * `fold_quantize` (bool): boolean flag for whether fold the quantize op or not

    Returns:
        quantized model, either in q/dq representation or reference representation

    Example::

        # prepared_model: the model produced by `prepare_pt2e`/`prepare_qat_pt2e` and calibration/training
        # `convert_pt2e` produces a quantized model that represents quantized computation with
        # quantize dequantize ops and fp32 ops by default.
        # Please refer to
        # https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html#convert-the-calibrated-model-to-a-quantized-model
        # for detailed explanation of output quantized model
        quantized_model = convert_pt2e(prepared_model)

    """  # flake8: noqa

    # 记录一次 API 使用，这里是量化 API 的一部分
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.convert_pt2e")

    # 检查 use_reference_representation 是否为布尔类型，否则抛出异常
    if not isinstance(use_reference_representation, bool):
        raise ValueError(
            "Unexpected argument type for `use_reference_representation`, "
            f"please make sure you intend to pass argument {use_reference_representation} to convert_pt2e")

    # 保存原始模型的元数据
    original_graph_meta = model.meta

    # 将模型转换为参考分解的 FX 格式
    model = _convert_to_reference_decomposed_fx(model)

    # 对模型进行卷积-批量归一化量化训练（QAT）折叠
    model = _fold_conv_bn_qat(model)

    # 创建 PassManager 对象并应用 `DuplicateDQPass` 传递
    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module

    # 创建 PassManager 对象并应用 `PortNodeMetaForQDQ` 传递
    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    # 如果指定了 fold_quantize 参数，则对模型进行常量折叠
    if fold_quantize:
        constant_fold(model, _quant_node_constraint)

    # 如果指定了 use_reference_representation 参数，则将模型重写为参考表示形式
    if use_reference_representation:
        model = reference_representation_rewrite(model)

    # 恢复原始模型的元数据
    model.meta.update(original_graph_meta)

    # 禁止评估训练状态的模型
    model = _disallow_eval_train(model)

    # 返回量化后的模型
    return model
```