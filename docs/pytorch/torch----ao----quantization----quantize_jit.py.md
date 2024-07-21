# `.\pytorch\torch\ao\quantization\quantize_jit.py`

```
# 指定允许未类型化定义，适用于类型检查工具如 mypy
mypy: allow-untyped-defs

# 导入 torch 库
import torch
# 导入量化配置相关模块
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
# 导入内部模块，用于处理 JIT（即时编译）模块
from torch.jit._recursive import wrap_cpp_module

# 定义 __all__ 列表，列出可以导出的符号（一般用于 from ... import *）
__all__ = [
    "script_qconfig",
    "script_qconfig_dict",
    "fuse_conv_bn_jit",
    "prepare_jit",
    "prepare_dynamic_jit",
    "convert_jit",
    "convert_dynamic_jit",
    "quantize_jit",
    "quantize_dynamic_jit",
]

# 检查输入的 model 是否为 ScriptModule 类型，否则抛出 ValueError 异常
def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

# 检查输入的 script module 是否包含 forward 方法，否则抛出 ValueError 异常
def _check_forward_method(model):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')

# 根据传入的 qconfig 实例化激活和权重观察模块，并将它们脚本化
def script_qconfig(qconfig):
    r"""Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c)

# 对给定的 qconfig_dict 应用 script_qconfig 函数，返回脚本化的 qconfig 字典
def script_qconfig_dict(qconfig_dict):
    r"""Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}

# 融合 Convolution 和 Batch Normalization 模块（仅适用于评估模型）
def fuse_conv_bn_jit(model, inplace=False):
    r""" Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.fuse_conv_bn_jit")
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

# 准备 JIT 模型的预处理，包括融合和插入观察器
def _prepare_jit(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observers(model._c,
                                                  'forward',
                                                  scripted_qconfig_dict,
                                                  inplace,
                                                  quant_type)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

# 准备 JIT 模型在设备上的预处理，包括插入观察器
def _prepare_ondevice_jit(model, qconfig_dict, method_name='forward', inplace=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    # 使用 script_qconfig_dict 函数处理 qconfig_dict，并返回处理后的字典
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    # 获取模型中指定方法的计算图
    method_graph = model._c._get_method(method_name).graph
    # 在方法的计算图上应用 Torch JIT 内联优化
    torch._C._jit_pass_inline(method_graph)
    # 对模型应用融合卷积和批量归一化操作
    model = fuse_conv_bn_jit(model, inplace)
    # 在模型的底层 C++ 对象中插入用于设备端量化观察的方法
    model_c = torch._C._jit_pass_insert_observer_method_for_ondevice_ptq(model._c,
                                                                         method_name,
                                                                         scripted_qconfig_dict,
                                                                         inplace,
                                                                         quant_type)
    # 如果 inplace 标志为 True，则重构原模型
    if inplace:
        model._reconstruct(model_c)
    else:
        # 否则，将底层 C++ 模型对象封装为 Python 模块
        model = wrap_cpp_module(model_c)
    # 返回处理后的模型对象
    return model
# 根据给定的模型、量化配置字典和是否原地操作的标志，调用 _prepare_jit 函数进行静态量化准备
def prepare_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)

# 根据给定的模型、量化配置字典和是否原地操作的标志，调用 _prepare_jit 函数进行动态量化准备
def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_dynamic_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)

# 根据给定的模型、量化配置字典、方法名和是否原地操作的标志，调用 _prepare_ondevice_jit 函数进行动态量化准备
def _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name='forward', inplace=False):
    return _prepare_ondevice_jit(model, qconfig_dict, method_name, inplace, quant_type=QuantType.DYNAMIC)

# 对模型进行静态量化转换，支持原地操作和调试模式，可选择保留的属性
def _convert_jit(model, inplace=False, debug=False, quant_type=QuantType.STATIC,
                 preserved_attrs=None):
    _check_is_script_module(model)  # 检查模型是否为脚本模块
    model.eval()  # 将模型设置为评估模式
    model_c = model._c  # 获取模型的 C++ 后端对象
    # 在模型的前向方法中插入量化和反量化操作
    model_c = torch._C._jit_pass_insert_quant_dequant(model_c, 'forward', inplace, debug, quant_type)
    if not debug:
        # 检查模型参数是否都在 XPU 设备上，若不是，则移动到 CPU
        is_xpu = all(p.device.type == 'xpu' for p in model.parameters())
        if not is_xpu:
            model.cpu()  # 将模型转移到 CPU
        if preserved_attrs is None:
            preserved_attrs = []
        # 最终的量化操作，设定量化类型和保留的属性
        model_c = torch._C._jit_pass_quant_finalize(model_c, quant_type, preserved_attrs)
    if inplace:
        model._reconstruct(model_c)  # 如果是原地操作，重建模型
    else:
        model = wrap_cpp_module(model_c)  # 否则，将 C++ 模块包装为 Python 模型
    # 对模型的计算图进行常量传播
    torch._C._jit_pass_constant_propagation(model.graph)
    # 对模型的计算图进行死代码消除
    torch._C._jit_pass_dce(model.graph)
    return model  # 返回量化后的模型

# 对模型进行动态量化转换，支持原地操作和调试模式
def _convert_ondevice_jit(model, method_name, inplace=False, debug=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)  # 检查模型是否为脚本模块
    assert quant_type == QuantType.DYNAMIC, "This API, while should work for static quant, is only tested for dynamic quant."
    assert not method_name.startswith("observe_"), "Pass in valid method to be quantized, e.g. forward"
    observe_method_name = "observe_" + method_name
    quantize_method_name = "quantize_" + method_name
    model_c = model._c  # 获取模型的 C++ 后端对象
    # 在模型的指定方法中插入量化和反量化操作，适用于设备上的动态量化
    model_c = torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq(
        model._c, observe_method_name, inplace, debug, QuantType.DYNAMIC)
    # 最终的量化操作，设定量化类型和量化后的方法名
    model_c = torch._C._jit_pass_quant_finalize_for_ondevice_ptq(model_c, QuantType.DYNAMIC, quantize_method_name)
    if inplace:
        model._reconstruct(model_c)  # 如果是原地操作，重建模型
    else:
        model = wrap_cpp_module(model_c)  # 否则，将 C++ 模块包装为 Python 模型
    return model  # 返回量化后的模型

# 对模型进行静态量化转换，支持原地操作和调试模式，可选择保留的属性
def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_jit")
    return _convert_jit(model, inplace, debug, quant_type=QuantType.STATIC, preserved_attrs=preserved_attrs)

# 对模型进行动态量化转换，支持原地操作和调试模式，可选择保留的属性
def convert_dynamic_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_dynamic_jit")
    # 调用一个名为 _convert_jit 的函数，将给定的参数传递给它
    # 这个函数的作用需要根据上下文来理解，可能是将模型转换为 JIT（即时编译）格式的函数
    # 参数包括 model（模型对象）、inplace（是否原地操作）、debug（是否启用调试模式）、quant_type（量化类型为动态）、preserved_attrs（需要保留的属性列表）
    return _convert_jit(model, inplace, debug, quant_type=QuantType.DYNAMIC, preserved_attrs=preserved_attrs)
# 使用动态量化类型将模型转换为 JIT 脚本格式
def _convert_ondevice_dynamic_jit(model, method_name, inplace=False, debug=False):
    return _convert_ondevice_jit(model, method_name, inplace, debug, quant_type=QuantType.DYNAMIC)


# 在设备上进行动态 JIT 脚本量化的实现，准备模型并进行转换
def _quantize_ondevice_dynamic_jit_impl(model, qconfig_dict, method_name, inplace=False):
    model = _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name, inplace)
    model = _convert_ondevice_dynamic_jit(model, method_name, inplace)
    return model

# 对模型进行 JIT 脚本量化，支持静态和动态两种量化类型
def _quantize_jit(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, quant_type=QuantType.STATIC):
    # 总是进行原地转换，因为当 inplace=False 时，张量在 prepare_jit 中已被复制
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        assert run_fn, "Must provide calibration function for post training static quantization"
        assert run_args, "Must provide calibration dataset for post training static quantization"
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)

    # 执行常量传播和死代码消除的 JIT 优化过程
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model

# 对输入的浮点 TorchScript 模型进行后训练静态量化
def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it's overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit
    # 使用 torch.jit.script 将 float_model 转换为 Torch 脚本模型，并设为评估模式
    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    # 获取默认的量化配置，这里使用 fbgemm 作为后端
    qconfig = get_default_qconfig('fbgemm')

    # 定义一个用于模型量化校准的函数 calibrate
    def calibrate(model, data_loader):
        # 将模型设为评估模式
        model.eval()
        with torch.no_grad():
            # 遍历数据加载器中的图像和目标数据
            for image, target in data_loader:
                # 对每个图像调用模型进行推断，但不返回任何结果
                model(image)

    # 使用 quantize_jit 函数对 ts_model 进行量化
    quantized_model = quantize_jit(
        ts_model,              # 待量化的 Torch 脚本模型
        {'': qconfig},         # 使用指定的量化配置字典
        calibrate,             # 量化校准函数
        [data_loader_test]     # 传入量化校准函数的参数列表，这里是包含测试数据的数据加载器
    )
# 使用 TorchScript 对输入的浮点模型进行后训练动态量化
def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: 输入的浮点 TorchScript 模型
        `qconfig_dict`: qconfig_dict 是一个字典，以子模块名称为键，
                        对应的 qconfig 为值，请参见详细描述：:func:`~torch.ao.quantization.quantize_jit`
        `inplace`: 是否原地进行模型转换，即修改原始模块
        `debug`: 是否生成适合调试的模型（保留权重属性）

    Return:
        Quantized TorchScript 模型.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(float_model.eval())  # 或者 torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_dynamic_jit(
        ts_model,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_dynamic_jit")
    return _quantize_jit(model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC)


# 为输入的浮点 TorchScript 模型准备*在设备上*的后训练动态量化
def _quantize_ondevice_dynamic_jit(model, qconfig_dict, method_name='forward', inplace=False):
    r"""Prepares the input float TorchScript model with
    *on-device* post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: 输入的浮点 TorchScript 模型
        `qconfig_dict`: qconfig_dict 是一个字典，以子模块名称为键，
                        对应的 qconfig 为值，请参见详细描述：:func:`~torch.ao.quantization.quantize_jit`
        `method_name`: 模型内的方法名称，准备进行量化
        `inplace`: 是否原地进行模型转换，即修改原始模块
    # 返回经过设备端量化准备的TorchScript模型。
    # 这意味着返回的模型具有以下特征：
    # - 方法被内联。
    # - 模型中插入了观察器模块。
    # - 模型中插入了打包参数，但它们是空的，即不包含有效的量化权重。
    # - 添加了 observe_<method_name> 方法，用于观察待量化的值。
    # - 添加了 reset_observers_<method_name> 方法，用于重置观察器。
    # - 添加了 quantize_<method_name> 方法到模型中：
    #   - 该方法提取尺度（scale）、零点（zero points）。
    #   - 对观察到的权重进行量化。
    #   - 创建打包参数，并更新模型属性以使用新的打包参数值。
    #   - 使用 SetAttr 方法将原始的 FP32 权重重置为空张量。
    # - 添加了 quantized_<method_name> 方法到模型中：
    #   - 该方法使用量化权重和量化的线性操作替代 FP32 操作。
    #   - 该方法应该在 PTQ（Post Training Quantization，训练后量化）后用于推断。
    # - 注意，所有方法的签名应与 method_name 相同。
    
    # 在设备上的后续步骤：
    # - 运行 reset_observers_<method_name>
    # - 运行 observe_<method_name>
    # - 运行 quantize_<method_name>
    # - 现在模型可以保存并在之后加载。
    # - 使用 quantized_<method_name> 方法运行模型。
    
    # 示例：
    # ```python
    # import torch
    # from torch.ao.quantization import per_channel_dynamic_qconfig
    # from torch.ao.quantization.quantize_jit import _quantize_ondevice_dynamic_jit
    
    # ts_model = torch.jit.script(float_model.eval())  # 或者 torch.jit.trace(float_model, input)
    # qconfig = get_default_qconfig('fbgemm')
    # quant_ready_model = _quantize_ondevice_dynamic_jit(
    #     ts_model,
    #     {'': qconfig},
    #     'forward',
    #     True)
    # ```
    def _quantize_ondevice_dynamic_jit(model, qconfig_dict, method_name, inplace=inplace):
        # 调用内部实现函数，执行设备端动态量化的具体操作
        return _quantize_ondevice_dynamic_jit_impl(model, qconfig_dict, method_name, inplace=inplace)
```