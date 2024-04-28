# `.\transformers\onnx\convert.py`

```
# 导入警告模块
import warnings
# 从 inspect 模块中导入 signature 函数
from inspect import signature
# 从 itertools 模块中导入 chain 函数
from itertools import chain
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 typing 模块中导入 TYPE_CHECKING 类，Iterable、List、Tuple、Union 类
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union
# 从 numpy 模块中导入 np 别名
import numpy as np
# 从 packaging.version 模块中导入 Version、parse 函数
from packaging.version import Version, parse
# 从 ..tokenization_utils_base 模块中导入 PreTrainedTokenizerBase 类
from ..tokenization_utils_base import PreTrainedTokenizerBase
# 从 ..utils 模块中导入 TensorType、is_tf_available、is_torch_available、logging 函数
from ..utils import (
    TensorType,
    is_tf_available,
    is_torch_available,
    logging,
)
# 从 .config 模块中导入 OnnxConfig 类
from .config import OnnxConfig

# 如果 Torch 可用
if is_torch_available():
    # 从 ..modeling_utils 模块中导入 PreTrainedModel 类
    from ..modeling_utils import PreTrainedModel

# 如果 TensorFlow 可用
if is_tf_available():
    # 从 ..modeling_tf_utils 模块中导入 TFPreTrainedModel 类
    from ..modeling_tf_utils import TFPreTrainedModel

# 如果是类型检查
if TYPE_CHECKING:
    # 从 ..feature_extraction_utils 模块中导入 FeatureExtractionMixin 类
    from ..feature_extraction_utils import FeatureExtractionMixin
    # 从 ..processing_utils 模块中导入 ProcessorMixin 类
    from ..processing_utils import ProcessorMixin
    # 从 ..tokenization_utils 模块中导入 PreTrainedTokenizer 类
    from ..tokenization_utils import PreTrainedTokenizer

# 获取logger对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 这是支持某些 ONNX Runtime 功能所需的最小版本
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")

# 检查 ONNX Runtime 要求
def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        # 尝试导入 onnxruntime 模块
        import onnxruntime
        
        # 解析已安装的 onnxruntime 版本
        ort_version = parse(onnxruntime.__version__)

        # 我们需要至少 1.4.0 版本
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(
                f"We found an older version of onnxruntime ({onnxruntime.__version__}) "
                f"but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.\n"
                "Please update onnxruntime by running `pip install --upgrade onnxruntime`"
            )

    except ImportError:
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. "
            "Please install the onnxruntime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )

# 导出 PyTorch 模型
def export_pytorch(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: "PreTrainedModel",
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: "PreTrainedTokenizer" = None,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Export the PyTorch model to ONNX format
    """
    # 将 PyTorch 模型导出为 ONNX 中间表示（IR）

    Args:
        preprocessor: ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            用于对数据进行编码的预处理器。
        model ([`PreTrainedModel`]):
            要导出的模型。
        config ([`~onnx.config.OnnxConfig`]):
            与导出模型关联的 ONNX 配置。
        opset (`int`):
            要使用的 ONNX 操作集的版本。
        output (`Path`):
            存储导出的 ONNX 模型的目录。
        device (`str`, *optional*, defaults to `cpu`):
            将要导出 ONNX 模型的设备。可以是 `cpu` 或 `cuda`。

    Returns:
        `Tuple[List[str], List[str]]`: 包含模型输入顺序列表和来自 ONNX 配置的命名输入的元组。

    """

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer
    # 检查 model 是否是 PreTrainedModel 类的子类
    if issubclass(type(model), PreTrainedModel):
        # 导入 torch 库
        import torch
        # 导入 torch.onnx 模块的 export 函数，并将其命名为 onnx_export
        from torch.onnx import export as onnx_export

        # 打印 PyTorch 版本信息
        logger.info(f"Using framework PyTorch: {torch.__version__}")
        # 在不计算梯度的情况下，设置 model 的 config 属性为返回字典，并将 model 设置为评估模式
        with torch.no_grad():
            model.config.return_dict = True
            model.eval()

            # 检查是否需要覆盖特定配置项
            if config.values_override is not None:
                logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
                # 遍历需要覆盖的配置项，打印覆盖的配置键值对
                for override_config_key, override_config_value in config.values_override.items():
                    logger.info(f"\t- {override_config_key} -> {override_config_value}")
                    # 设置 model.config 对应的配置项为给定值
                    setattr(model.config, override_config_key, override_config_value)

            # 确保输入匹配
            # TODO: Check when exporting QA we provide "is_pair=True"
            # 生成模拟输入数据，指定框架为 PyTorch
            model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)
            # 将模型移动到指定设备
            device = torch.device(device)
            # 如果是 CUDA 设备且可用，则将模型移动到设备，并将输入数据也移动到设备
            if device.type == "cuda" and torch.cuda.is_available():
                model.to(device)
                model_inputs_device = {}
                for k, v in model_inputs.items():
                    if isinstance(v, Tuple):
                        model_inputs_device[k] = tuple(
                            x.to(device) if isinstance(x, torch.Tensor) else None for x in v
                        )
                    elif isinstance(v, List):
                        model_inputs_device[k] = [
                            tuple(x.to(device) if isinstance(x, torch.Tensor) else None for x in t) for t in v
                        ]
                    else:
                        model_inputs_device[k] = v.to(device)

                model_inputs = model_inputs_device

            # 确保模型和配置输入匹配，并获取匹配的输入列表
            inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
            # 获取输出的名称列表
            onnx_outputs = list(config.outputs.keys())

            # 如果输入不匹配，抛出数值错误
            if not inputs_match:
                raise ValueError("Model and config inputs doesn't match")

            # 应用配置的修补操作
            config.patch_ops()

            # 导出 ONNX 模型
            onnx_export(
                model,
                (model_inputs,),
                f=output.as_posix(),
                input_names=list(config.inputs.keys()),
                output_names=onnx_outputs,
                dynamic_axes=dict(chain(config.inputs.items(), config.outputs.items())),
                do_constant_folding=True,
                opset_version=opset,
            )

            # 恢复配置的修补操作
            config.restore_ops()

    # 返回匹配的输入和 ONNX 模型的输出列表
    return matched_inputs, onnx_outputs
) -> Tuple[List[str], List[str]]:
    """
    Export a TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`] or [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`TFPreTrainedModel`] or [`PreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    import onnx
    import tensorflow as tf
    import tf2onnx

    # 检查是否同时提供了 tokenizer 和 preprocessor
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and preprocessor to export the model.")
    # 如果提供了 tokenizer，则产生 FutureWarning
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    # 设置模型返回值为字典类型
    model.config.return_dict = True

    # 检查是否需要覆盖某些配置项
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    # 确保输入匹配
    model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.TENSORFLOW)
    inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
    onnx_outputs = list(config.outputs.keys())

    # 生成输入签名
    input_signature = [
        tf.TensorSpec([None] * tensor.ndim, dtype=tensor.dtype, name=key) for key, tensor in model_inputs.items()
    ]
    # 将 TensorFlow 模型转换为 ONNX 模型
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    # 保存 ONNX 模型
    onnx.save(onnx_model, output.as_posix())
    config.restore_ops()

    return matched_inputs, onnx_outputs
    device: str = "cpu",


    # 设备参数，指定为字符串类型，默认为 "cpu"
    device: str = "cpu",
def export_onnx_model(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: Optional[str] = "cpu"
) -> Tuple[List[str], List[str]]:
    """
    Export a Pytorch or TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """

    # Check if PyTorch or TensorFlow is available, raise ImportError if not
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are not installed. "
            "Please install torch or tensorflow first."
        )

    # Check if TF model export on CUDA device is supported, raise RuntimeError if not
    if is_tf_available() and isinstance(model, TFPreTrainedModel) and device == "cuda":
        raise RuntimeError("`tf2onnx` does not support export on CUDA device.")

    # Check if both tokenizer and preprocessor are provided, raise ValueError if so
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")

    # Warn user about using deprecated argument 'tokenizer' and suggest using 'preprocessor' instead
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummy inputs.")
        preprocessor = tokenizer

    # Check and log unsupported PyTorch version
    if is_torch_available():
        from ..utils import get_torch_version

        if not config.is_torch_support_available:
            logger.warning(
                f"Unsupported PyTorch version for this model. Minimum required is {config.torch_onnx_minimum_version},"
                f" got: {get_torch_version()}"
            )

    # Export PyTorch model if PyTorch is available and model is subclass of PreTrainedModel
    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        return export_pytorch(preprocessor, model, config, opset, output, tokenizer=tokenizer, device=device)
    # Export TensorFlow model if TensorFlow is available and model is subclass of TFPreTrainedModel
    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        return export_tensorflow(preprocessor, model, config, opset, output, tokenizer=tokenizer)

def validate_model_outputs(
    config: OnnxConfig,
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    # 定义参数 atol，表示用于比较浮点数相等的绝对容差
    atol: float,
    # 定义参数 tokenizer，默认为 None，类型为 "PreTrainedTokenizer"
    tokenizer: "PreTrainedTokenizer" = None,
    # 导入所需的库
    from onnxruntime import InferenceSession, SessionOptions

    # 打印信息，说明正在验证 ONNX 模型
    logger.info("Validating ONNX model...")

    # 检查 preprocessor 和 tokenizer 是否同时提供，如果是则抛出 ValueError
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to validate the model outputs.")
    
    # 如果 tokenizer 存在，则显示警告，并将 preprocessor 覆盖为 tokenizer
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummy inputs.")
        preprocessor = tokenizer

    # 生成 inputs，使用不同的 batch_size 和 seq_len 来进行测试动态输入形状
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model_inputs = config.generate_dummy_inputs(
            preprocessor,
            batch_size=config.default_fixed_batch + 1,
            seq_length=config.default_fixed_sequence + 1,
            framework=TensorType.PYTORCH,
        )
    else:
        reference_model_inputs = config.generate_dummy_inputs(
            preprocessor,
            batch_size=config.default_fixed_batch + 1,
            seq_length=config.default_fixed_sequence + 1,
            framework=TensorType.TENSORFLOW,
        )

    # 创建 ONNX Runtime 会话
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options, providers=["CPUExecutionProvider"])

    # 从 reference_model 计算输出
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model.to("cpu")
    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}

    # 将潜在的输出集合展平为一维结构
    for name, value in ref_outputs.items():
        # 将输出名称重写为“present”，因为它是用于 ONNX 输出的名称（"past_key_values" 用于 ONNX 输入）
        if name == "past_key_values":
            name = "present"
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # 从 reference_model_inputs 创建 onnxruntime 输入
    reference_model_inputs_onnxruntime = config.generate_dummy_inputs_onnxruntime(reference_model_inputs)

    # 将潜在的输入集合展平为一维结构
    onnx_inputs = {}
    # 遍历 reference_model_inputs_onnxruntime 字典中的每个键值对
    for name, value in reference_model_inputs_onnxruntime.items():
        # 如果值是列表或元组类型，则将其展平化
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            # 更新 onnx_inputs 字典，将展平后的数值添加进去
            onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
        else:
            # 如果值不是列表或元组类型，则将其转换成 numpy 格式并添加到 onnx_inputs 字典中
            onnx_inputs[name] = value.numpy()

    # 使用 ONNX 模型计算输出
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # 检查 onnx_outputs 的键集合是否是 ref_outputs_dict 的子集
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        # 输出警告信息并抛出数值错误
        logger.info(
            f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}"
        )
        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        # 输出信息，表明 ONNX 模型输出的键集合和参考模型的一致
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

    # 检查形状和数值是否匹配
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # 检查形状是否匹配
        if not ort_value.shape == ref_value.shape:
            logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(
                "Outputs shape doesn't match between reference model and ONNX exported model: "
                f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
            )
        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # 检查数值是否匹配
        if not np.allclose(ref_value, ort_value, atol=atol):
            bad_indices = np.logical_not(np.isclose(ref_value, ort_value, atol=atol))
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))} for "
                f"{ref_value[bad_indices]} vs {ort_value[bad_indices]}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")
def ensure_model_and_config_inputs_match(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """
    确保模型的输入与配置的输入匹配
    :param model_inputs: 模型的输入
    :param config_inputs: 配置的输入
    :return: 返回一个布尔值和一个有序的输入列表
    """
    # 如果 Torch 可用并且模型是 PreTrainedModel 的子类，则获取模型的前向参数
    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        forward_parameters = signature(model.forward).parameters
    else:
        # 否则获取 TensorFlow 模型的前向参数
        forward_parameters = signature(model.call).parameters
    model_inputs_set = set(model_inputs)

    # 如果配置的输入键数量多于模型的输入键数量，则视为匹配
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # 确保输入顺序匹配（非常重要）
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs
```