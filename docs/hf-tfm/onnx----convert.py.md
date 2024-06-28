# `.\onnx\convert.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 从inspect模块中导入signature函数，用于获取函数签名信息
from inspect import signature
# 从itertools模块中导入chain函数，用于扁平化多个可迭代对象
from itertools import chain
# 从pathlib模块中导入Path类，用于处理文件路径
from pathlib import Path
# 从typing模块中导入必要的类型提示
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

# 导入numpy库，通常用于数值计算
import numpy as np
# 从packaging.version模块中导入Version和parse函数，用于处理版本号信息
from packaging.version import Version, parse

# 从上级目录中导入tokenization_utils_base模块中的PreTrainedTokenizerBase类
from ..tokenization_utils_base import PreTrainedTokenizerBase
# 从上级目录中导入utils模块中的TensorType、is_tf_available、is_torch_available和logging函数
from ..utils import (
    TensorType,
    is_tf_available,
    is_torch_available,
    logging,
)
# 从当前目录中导入config模块中的OnnxConfig类
from .config import OnnxConfig

# 如果torch可用，则从..modeling_utils模块中导入PreTrainedModel类
if is_torch_available():
    from ..modeling_utils import PreTrainedModel

# 如果tensorflow可用，则从..modeling_tf_utils模块中导入TFPreTrainedModel类
if is_tf_available():
    from ..modeling_tf_utils import TFPreTrainedModel

# 如果当前是类型检查状态，则从..feature_extraction_utils和..processing_utils模块中导入相应类
if TYPE_CHECKING:
    from ..feature_extraction_utils import FeatureExtractionMixin
    from ..processing_utils import ProcessorMixin
    from ..tokenization_utils import PreTrainedTokenizer

# 从logging模块中获取名为__name__的logger对象，并赋值给logger变量
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义最低要求的ONNX Runtime版本号
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")


def check_onnxruntime_requirements(minimum_version: Version):
    """
    检查是否安装了ONNX Runtime，并且安装的版本是否足够新

    Raises:
        ImportError: 如果未安装ONNX Runtime或版本太旧
    """
    try:
        # 尝试导入onnxruntime模块
        import onnxruntime

        # 解析已安装onnxruntime的版本号
        ort_version = parse(onnxruntime.__version__)

        # 要求至少是1.4.0版本
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
    导出PyTorch模型至ONNX格式

    Args:
        preprocessor (Union[PreTrainedTokenizer, FeatureExtractionMixin, ProcessorMixin]):
            预处理器对象，可能是PreTrainedTokenizer、FeatureExtractionMixin或ProcessorMixin的子类实例
        model (PreTrainedModel): 预训练模型对象，是PreTrainedModel的子类实例
        config (OnnxConfig): ONNX导出配置对象，是OnnxConfig类的实例
        opset (int): ONNX操作集版本号
        output (Path): 导出的ONNX模型路径
        tokenizer (PreTrainedTokenizer, optional):
            如果模型需要tokenizer，此处提供其对象，可能是PreTrainedTokenizer的子类实例. Defaults to None.
        device (str, optional): 设备类型，例如'cpu'或'cuda'. Defaults to "cpu".

    Returns:
        Tuple[List[str], List[str]]: 返回两个字符串列表，分别表示成功和失败的导出步骤

    """
    # 检查预处理器的类型是否为 `PreTrainedTokenizerBase`，并且确保没有同时提供 tokenizer 参数
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        # 如果同时提供了 tokenizer 和 preprocessor，则抛出数值错误异常
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")
    
    # 如果提供了 tokenizer 参数，则发出警告信息，表示在未来版本中将移除 tokenizer 参数，建议使用 preprocessor 参数代替
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        # 记录日志信息，指示将 preprocessor 参数重写为 tokenizer 参数，用于生成虚拟输入
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummy inputs.")
        preprocessor = tokenizer
    # 检查模型是否是 PreTrainedModel 的子类
    if issubclass(type(model), PreTrainedModel):
        import torch
        from torch.onnx import export as onnx_export

        # 输出使用的 PyTorch 框架版本信息
        logger.info(f"Using framework PyTorch: {torch.__version__}")
        
        # 禁止梯度计算，并设置模型返回字典形式的输出
        with torch.no_grad():
            model.config.return_dict = True
            # 将模型设置为评估模式
            model.eval()

            # 检查是否需要覆盖某些配置项
            if config.values_override is not None:
                logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
                # 遍历并覆盖配置项
                for override_config_key, override_config_value in config.values_override.items():
                    logger.info(f"\t- {override_config_key} -> {override_config_value}")
                    setattr(model.config, override_config_key, override_config_value)

            # 确保输入数据与模型要求匹配
            # TODO: 在导出 QA 模型时，需要确认是否提供了 "is_pair=True"
            model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)
            # 设置设备类型并将模型移动到相应设备
            device = torch.device(device)
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

            # 确保模型输入与配置输入匹配
            inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
            # 获取配置中的输出项列表
            onnx_outputs = list(config.outputs.keys())

            # 如果模型和配置的输入不匹配，则抛出数值错误
            if not inputs_match:
                raise ValueError("Model and config inputs doesn't match")

            # 应用配置的操作补丁
            config.patch_ops()

            # 导出模型到 ONNX 格式
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

            # 恢复配置的操作
            config.restore_ops()

    # 返回匹配的输入和 ONNX 输出列表
    return matched_inputs, onnx_outputs
# 将 TensorFlow 模型导出为 ONNX 中间表示（IR）

def export_tensorflow(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin"],
    model: "TFPreTrainedModel",
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: "PreTrainedTokenizer" = None,
) -> Tuple[List[str], List[str]]:
    """
    Args:
        preprocessor: ([`PreTrainedTokenizer`] or [`FeatureExtractionMixin`]):
            用于对数据进行编码的预处理器。
        model ([`TFPreTrainedModel`]):
            要导出的模型。
        config ([`~onnx.config.OnnxConfig`]):
            导出模型相关的 ONNX 配置。
        opset (`int`):
            要使用的 ONNX 操作集的版本。
        output (`Path`):
            存储导出的 ONNX 模型的目录。

    Returns:
        `Tuple[List[str], List[str]]`: 包含模型输入顺序列表和来自 ONNX 配置的命名输入的元组。
    """
    import onnx
    import tensorflow as tf
    import tf2onnx

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and preprocessor to export the model.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    # 设置模型配置以返回字典形式的输出
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

    # 创建 TensorFlow 的输入签名
    input_signature = [
        tf.TensorSpec([None] * tensor.ndim, dtype=tensor.dtype, name=key) for key, tensor in model_inputs.items()
    ]

    # 将 Keras 模型转换为 ONNX 模型
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    # 将 ONNX 模型保存到文件
    onnx.save(onnx_model, output.as_posix())
    # 恢复操作
    config.restore_ops()

    # 返回匹配的输入和 ONNX 输出
    return matched_inputs, onnx_outputs
    device: str = "cpu",
def export_to_onnx(
    preprocessor: Union['PreTrainedTokenizer', 'FeatureExtractionMixin', 'ProcessorMixin'],
    model: Union['PreTrainedModel', 'TFPreTrainedModel'],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = 'cpu'
) -> Tuple[List[str], List[str]]:
    """
    Export a Pytorch or TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor (Union['PreTrainedTokenizer', 'FeatureExtractionMixin', 'ProcessorMixin']):
            The preprocessor used for encoding the data.
        model (Union['PreTrainedModel', 'TFPreTrainedModel']):
            The model to export.
        config (OnnxConfig):
            The ONNX configuration associated with the exported model.
        opset (int):
            The version of the ONNX operator set to use.
        output (Path):
            Directory to store the exported ONNX model.
        device (str, optional, defaults to 'cpu'):
            The device on which the ONNX model will be exported. Either 'cpu' or 'cuda'. Only PyTorch is supported for
            export on CUDA devices.

    Returns:
        Tuple[List[str], List[str]]: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """

    # Check if either PyTorch or TensorFlow is available; raise ImportError if not
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are installed. "
            "Please install torch or tensorflow first."
        )

    # Raise RuntimeError if trying to export a TensorFlow model on CUDA device
    if is_tf_available() and isinstance(model, TFPreTrainedModel) and device == "cuda":
        raise RuntimeError("`tf2onnx` does not support export on CUDA device.")

    # Raise ValueError if both a tokenizer and a preprocessor are provided
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")

    # Warn and log if 'tokenizer' argument is used; it's deprecated
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummy inputs.")
        preprocessor = tokenizer

    # Check PyTorch version compatibility
    if is_torch_available():
        from ..utils import get_torch_version

        if not config.is_torch_support_available:
            logger.warning(
                f"Unsupported PyTorch version for this model. Minimum required is {config.torch_onnx_minimum_version},"
                f" got: {get_torch_version()}"
            )

    # Export using PyTorch if available and model is a subclass of PreTrainedModel
    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        return export_pytorch(preprocessor, model, config, opset, output, tokenizer=tokenizer, device=device)
    # Export using TensorFlow if available and model is a subclass of TFPreTrainedModel
    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        return export_tensorflow(preprocessor, model, config, opset, output, tokenizer=tokenizer)
    atol: float,
    tokenizer: "PreTrainedTokenizer" = None,


    # 定义一个名为 atol 的参数，类型为 float，表示绝对误差容限
    atol: float,
    # 定义一个名为 tokenizer 的参数，默认为 None，类型为 "PreTrainedTokenizer"，表示一个预训练的分词器对象
    tokenizer: "PreTrainedTokenizer" = None,
    # 导入所需的模块和类
    from onnxruntime import InferenceSession, SessionOptions

    # 输出信息，验证 ONNX 模型的有效性
    logger.info("Validating ONNX model...")

    # 如果 preprocessor 是 PreTrainedTokenizerBase 的实例且 tokenizer 不为空，则抛出异常
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to validate the model outputs.")
    
    # 如果存在 tokenizer 参数，则发出警告，并用 tokenizer 覆盖 preprocessor 参数以生成虚拟输入
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummy inputs.")
        preprocessor = tokenizer

    # 生成具有不同 batch_size 和 seq_len 的输入，用于测试动态输入形状
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

    # 如果是 PyTorch 可用且 reference_model 是 PreTrainedModel 的子类，则将 reference_model 移到 CPU 上
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model.to("cpu")

    # 使用 reference_model_inputs 计算 reference_model 的输出
    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}

    # 将可能的输出集合（如 past_keys）展平为一个平面结构
    for name, value in ref_outputs.items():
        # 将输出名称重写为 "present"，因为这是用于 ONNX 输出的名称（"past_key_values" 用于 ONNX 输入）
        if name == "past_key_values":
            name = "present"
        # 如果值是列表或元组，则通过 config.flatten_output_collection_property 展平并更新 ref_outputs_dict
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # 根据 reference_model_inputs 创建 onnxruntime 输入
    reference_model_inputs_onnxruntime = config.generate_dummy_inputs_onnxruntime(reference_model_inputs)

    # 将可能的输入集合（如 past_keys）展平为一个平面结构
    onnx_inputs = {}
    # 遍历reference_model`
    # Iterate over each name-value pair in reference_model_inputs_onnxruntime dictionary
    for name, value in reference_model_inputs_onnxruntime.items():
        # Check if the value is a list or tuple
        if isinstance(value, (list, tuple)):
            # Flatten the output collection property using config.flatten_output_collection_property method
            value = config.flatten_output_collection_property(name, value)
            # Update onnx_inputs dictionary with flattened values converted to numpy arrays
            onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
        else:
            # Convert value to numpy array and assign to onnx_inputs dictionary
            onnx_inputs[name] = value.numpy()

    # Compute outputs from the ONNX model using session.run
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Check if the set of keys in onnx_outputs is a subset of keys in ref_outputs_dict
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        # Log mismatched output names if sets do not match
        logger.info(
            f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}"
        )
        # Raise ValueError if output names do not match
        raise ValueError(
            "Outputs don't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        # Log matching output names if sets match
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

    # Validate shape and values of ONNX model outputs against reference model
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        # Determine reference value based on framework availability and model type
        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # Check if shapes match
        if not ort_value.shape == ref_value.shape:
            logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            # Raise ValueError if shapes do not match
            raise ValueError(
                "Outputs shape doesn't match between reference model and ONNX exported model: "
                f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
            )
        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # Check if values are close within specified tolerance
        if not np.allclose(ref_value, ort_value, atol=atol):
            bad_indices = np.logical_not(np.isclose(ref_value, ort_value, atol=atol))
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            # Raise ValueError if values are not sufficiently close
            raise ValueError(
                "Outputs values don't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))} for "
                f"{ref_value[bad_indices]} vs {ort_value[bad_indices]}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")
# 确保模型输入和配置输入匹配的函数
def ensure_model_and_config_inputs_match(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """
    确保模型输入和配置输入匹配的函数。

    :param model: 预训练模型对象，可以是 `PreTrainedModel` 或 `TFPreTrainedModel` 的子类之一
    :param model_inputs: 模型期望的输入参数的可迭代对象，通常是字符串列表
    :return: 返回一个元组，包含一个布尔值和一个字符串列表。布尔值表示模型输入是否与配置输入匹配，字符串列表表示匹配的输入参数的有序列表。
    """

    # 如果当前环境支持 PyTorch 并且 model 是 PreTrainedModel 的子类
    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        # 获取模型的 forward 方法的参数签名
        forward_parameters = signature(model.forward).parameters
    else:
        # 否则获取模型的 call 方法的参数签名（通常是 TensorFlow 模型）
        forward_parameters = signature(model.call).parameters

    # 将模型期望的输入参数转换为集合
    model_inputs_set = set(model_inputs)

    # 获取模型 forward 方法的参数名称集合
    forward_inputs_set = set(forward_parameters.keys())

    # 检查模型期望的输入参数是否都在 forward 方法的参数中
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # 确保输入参数的顺序匹配（非常重要！！！）
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]

    # 返回匹配结果和有序的输入参数列表
    return is_ok, ordered_inputs
```