# `.\transformers\convert_graph_to_onnx.py`

```
# 导入必要的库
import warnings
from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from packaging.version import Version, parse

# 导入 transformers 库中的相关模块和函数
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import ModelOutput, is_tf_available, is_torch_available

# 定义最低要求的 ONNX Runtime 版本
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")

# 支持的 pipeline 列表
SUPPORTED_PIPELINES = [
    "feature-extraction",
    "ner",
    "sentiment-analysis",
    "fill-mask",
    "question-answering",
    "text-generation",
    "translation_en_to_fr",
    "translation_en_to_de",
    "translation_en_to_ro",
]

# 定义一个自定义的参数解析器类，用于导出 transformers 模型到 ONNX IR
class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super().__init__("ONNX Converter")

        # 添加脚本支持的参数
        self.add_argument(
            "--pipeline",
            type=str,
            choices=SUPPORTED_PIPELINES,
            default="feature-extraction",
        )
        self.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model's id or path (ex: bert-base-cased)",
        )
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument(
            "--framework",
            type=str,
            choices=["pt", "tf"],
            help="Framework for loading the model",
        )
        self.add_argument("--opset", type=int, default=11, help="ONNX opset to use")
        self.add_argument(
            "--check-loading",
            action="store_true",
            help="Check ONNX is able to load the model",
        )
        self.add_argument(
            "--use-external-format",
            action="store_true",
            help="Allow exporting model >= than 2Gb",
        )
        self.add_argument(
            "--quantize",
            action="store_true",
            help="Quantize the neural network to be run with int8",
        )
        self.add_argument("output")

# 定义一个函数，用于在文件名末尾添加标识符
def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath
    # 将给定的文件路径对象的文件名末尾添加一个标识符后缀，并返回新的文件路径字符串
    def add_identifier_suffix(filename, identifier):
        # 使用路径对象的父目录路径和文件名（不含后缀）拼接标识符后缀，生成新的路径对象
        new_path = filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)
        # 返回新路径对象的字符串表示形式
        return new_path
def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        # 尝试导入 onnxruntime 模块
        import onnxruntime

        # 解析已安装的 onnxruntime 的版本
        ort_version = parse(onnxruntime.__version__)

        # 我们需要至少版本 1.4.0
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            # 如果找到旧版本的 onnxruntime，则引发 ImportError
            raise ImportError(
                f"We found an older version of onnxruntime ({onnxruntime.__version__}) "
                f"but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.\n"
                "Please update onnxruntime by running `pip install --upgrade onnxruntime`"
            )

    except ImportError:
        # 如果未安装 onnxruntime，则引发 ImportError
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. "
            "Please install the onnxruntime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )


def ensure_valid_input(model, tokens, input_names):
    """
    Ensure inputs are presented in the correct order, without any Non

    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    print("Ensuring inputs are in correct order")

    # 获取模型的参数名称
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # 从索引1开始以跳过 "self" 参数
        if arg_name in input_names:
            # 如果参数名存在于输入名称中，则按顺序添加到列表中
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            # 如果参数名不在生成的输入列表中，则打印消息并跳出循环
            print(f"{arg_name} is not present in the generated input list.")
            break

    # 打印生成的输入名称顺序
    print(f"Generated inputs order: {ordered_input_names}")
    # 返回已排序的输入名称和模型参数的元组
    return ordered_input_names, tuple(model_args)


def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model

    Args:
        nlp: The pipeline object holding the model to be exported
        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)

    Returns:

        - List of the inferred input variable names
        - List of the inferred output variable names
        - Dictionary with input/output variables names as key and shape tensor as value
        - a BatchEncoding reference which was used to infer all the above information
    """
    # 定义一个函数，用于构建张量的形状字典
    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        # 如果张量是元组或列表类型，则递归调用 build_shape_dict 函数
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]
        else:
            # 假设批量维度是第一个轴，且只有一个元素（这可能并不总是正确的...）
            axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
            # 如果是输入张量
            if is_input:
                # 如果张量的维度为2，则将第二个维度标记为 "sequence"，否则引发 ValueError
                if len(tensor.shape) == 2:
                    axes[1] = "sequence"
                else:
                    raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
            else:
                # 查找与序列长度相同的维度，并将其标记为 "sequence"
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: "sequence" for dim in seq_axes})
    
        # 打印找到的输入或输出张量的形状字典
        print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
        return axes
    
    # 使用 tokenizer 函数对输入文本进行分词，并返回张量
    tokens = nlp.tokenizer("This is a sample output", return_tensors=framework)
    # 获取序列长度
    seq_len = tokens.input_ids.shape[-1]
    # 使用模型处理 tokens，如果框架是 "pt" 则调用 model 方法，否则直接传入 tokens
    outputs = nlp.model(**tokens) if framework == "pt" else nlp.model(tokens)
    # 如果输出是 ModelOutput 类型，则将其转换为元组
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    # 如果输出不是列表或元组类型，则将其包装成单元素的元组
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    
    # 生成输入变量名和轴的动态字典
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}
    
    # 展开可能为组合的输出（例如 gpt2 中的过去状态或注意力权重）
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)
    
    # 生成输出变量名和轴的动态字典
    output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}
    
    # 创建聚合轴表示
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    # 返回输入变量名、输出变量名、动态轴和 tokens
    return input_vars, output_names, dynamic_axes, tokens
# 将通过命令行界面提供的参数集合转换为实际的管道引用（tokenizer + model）
def load_graph_from_args(
    pipeline_name: str, framework: str, model: str, tokenizer: Optional[str] = None, **models_kwargs
) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model

    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object

    """
    # 如果未提供 tokenizer
    if tokenizer is None:
        # 使用 model 作为 tokenizer
        tokenizer = model

    # 检查所需的框架是否可用
    if framework == "pt" and not is_torch_available():
        # 如果 PyTorch 未安装，则抛出异常
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")
    if framework == "tf" and not is_tf_available():
        # 如果 TensorFlow 未安装，则抛出异常
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")

    # 打印加载管道的信息（模型: model，tokenizer: tokenizer）
    print(f"Loading pipeline (model: {model}, tokenizer: {tokenizer})")

    # 分配 tokenizer 和模型
    return pipeline(pipeline_name, model=model, tokenizer=tokenizer, framework=framework, model_kwargs=models_kwargs)


# 将基于 PyTorch 的管道导出为 ONNX 中间表示（IR）
def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:

    """
    # 如果 PyTorch 未安装，则抛出异常
    if not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    # 打印使用 PyTorch 框架的信息和版本号
    print(f"Using framework PyTorch: {torch.__version__}")

    with torch.no_grad():
        # 推断形状
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        # 确保输入有效
        ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)

        # 导出模型
        export(
            nlp.model,
            model_args,
            f=output.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


# 将基于 TensorFlow 的管道导出为 ONNX 中间表示（IR）
def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR)

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model


    """
    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    如果 TensorFlow 不可用：
        抛出异常，指示无法转换因为未安装 TensorFlow，请先安装 TensorFlow。

    打印警告信息，提醒用户 TensorFlow 不支持导出超过 2GB 的模型。

    尝试导入 TensorFlow 和 tf2onnx 库，并获取其版本信息。

    打印所使用的框架为 TensorFlow 的版本号以及 tf2onnx 的版本号。

    # 构建
    调用 infer_shapes 函数，推断输入输出形状和动态轴信息。

    # 前向传播
    使用 nlp 模型进行预测，得到 tokens 数据。
    根据 tokens 的数据构建输入签名。
    使用 tf2onnx 的 from_keras 函数，将 Keras 模型转换为 ONNX 格式，指定 opset 版本和输出路径。

    如果导入模块失败：
        抛出异常，指示无法导入所需模块来将 TF 模型转换为 ONNX，请先安装所需模块。
```  
def convert(
    framework: str,
    model: str,
    output: Path,
    opset: int,
    tokenizer: Optional[str] = None,
    use_external_format: bool = False,
    pipeline_name: str = "feature-extraction",
    **model_kwargs,
):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format

    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format:
            Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)
        model_kwargs: Keyword arguments to be forwarded to the model constructor

    Returns:

    """
    # 发出警告，提醒用户该功能即将被移除
    warnings.warn(
        "The `transformers.convert_graph_to_onnx` package is deprecated and will be removed in version 5 of"
        " Transformers",
        FutureWarning,
    )
    # 打印 ONNX 操作集版本信息
    print(f"ONNX opset version set to: {opset}")

    # 加载 pipeline
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer, **model_kwargs)

    # 检查输出路径的父目录是否存在，若不存在则创建
    if not output.parent.exists():
        print(f"Creating folder {output.parent}")
        makedirs(output.parent.as_posix())
    # 如果输出路径的父目录不为空，则抛出异常
    elif len(listdir(output.parent.as_posix())) > 0:
        raise Exception(f"Folder {output.parent.as_posix()} is not empty, aborting conversion")

    # 导出图形
    # 若使用 PyTorch 框架，则调用 convert_pytorch 函数
    if framework == "pt":
        convert_pytorch(nlp, opset, output, use_external_format)
    # 若使用 TensorFlow 框架，则调用 convert_tensorflow 函数
    else:
        convert_tensorflow(nlp, opset, output)


def optimize(onnx_model_path: Path) -> Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possible

    Args:
        onnx_model_path: filepath where the model binary description is stored

    Returns: Path where the optimized model binary description has been saved

    """
    # 导入必要的模块
    from onnxruntime import InferenceSession, SessionOptions

    # 生成优化后的模型文件路径
    opt_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    # 设置会话选项
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    # 加载原始的 ONNX 模型
    _ = InferenceSession(onnx_model_path.as_posix(), sess_option)

    # 打印优化后的模型路径
    print(f"Optimized model has been written at {opt_model_path}: \N{heavy check mark}")
    # 提示优化后的模型可能包含特定于硬件的运算符，可能不可移植
    print("/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\")

    return opt_model_path


def quantize(onnx_model_path: Path) -> Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Args:
        onnx_model_path: 存储导出的 ONNX 模型的路径

    Returns: 生成的量化模型的路径
    """
    import onnx
    import onnxruntime
    from onnx.onnx_pb import ModelProto
    from onnxruntime.quantization import QuantizationMode
    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
    from onnxruntime.quantization.registry import IntegerOpsRegistry

    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_model_path.as_posix())

    if parse(onnx.__version__) < parse("1.5.0"):
        print(
            "Models larger than 2GB will fail to quantize due to protobuf constraint.\n"
            "Please upgrade to onnxruntime >= 1.5.0."
        )

    # 复制模型
    copy_model = ModelProto()
    copy_model.CopyFrom(onnx_model)

    # 构建量化器
    # onnxruntime 在 v1.13.1 中将 input_qType 重命名为 activation_qType，因此我们
    # 检查 onnxruntime 版本以确保向后兼容性。
    # 参见：https://github.com/microsoft/onnxruntime/pull/12873
    if parse(onnxruntime.__version__) < parse("1.13.1"):
        quantizer = ONNXQuantizer(
            model=copy_model,
            per_channel=False,
            reduce_range=False,
            mode=QuantizationMode.IntegerOps,
            static=False,
            weight_qType=True,
            input_qType=False,
            tensors_range=None,
            nodes_to_quantize=None,
            nodes_to_exclude=None,
            op_types_to_quantize=list(IntegerOpsRegistry),
        )
    else:
        quantizer = ONNXQuantizer(
            model=copy_model,
            per_channel=False,
            reduce_range=False,
            mode=QuantizationMode.IntegerOps,
            static=False,
            weight_qType=True,
            activation_qType=False,
            tensors_range=None,
            nodes_to_quantize=None,
            nodes_to_exclude=None,
            op_types_to_quantize=list(IntegerOpsRegistry),
        )

    # 量化并导出
    quantizer.quantize_model()

    # 在模型名称末尾添加 "-quantized"
    quantized_model_path = generate_identified_filename(onnx_model_path, "-quantized")

    # 保存模型
    print(f"Quantized model has been written at {quantized_model_path}: \N{heavy check mark}")
    onnx.save_model(quantizer.model.model, quantized_model_path.as_posix())

    return quantized_model_path
# 定义一个名为 verify 的函数，用于验证 ONNX 模型的加载情况
def verify(path: Path):
    # 从 onnxruntime 库中导入 InferenceSession 和 SessionOptions 类，以及 RuntimeException 异常类
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    # 打印正在检查的 ONNX 模型加载路径
    print(f"Checking ONNX model loading from: {path} ...")
    try:
        # 创建 SessionOptions 实例
        onnx_options = SessionOptions()
        # 使用 InferenceSession 加载 ONNX 模型，并指定使用 CPUExecutionProvider
        _ = InferenceSession(path.as_posix(), onnx_options, providers=["CPUExecutionProvider"])
        # 打印模型成功加载的消息
        print(f"Model {path} correctly loaded: \N{heavy check mark}")
    except RuntimeException as re:
        # 捕获并打印加载模型时的异常信息
        print(f"Error while loading the model {re}: \N{heavy ballot x}")


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 OnnxConverterArgumentParser 实例
    parser = OnnxConverterArgumentParser()
    # 解析命令行参数
    args = parser.parse_args()

    # 确保输出路径是绝对路径
    args.output = Path(args.output).absolute()

    try:
        # 打印转换模型为 ONNX 的提示信息
        print("\n====== Converting model to ONNX ======")
        # 转换模型为 ONNX 格式
        convert(
            args.framework,
            args.model,
            args.output,
            args.opset,
            args.tokenizer,
            args.use_external_format,
            args.pipeline,
        )

        # 如果需要量化
        if args.quantize:
            # 确保满足 ONNX Runtime 量化的要求
            check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)

            # 在 TensorFlow 上，ONNX Runtime 的优化效果可能不如 PyTorch
            if args.framework == "tf":
                print(
                    "\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n"
                    "\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n"
                    "\t For more information, please refer to the onnxruntime documentation:\n"
                    "\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n"
                )

            # 打印优化 ONNX 模型的提示信息
            print("\n====== Optimizing ONNX model ======")

            # 对输出的 ONNX 模型进行优化
            args.optimized_output = optimize(args.output)

            # 对优化后的 ONNX 模型进行量化
            args.quantized_output = quantize(args.optimized_output)

        # 进行模型加载验证
        if args.check_loading:
            # 打印验证导出的 ONNX 模型的提示信息
            print("\n====== Check exported ONNX model(s) ======")
            # 验证导出的 ONNX 模型
            verify(args.output)

            # 如果存在优化后的输出路径，则验证优化后的模型
            if hasattr(args, "optimized_output"):
                verify(args.optimized_output)

            # 如果存在量化后的输出路径，则验证量化后的模型
            if hasattr(args, "quantized_output"):
                verify(args.quantized_output)

    except Exception as e:
        # 捕获并打印转换模型时的异常信息
        print(f"Error while converting the model: {e}")
        # 退出程序，返回错误码 1
        exit(1)
```