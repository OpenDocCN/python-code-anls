# `.\convert_graph_to_onnx.py`

```
# 版权声明和许可信息
# 版权所有 2020 年 HuggingFace 团队保留所有权利。
# 
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据许可证分发的软件是基于“按原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查阅许可证以获取具体的法律语言。
#

import warnings  # 导入警告模块
from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser 类
from os import listdir, makedirs  # 从 os 模块导入 listdir 和 makedirs 函数
from pathlib import Path  # 导入 Path 类
from typing import Dict, List, Optional, Tuple  # 导入类型提示

from packaging.version import Version, parse  # 从 packaging.version 模块导入 Version 和 parse 函数

# 导入 transformers 库中的相关模块和类
from transformers.pipelines import Pipeline, pipeline  
from transformers.tokenization_utils import BatchEncoding  
from transformers.utils import ModelOutput, is_tf_available, is_torch_available  

# 定义最小支持的 ONNX Runtime 版本
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")

# 定义支持的 pipeline 类型列表
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

# 定义一个 ArgumentParser 的子类，用于解析 ONNX 转换器的命令行参数
class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super().__init__("ONNX Converter")  # 调用父类构造函数，设置解析器的描述信息为 "ONNX Converter"

        # 添加命令行参数
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
            help="Model's id or path (ex: google-bert/bert-base-cased)",
        )
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: google-bert/bert-base-cased)")
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
        self.add_argument("output")  # 添加输出参数

# 定义一个函数，生成带有标识符的文件名
def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    # 在提供的文件路径末尾（在扩展名之前，如果有的话）添加一个字符串标识符
    
    Args:
        filename: pathlib.Path 实际的路径对象，我们希望在其末尾添加标识符后缀
        identifier: 要添加的后缀
    
    Returns: 添加了标识符的字符串，连接在文件名的末尾
# 检查 onnxruntime 的安装情况及版本是否符合要求
def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime

        # 解析已安装的 onnxruntime 的版本
        ort_version = parse(onnxruntime.__version__)

        # 要求最低版本为 1.4.0
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


# 确保输入在正确顺序中，没有非法输入
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

    # 获取模型前向方法的参数名列表
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # 从索引1开始以跳过 "self" 参数
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f"{arg_name} is not present in the generated input list.")
            break

    # 打印生成的输入顺序
    print(f"Generated inputs order: {ordered_input_names}")
    return ordered_input_names, tuple(model_args)


# 推断模型输入输出张量的静态与动态轴
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
    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        # 如果 tensor 是元组或列表，则递归调用 build_shape_dict 处理每个元素
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]
        
        else:
            # 假设第一个维度是批处理维度，且只有一个元素
            axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
            # 如果是输入数据，判断维度是否为二维，将第二个维度标记为 "sequence"
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = "sequence"
                else:
                    raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
            else:
                # 找到与指定序列长度相匹配的维度，并将其标记为 "sequence"
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: "sequence" for dim in seq_axes})

        # 打印找到的输入或输出的名称、形状信息
        print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
        return axes

    # 使用 NLP 模型的分词器生成 tokens，并返回张量表示
    tokens = nlp.tokenizer("This is a sample output", return_tensors=framework)
    # 获取序列长度
    seq_len = tokens.input_ids.shape[-1]
    # 根据框架类型调用 NLP 模型
    outputs = nlp.model(**tokens) if framework == "pt" else nlp.model(tokens)
    # 如果输出是 ModelOutput 类型，则转换为元组
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    # 如果输出不是列表或元组，则将其包装成元组
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # 生成输入变量的名称及其动态轴信息
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}

    # 将可能包含分组输出（例如 gpt2 中的过去状态或注意力）展平
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    # 生成输出变量的名称及其动态轴信息
    output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}

    # 创建汇总的动态轴表示
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens
def load_graph_from_args(
    pipeline_name: str, framework: str, model: str, tokenizer: Optional[str] = None, **models_kwargs
) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model)

    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object
    """
    # 如果未提供 tokenizer，则使用 model 作为 tokenizer
    if tokenizer is None:
        tokenizer = model

    # 检查所需的 framework 是否可用
    if framework == "pt" and not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")
    if framework == "tf" and not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")

    print(f"Loading pipeline (model: {model}, tokenizer: {tokenizer})")

    # 分配 tokenizer 和 model
    return pipeline(pipeline_name, model=model, tokenizer=tokenizer, framework=framework, model_kwargs=models_kwargs)


def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR)

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:
    """
    if not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    print(f"Using framework PyTorch: {torch.__version__}")

    # 通过 infer_shapes 推断输入、输出和动态轴
    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        # 确保输入名称有效，并按顺序提供模型参数
        ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)

        # 导出模型到 ONNX
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


def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR)

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
    """
    # 检查是否安装了 TensorFlow
    if not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")
    # 检查 TensorFlow 是否可用，若不可用则引发异常提示安装 TensorFlow
    if not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")
    
    # 提示用户注意：TensorFlow 不支持导出超过2GB的模型
    print("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")
    
    try:
        # 尝试导入 TensorFlow 和 tf2onnx
        import tensorflow as tf
        import tf2onnx
        from tf2onnx import __version__ as t2ov
        
        # 打印当前使用的框架和 tf2onnx 的版本信息
        print(f"Using framework TensorFlow: {tf.version.VERSION}, tf2onnx: {t2ov}")
    
        # 推断模型输入形状等信息，并获取 tokens
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "tf")
    
        # 使用模型进行前向推断
        nlp.model.predict(tokens.data)
        
        # 根据 tokens 的数据创建输入签名
        input_signature = [tf.TensorSpec.from_tensor(tensor, name=key) for key, tensor in tokens.items()]
        
        # 使用 tf2onnx 将 Keras 模型转换为 ONNX 格式
        model_proto, _ = tf2onnx.convert.from_keras(
            nlp.model, input_signature, opset=opset, output_path=output.as_posix()
        )
    
    except ImportError as e:
        # 若导入出错，引发异常提示缺少必要的包
        raise Exception(
            f"Cannot import {e.name} required to convert TF model to ONNX. Please install {e.name} first. {e}"
        )
# 定义一个函数 convert，用于将管道对象转换为 ONNX 中间表示（IR）格式
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
        framework: 管道所使用的框架 ("pt" 或 "tf")
        model: 管道加载的模型名称
        output: 存储 ONNX 图的路径
        opset: 使用的 ONNX 运算集的实际版本
        tokenizer: 管道所使用的分词器名称，如果未提供则默认使用模型名称
        use_external_format:
            是否将模型定义与其参数分离，以允许超过 2GB 的模型大小（仅适用于 PyTorch）
        pipeline_name: 实例化的管道类型（ner、question-answering 等）
        model_kwargs: 转发给模型构造函数的关键字参数

    Returns:

    """
    # 发出警告，指示 `transformers.convert_graph_to_onnx` 包已过时，并将在 Transformers 的第五个版本中移除
    warnings.warn(
        "The `transformers.convert_graph_to_onnx` package is deprecated and will be removed in version 5 of"
        " Transformers",
        FutureWarning,
    )
    # 打印设置的 ONNX 运算集版本号
    print(f"ONNX opset version set to: {opset}")

    # 加载管道对象
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer, **model_kwargs)

    # 检查输出路径的父目录是否存在，若不存在则创建
    if not output.parent.exists():
        print(f"Creating folder {output.parent}")
        makedirs(output.parent.as_posix())
    # 若输出路径的父目录非空，则抛出异常
    elif len(listdir(output.parent.as_posix())) > 0:
        raise Exception(f"Folder {output.parent.as_posix()} is not empty, aborting conversion")

    # 根据不同的框架导出图
    if framework == "pt":
        convert_pytorch(nlp, opset, output, use_external_format)
    else:
        convert_tensorflow(nlp, opset, output)


# 定义一个函数 optimize，用于优化 ONNX 模型
def optimize(onnx_model_path: Path) -> Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possible

    Args:
        onnx_model_path: 模型二进制描述文件的路径

    Returns: 优化后的模型二进制描述文件保存的路径

    """
    from onnxruntime import InferenceSession, SessionOptions

    # 生成带有后缀 "-optimized" 的优化模型文件名
    opt_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    sess_option = SessionOptions()
    # 设置优化后的模型文件路径
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    _ = InferenceSession(onnx_model_path.as_posix(), sess_option)

    # 打印优化后的模型写入路径
    print(f"Optimized model has been written at {opt_model_path}: \N{heavy check mark}")
    # 提示优化后的模型包含特定硬件操作符，可能不具备可移植性
    print("/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\")

    return opt_model_path


# 定义一个函数 quantize，用于将模型权重从 float32 量化为 int8，以实现在现代 CPU 上高效推断
def quantize(onnx_model_path: Path) -> Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: 模型二进制描述文件的路径

    Returns: 量化后的模型二进制描述文件保存的路径

    """
    # 函数体未完，暂时省略
    # 导入必要的库和模块
    import onnx
    import onnxruntime
    from onnx.onnx_pb import ModelProto
    from onnxruntime.quantization import QuantizationMode
    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
    from onnxruntime.quantization.registry import IntegerOpsRegistry

    # 加载指定路径下的 ONNX 模型
    onnx_model = onnx.load(onnx_model_path.as_posix())

    # 检查 ONNX 版本是否小于 1.5.0，提示模型大小限制问题
    if parse(onnx.__version__) < parse("1.5.0"):
        print(
            "Models larger than 2GB will fail to quantize due to protobuf constraint.\n"
            "Please upgrade to onnxruntime >= 1.5.0."
        )

    # 创建 ONNX 模型的副本
    copy_model = ModelProto()
    copy_model.CopyFrom(onnx_model)

    # 构造量化器
    # 检查 ONNX Runtime 版本，根据版本选择合适的量化器参数设置
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

    # 执行模型量化
    quantizer.quantize_model()

    # 生成量化后模型的文件名，并在原模型文件名末尾添加 "-quantized" 后缀
    quantized_model_path = generate_identified_filename(onnx_model_path, "-quantized")

    # 保存量化后的模型
    print(f"Quantized model has been written at {quantized_model_path}: \N{heavy check mark}")
    onnx.save_model(quantizer.model.model, quantized_model_path.as_posix())

    # 返回量化后模型的路径
    return quantized_model_path
def verify(path: Path):
    # 引入需要的库和模块
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    # 打印正在加载的 ONNX 模型路径
    print(f"Checking ONNX model loading from: {path} ...")
    try:
        # 设置 ONNX 运行时的选项
        onnx_options = SessionOptions()
        # 创建推理会话，加载模型并指定 CPU 执行提供者
        _ = InferenceSession(path.as_posix(), onnx_options, providers=["CPUExecutionProvider"])
        # 打印模型加载成功的消息
        print(f"Model {path} correctly loaded: \N{heavy check mark}")
    except RuntimeException as re:
        # 捕获模型加载时的异常并打印错误消息
        print(f"Error while loading the model {re}: \N{heavy ballot x}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()

    # 确保输出路径为绝对路径
    args.output = Path(args.output).absolute()

    try:
        print("\n====== Converting model to ONNX ======")
        # 执行模型转换
        convert(
            args.framework,
            args.model,
            args.output,
            args.opset,
            args.tokenizer,
            args.use_external_format,
            args.pipeline,
        )

        if args.quantize:
            # 确保满足 quantization 在 onnxruntime 上的要求
            check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)

            # 对于 TensorFlow 框架，性能优化不如 PyTorch 显著
            if args.framework == "tf":
                print(
                    "\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n"
                    "\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n"
                    "\t For more information, please refer to the onnxruntime documentation:\n"
                    "\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n"
                )

            print("\n====== Optimizing ONNX model ======")

            # 对优化后的模型进行量化
            args.optimized_output = optimize(args.output)

            # 在正确的图上执行量化
            args.quantized_output = quantize(args.optimized_output)

        # 验证转换后的模型
        if args.check_loading:
            print("\n====== Check exported ONNX model(s) ======")
            verify(args.output)

            if hasattr(args, "optimized_output"):
                verify(args.optimized_output)

            if hasattr(args, "quantized_output"):
                verify(args.quantized_output)

    except Exception as e:
        # 捕获转换过程中的异常并打印错误消息
        print(f"Error while converting the model: {e}")
        exit(1)
```