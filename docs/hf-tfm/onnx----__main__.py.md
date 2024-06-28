# `.\onnx\__main__.py`

```py
# 版权声明和许可信息
#
# 版权所有 2021 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
import subprocess  # 导入 subprocess 模块，用于执行外部命令和进程管理
import sys  # 导入 sys 模块，提供对 Python 运行时系统的访问
import warnings  # 导入 warnings 模块，用于管理警告信息
from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser 类，用于解析命令行参数
from pathlib import Path  # 导入 Path 类，用于操作路径

from packaging import version  # 导入 version 模块，用于处理版本信息

from .. import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer  # 导入自定义模块
from ..utils import logging  # 导入自定义模块中的 logging 工具
from ..utils.import_utils import is_optimum_available  # 导入自定义模块中的 is_optimum_available 函数
from .convert import export, validate_model_outputs  # 从当前目录下的 convert 模块导入 export 和 validate_model_outputs 函数
from .features import FeaturesManager  # 从当前目录下的 features 模块导入 FeaturesManager 类
from .utils import get_preprocessor  # 从当前目录下的 utils 模块导入 get_preprocessor 函数

MIN_OPTIMUM_VERSION = "1.5.0"  # 定义最小的 optimum 版本号

ENCODER_DECODER_MODELS = ["vision-encoder-decoder"]  # 定义编码-解码模型列表

# 使用 optimum 导出模型
def export_with_optimum(args):
    if is_optimum_available():  # 如果 optimum 可用
        from optimum.version import __version__ as optimum_version  # 导入 optimum 的版本信息

        parsed_optimum_version = version.parse(optimum_version)  # 解析 optimum 的版本号
        if parsed_optimum_version < version.parse(MIN_OPTIMUM_VERSION):  # 如果 optimum 的版本低于要求的最小版本
            raise RuntimeError(  # 抛出运行时异常
                f"transformers.onnx requires optimum >= {MIN_OPTIMUM_VERSION} but {optimum_version} is installed. You "
                "can upgrade optimum by running: pip install -U optimum[exporters]"
            )
    else:  # 如果 optimum 不可用
        raise RuntimeError(  # 抛出运行时异常
            "transformers.onnx requires optimum to run, you can install the library by running: pip install "
            "optimum[exporters]"
        )
    # 构建命令行参数列表
    cmd_line = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        f"--model {args.model}",
        f"--task {args.feature}",
        f"--framework {args.framework}" if args.framework is not None else "",
        f"{args.output}",
    ]
    proc = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)  # 执行命令行，并获取子进程对象
    proc.wait()  # 等待子进程执行完毕

    logger.info(  # 使用 logger 输出信息
        "The export was done by optimum.exporters.onnx. We recommend using to use this package directly in future, as "
        "transformers.onnx is deprecated, and will be removed in v5. You can find more information here: "
        "https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model."
    )


# 使用 transformers 导出模型
def export_with_transformers(args):
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")  # 如果输出路径不是文件，则拼接文件名
    if not args.output.parent.exists():  # 如果输出路径的父目录不存在
        args.output.parent.mkdir(parents=True)  # 创建父目录及其所有必需的上级目录

    # 分配模型
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, cache_dir=args.cache_dir
    )
    # 检查给定模型是否被支持，并返回模型类型和对应的配置对象
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    # 根据模型配置创建对应的 ONNX 配置对象
    onnx_config = model_onnx_config(model.config)

    # 如果模型类型是编码器-解码器模型
    if model_kind in ENCODER_DECODER_MODELS:
        # 获取编码器和解码器模型对象
        encoder_model = model.get_encoder()
        decoder_model = model.get_decoder()

        # 获取编码器和解码器模型的 ONNX 配置
        encoder_onnx_config = onnx_config.get_encoder_config(encoder_model.config)
        decoder_onnx_config = onnx_config.get_decoder_config(
            encoder_model.config, decoder_model.config, feature=args.feature
        )

        # 如果未指定操作集，则选择编码器和解码器的默认操作集中的最大值
        if args.opset is None:
            args.opset = max(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)

        # 检查指定的操作集是否满足编码器和解码器的最小要求
        if args.opset < min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset):
            raise ValueError(
                f"Opset {args.opset} is not sufficient to export {model_kind}. At least "
                f"{min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)} is required."
            )

        # 根据模型类型创建自动特征提取器对象
        preprocessor = AutoFeatureExtractor.from_pretrained(args.model)

        # 导出编码器模型的 ONNX 文件，并获取输入和输出信息
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            encoder_model,
            encoder_onnx_config,
            args.opset,
            args.output.parent.joinpath("encoder_model.onnx"),
        )

        # 验证编码器模型输出的正确性
        validate_model_outputs(
            encoder_onnx_config,
            preprocessor,
            encoder_model,
            args.output.parent.joinpath("encoder_model.onnx"),
            onnx_outputs,
            args.atol if args.atol else encoder_onnx_config.atol_for_validation,
        )

        # 根据模型类型创建自动分词器对象
        preprocessor = AutoTokenizer.from_pretrained(args.model)

        # 导出解码器模型的 ONNX 文件，并获取输入和输出信息
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            decoder_model,
            decoder_onnx_config,
            args.opset,
            args.output.parent.joinpath("decoder_model.onnx"),
        )

        # 验证解码器模型输出的正确性
        validate_model_outputs(
            decoder_onnx_config,
            preprocessor,
            decoder_model,
            args.output.parent.joinpath("decoder_model.onnx"),
            onnx_outputs,
            args.atol if args.atol else decoder_onnx_config.atol_for_validation,
        )
        # 记录信息，显示模型保存的路径
        logger.info(
            f"All good, model saved at: {args.output.parent.joinpath('encoder_model.onnx').as_posix()},"
            f" {args.output.parent.joinpath('decoder_model.onnx').as_posix()}"
        )
    else:
        # 如果不是第一个分支，则实例化适当的预处理器

        if args.preprocessor == "auto":
            # 如果预处理器类型是 "auto"，则根据模型获取适当的预处理器对象
            preprocessor = get_preprocessor(args.model)
        elif args.preprocessor == "tokenizer":
            # 如果预处理器类型是 "tokenizer"，则使用预训练的 AutoTokenizer 创建预处理器对象
            preprocessor = AutoTokenizer.from_pretrained(args.model)
        elif args.preprocessor == "image_processor":
            # 如果预处理器类型是 "image_processor"，则使用预训练的 AutoImageProcessor 创建预处理器对象
            preprocessor = AutoImageProcessor.from_pretrained(args.model)
        elif args.preprocessor == "feature_extractor":
            # 如果预处理器类型是 "feature_extractor"，则使用预训练的 AutoFeatureExtractor 创建预处理器对象
            preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
        elif args.preprocessor == "processor":
            # 如果预处理器类型是 "processor"，则使用预训练的 AutoProcessor 创建预处理器对象
            preprocessor = AutoProcessor.from_pretrained(args.model)
        else:
            # 如果预处理器类型未知，则抛出 ValueError 异常
            raise ValueError(f"Unknown preprocessor type '{args.preprocessor}'")

        # 确保请求的 opset 足够
        if args.opset is None:
            args.opset = onnx_config.default_onnx_opset

        if args.opset < onnx_config.default_onnx_opset:
            # 如果请求的 opset 小于默认的 opset，抛出 ValueError 异常
            raise ValueError(
                f"Opset {args.opset} is not sufficient to export {model_kind}. "
                f"At least  {onnx_config.default_onnx_opset} is required."
            )

        # 导出模型到 ONNX 格式，获取输入和输出
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            model,
            onnx_config,
            args.opset,
            args.output,
        )

        if args.atol is None:
            # 如果未指定 atol，则使用默认的验证容差值
            args.atol = onnx_config.atol_for_validation

        # 验证导出的模型输出是否符合预期
        validate_model_outputs(onnx_config, preprocessor, model, args.output, onnx_outputs, args.atol)
        
        # 记录信息，指示模型已成功保存
        logger.info(f"All good, model saved at: {args.output.as_posix()}")
        
        # 发出警告，提示使用过时的 ONNX 导出工具，建议将来使用新的导出器
        warnings.warn(
            "The export was done by transformers.onnx which is deprecated and will be removed in v5. We recommend"
            " using optimum.exporters.onnx in future. You can find more information here:"
            " https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model.",
            FutureWarning,
        )
# 主程序入口函数
def main():
    # 创建参数解析器实例，用于解析命令行参数
    parser = ArgumentParser("Hugging Face Transformers ONNX exporter")
    
    # 添加必需参数：模型 ID 或者磁盘上的模型路径
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    
    # 添加可选参数：导出模型时使用的特性类型，默认为 "default"
    parser.add_argument(
        "--feature",
        default="default",
        help="The type of features to export the model with.",
    )
    
    # 添加可选参数：导出模型时使用的 ONNX opset 版本
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    
    # 添加可选参数：验证模型时的绝对差值容忍度
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerance when validating the model."
    )
    
    # 添加可选参数：指定导出模型时使用的框架，可选项为 "pt" 或 "tf"
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the ONNX export."
            " If not provided, will attempt to use the local checkpoint's original framework"
            " or what is available in the environment."
        ),
    )
    
    # 添加位置参数：指定生成的 ONNX 模型存储路径
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")
    
    # 添加可选参数：指定缓存目录的路径
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    
    # 添加可选参数：指定使用的预处理器类型，可选项有多种，如 "auto"、"tokenizer" 等
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["auto", "tokenizer", "feature_extractor", "image_processor", "processor"],
        default="auto",
        help="Which type of preprocessor to use. 'auto' tries to automatically detect it.",
    )
    
    # 添加可选参数：是否使用 transformers.onnx 而非 optimum.exporters.onnx 来执行 ONNX 导出
    parser.add_argument(
        "--export_with_transformers",
        action="store_true",
        help=(
            "Whether to use transformers.onnx instead of optimum.exporters.onnx to perform the ONNX export. It can be "
            "useful when exporting a model supported in transformers but not in optimum, otherwise it is not "
            "recommended."
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了 --export_with_transformers 或者 optimum 模块不可用，使用 transformers.onnx 导出模型
    if args.export_with_transformers or not is_optimum_available():
        export_with_transformers(args)
    else:
        # 否则，使用 optimum.exporters.onnx 导出模型
        export_with_optimum(args)


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 获取日志记录器实例，并设置日志级别为 INFO
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    
    # 调用主程序入口函数 main()
    main()
```