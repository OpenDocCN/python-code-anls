# `.\transformers\onnx\__main__.py`

```
# 版权声明，版权归 The HuggingFace Team 所有
#
# 根据 Apache 许可证，版本 2.0 进行许可
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下地址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“原样”分发的基准分发软件，没有明示或暗示的任何保证或条件
# 请查看许可证，了解特定语言下的权限和限制
import subprocess  # 导入 subprocess 模块
import sys  # 导入 sys 模块
import warnings  # 导入 warnings 模块
from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser 类
from pathlib import Path  # 从 pathlib 模块导入 Path 类

from packaging import version  # 从 packaging 模块导入 version 类

from .. import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer  # 导入特定模块
from ..utils import logging  # 从 utils 模块中导入 logging
from ..utils.import_utils import is_optimum_available  # 从 import_utils 模块导入 is_optimum_available 方法
from .convert import export, validate_model_outputs  # 从 convert 模块导入 export、validate_model_outputs 函数
from .features import FeaturesManager  # 从 features 模块导入 FeaturesManager 类
from .utils import get_preprocessor  # 从 utils 模块导入 get_preprocessor 函数

MIN_OPTIMUM_VERSION = "1.5.0"  # 设置最低 optimum 版本为 1.5.0
ENCODER_DECODER_MODELS = ["vision-encoder-decoder"]  # 设置编码器-解码器模型列表为 ["vision-encoder-decoder"]

def export_with_optimum(args):
    # 如果 optimum 可用
    if is_optimum_available():
        from optimum.version import __version__ as optimum_version  # 导入 optimum 版本号

        parsed_optimum_version = version.parse(optimum_version)  # 解析 optimum 版本号
        # 如果解析后的 optimum 版本小于最低版本要求
        if parsed_optimum_version < version.parse(MIN_OPTIMUM_VERSION):
            # 抛出运行时错误，提示升级 optimum 版本
            raise RuntimeError(
                f"transformers.onnx requires optimum >= {MIN_OPTIMUM_VERSION} but {optimum_version} is installed. You "
                "can upgrade optimum by running: pip install -U optimum[exporters]"
            )
    else:
        # 如果 optimum 不可用，抛出错误提示安装 optimum
        raise RuntimeError(
            "transformers.onnx requires optimum to run, you can install the library by running: pip install "
            "optimum[exporters]"
        )
    # 构造命令行参数列表
    cmd_line = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        f"--model {args.model}",
        f"--task {args.feature}",
        f"--framework {args.framework}" if args.framework is not None else "",
        f"{args.output}",
    ]
    # 启动子进程执行命令行
    proc = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
    proc.wait()

    logger.info(
        "The export was done by optimum.exporters.onnx. We recommend using to use this package directly in future, as "
        "transformers.onnx is deprecated, and will be removed in v5. You can find more information here: "
        "https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model."
    )

def export_with_transformers(args):
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")  # 设置输出路径
    if not args.output.parent.exists():  # 如果输出路径的父目录不存在
        args.output.parent.mkdir(parents=True)  # 创建输出路径的父目录

    # 分配模型
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, cache_dir=args.cache_dir
    )
    # 检查给定模型是否支持，并返回模型种类和对应的ONNX配置
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    # 根据模型的配置生成ONNX配置
    onnx_config = model_onnx_config(model.config)
    
    # 如果模型种类为编码器-解码器类型
    if model_kind in ENCODER_DECODER_MODELS:
        # 获取编码器模型和解码器模型
        encoder_model = model.get_encoder()
        decoder_model = model.get_decoder()
    
        # 获取编码器和解码器的ONNX配置
        encoder_onnx_config = onnx_config.get_encoder_config(encoder_model.config)
        decoder_onnx_config = onnx_config.get_decoder_config(
            encoder_model.config, decoder_model.config, feature=args.feature
        )
    
        # 如果未提供操作集版本，则使用编码器和解码器的默认ONNX操作集版本的最大值
        if args.opset is None:
            args.opset = max(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)
    
        # 如果提供的操作集版本小于编码器和解码器默认的最小ONNX操作集版本
        if args.opset < min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset):
            # 抛出数值错误并提示需要的最小操作集版本
            raise ValueError(
                f"Opset {args.opset} is not sufficient to export {model_kind}. At least "
                f" {min(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)} is required."
            )
    
        # 从预训练模型中创建预处理器
        preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
    
        # 导出编码器模型的ONNX输入和输出
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            encoder_model,
            encoder_onnx_config,
            args.opset,
            args.output.parent.joinpath("encoder_model.onnx"),
        )
    
        # 验证编码器模型的输出
        validate_model_outputs(
            encoder_onnx_config,
            preprocessor,
            encoder_model,
            args.output.parent.joinpath("encoder_model.onnx"),
            onnx_outputs,
            args.atol if args.atol else encoder_onnx_config.atol_for_validation,
        )
    
        # 从预训练模型中创建预处理器
        preprocessor = AutoTokenizer.from_pretrained(args.model)
    
        # 导出解码器模型的ONNX输入和输出
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            decoder_model,
            decoder_onnx_config,
            args.opset,
            args.output.parent.joinpath("decoder_model.onnx"),
        )
    
        # 验证解码器模型的输出
        validate_model_outputs(
            decoder_onnx_config,
            preprocessor,
            decoder_model,
            args.output.parent.joinpath("decoder_model.onnx"),
            onnx_outputs,
            args.atol if args.atol else decoder_onnx_config.atol_for_validation,
        )
        
        # 记录模型已保存
        logger.info(
            f"All good, model saved at: {args.output.parent.joinpath('encoder_model.onnx').as_posix()},"
            f" {args.output.parent.joinpath('decoder_model.onnx').as_posix()}"
        )
    else:
        # 如果不是首次运行，则实例化相应的预处理器
        if args.preprocessor == "auto":
            # 根据模型选择相应的预处理器
            preprocessor = get_preprocessor(args.model)
        elif args.preprocessor == "tokenizer":
            # 使用AutoTokenizer模块中的方法获取预处理器实例
            preprocessor = AutoTokenizer.from_pretrained(args.model)
        elif args.preprocessor == "image_processor":
            # 使用AutoImageProcessor模块中的方法获取预处理器实例
            preprocessor = AutoImageProcessor.from_pretrained(args.model)
        elif args.preprocessor == "feature_extractor":
            # 使用AutoFeatureExtractor模块中的方法获取预处理器实例
            preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
        elif args.preprocessor == "processor":
            # 使用AutoProcessor模块中的方法获取预处理器实例
            preprocessor = AutoProcessor.from_pretrained(args.model)
        else:
            # 如果预处理器类型未知，则报错
            raise ValueError(f"Unknown preprocessor type '{args.preprocessor}'")
        
        # 确保所请求的 opset 足够
        if args.opset is None:
            # 如果未指定opset，则使用默认的opset
            args.opset = onnx_config.default_onnx_opset
        
        if args.opset < onnx_config.default_onnx_opset:
            # 如果所请求的opset小于默认的opset，则报错
            raise ValueError(
                f"Opset {args.opset} is not sufficient to export {model_kind}. "
                f"At least  {onnx_config.default_onnx_opset} is required."
            )
        
        # 导出模型，得到输入和输出的ONNX表示
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            model,
            onnx_config,
            args.opset,
            args.output,
        )
        
        if args.atol is None:
            # 如果未指定atol，则使用默认的atol
            args.atol = onnx_config.atol_for_validation
        
        # 验证模型的输出是否正确
        validate_model_outputs(onnx_config, preprocessor, model, args.output, onnx_outputs, args.atol)
        
        # 输出模型位置信息
        logger.info(f"All good, model saved at: {args.output.as_posix()}")
        
        # 发出警告，说明该功能即将被删除并建议使用新的导出方法
        warnings.warn(
            "The export was done by transformers.onnx which is deprecated and will be removed in v5. We recommend"
            " using optimum.exporters.onnx in future. You can find more information here:"
            " https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model.",
            FutureWarning,
        )
# 定义主函数
def main():
    # 创建参数解析器对象，并设置描述
    parser = ArgumentParser("Hugging Face Transformers ONNX exporter")
    # 添加模型参数，类型为字符串，必填项，用于指定模型在huggingface.co上的ID或者磁盘上模型的路径
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    # 添加特性参数，默认为"default"，用于指定导出模型所使用的特性类型
    parser.add_argument(
        "--feature",
        default="default",
        help="The type of features to export the model with.",
    )
    # 添加ONNX opset版本参数，默认为None，用于指定导出模型所使用的ONNX opset版本
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    # 添加绝对误差容忍度参数，默认为None，用于在验证模型时指定绝对差异容忍度
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerance when validating the model."
    )
    # 添加框架参数，类型为字符串，可选项为"pt"和"tf"，默认为None，用于指定ONNX导出所使用的框架
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
    # 添加输出路径参数，类型为Path，用于指定生成的ONNX模型存储路径
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")
    # 添加缓存路径参数，类型为字符串，默认为None，用于指定缓存存储路径
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    # 添加预处理器类型参数，类型为字符串，可选项为"auto"、"tokenizer"、"feature_extractor"、"image_processor"、"processor"，默认为"auto"，用于指定预处理器的类型
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["auto", "tokenizer", "feature_extractor", "image_processor", "processor"],
        default="auto",
        help="Which type of preprocessor to use. 'auto' tries to automatically detect it.",
    )
    # 添加使用transformers.onnx导出的选项，是布尔类型的标志，用于指明是否使用transformers.onnx而非optimum.exporters.onnx进行ONNX导出
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
    # 如果选择使用transformers.onnx导出，或者optimum库不可用，则执行export_with_transformers函数
    if args.export_with_transformers or not is_optimum_available():
        export_with_transformers(args)
    # 否则执行export_with_optimum函数
    else:
        export_with_optimum(args)


# 如果是作为主程序运行，则设置日志级别为INFO，并执行main函数
if __name__ == "__main__":
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
```