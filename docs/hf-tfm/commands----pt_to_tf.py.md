# `.\transformers\commands\pt_to_tf.py`

```py
# 导入模块
import inspect  # 导入 inspect 模块，用于检查对象的属性和方法
import os  # 导入 os 模块，提供了一种与操作系统交互的方法
from argparse import ArgumentParser, Namespace  # 从 argparse 模块导入 ArgumentParser 类和 Namespace 类
from importlib import import_module  # 从 importlib 模块导入 import_module 函数

# 导入第三方库和模块
import huggingface_hub  # 导入 huggingface_hub 库
import numpy as np  # 导入 numpy 库，并将其重命名为 np
from packaging import version  # 从 packaging 模块导入 version 类

# 导入 transformers 库中的相关模块和函数
from .. import (  # 从当前目录的父目录中导入模块或子包
    FEATURE_EXTRACTOR_MAPPING,  # 导入 FEATURE_EXTRACTOR_MAPPING 常量
    IMAGE_PROCESSOR_MAPPING,  # 导入 IMAGE_PROCESSOR_MAPPING 常量
    PROCESSOR_MAPPING,  # 导入 PROCESSOR_MAPPING 常量
    TOKENIZER_MAPPING,  # 导入 TOKENIZER_MAPPING 常量
    AutoConfig,  # 导入 AutoConfig 类
    AutoFeatureExtractor,  # 导入 AutoFeatureExtractor 类
    AutoImageProcessor,  # 导入 AutoImageProcessor 类
    AutoProcessor,  # 导入 AutoProcessor 类
    AutoTokenizer,  # 导入 AutoTokenizer 类
    is_datasets_available,  # 导入 is_datasets_available 函数
    is_tf_available,  # 导入 is_tf_available 函数
    is_torch_available,  # 导入 is_torch_available 函数
)
from ..utils import TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging  # 从当前目录的父目录中的 utils 模块导入常量和 logging 函数
from . import BaseTransformersCLICommand  # 从当前目录的子模块中导入 BaseTransformersCLICommand 类

# 检查 TensorFlow 是否可用，并根据情况进行配置
if is_tf_available():  # 如果 TensorFlow 可用
    import tensorflow as tf  # 导入 tensorflow 库
    tf.config.experimental.enable_tensor_float_32_execution(False)  # 配置 TensorFlow 以禁用 32 位浮点数执行

# 检查 PyTorch 是否可用，并根据情况导入
if is_torch_available():  # 如果 PyTorch 可用
    import torch  # 导入 torch 库

# 检查 datasets 是否可用，并根据情况导入
if is_datasets_available():  # 如果 datasets 可用
    from datasets import load_dataset  # 导入 load_dataset 函数

# 定义最大误差常量
MAX_ERROR = 5e-5  # 设定最大误差，用于检查模型转换是否成功，比内部测试的误差容限略大，以避免用户界面的不稳定性错误

# 定义转换命令的工厂函数
def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(
        args.model_name,  # 模型名称参数
        args.local_dir,  # 本地目录参数
        args.max_error,  # 最大误差参数
        args.new_weights,  # 新权重参数
        args.no_pr,  # 不推送请求参数
        args.push,  # 推送参数
        args.extra_commit_description,  # 额外的提交描述参数
        args.override_model_class,  # 覆盖模型类参数
    )

# 定义 PyTorch 转 TensorFlow 2 的命令类
class PTtoTFCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        # 创建子命令解析器实例，用于将该命令注册到 argparse，使其可用于 transformer-cli
        train_parser = parser.add_parser(
            "pt-to-tf",
            help=(
                "CLI tool to run convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint."
                " Can also be used to validate existing weights without opening PRs, with --no-pr."
            ),
        )
        # 添加命令行参数，指定模型名称，包括所有者/组织，如在 hub 上看到的那样
        train_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The model name, including owner/organization, as seen on the hub.",
        )
        # 添加命令行参数，指定模型存储的本地目录，可选，默认为 /tmp/{model_name}
        train_parser.add_argument(
            "--local-dir",
            type=str,
            default="",
            help="Optional local directory of the model repository. Defaults to /tmp/{model_name}",
        )
        # 添加命令行参数，指定最大误差容限，默认为预设的最大误差值 MAX_ERROR
        train_parser.add_argument(
            "--max-error",
            type=float,
            default=MAX_ERROR,
            help=(
                f"Maximum error tolerance. Defaults to {MAX_ERROR}. This flag should be avoided, use at your own risk."
            ),
        )
        # 添加命令行参数，指定是否创建新的 TensorFlow 权重，默认为 False
        train_parser.add_argument(
            "--new-weights",
            action="store_true",
            help="Optional flag to create new TensorFlow weights, even if they already exist.",
        )
        # 添加命令行参数，指定是否不要打开 PR 来转换权重，默认为 False
        train_parser.add_argument(
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        # 添加命令行参数，指定是否直接将权重推送到 `main` 分支（需要权限）
        train_parser.add_argument(
            "--push",
            action="store_true",
            help="Optional flag to push the weights directly to `main` (requires permissions)",
        )
        # 添加命令行参数，指定额外的提交说明，用于在打开 PR 时使用（例如标记所有者）
        train_parser.add_argument(
            "--extra-commit-description",
            type=str,
            default="",
            help="Optional additional commit description to use when opening a PR (e.g. to tag the owner).",
        )
        # 添加命令行参数，指定覆盖自动检测器的模型类别
        train_parser.add_argument(
            "--override-model-class",
            type=str,
            default=None,
            help="If you think you know better than the auto-detector, you can specify the model class here. "
            "Can be either an AutoModel class or a specific model class like BertForSequenceClassification.",
        )
        # 设置将要执行的函数，默认为 convert_command_factory
        train_parser.set_defaults(func=convert_command_factory)

    @staticmethod
    def find_pt_tf_differences(pt_outputs, tf_outputs):
        """
        Compares the TensorFlow and PyTorch outputs, returning a dictionary with all tensor differences.
        """
        # 1. All output attributes must be the same
        # 获取PyTorch输出和TensorFlow输出的属性集合
        pt_out_attrs = set(pt_outputs.keys())
        tf_out_attrs = set(tf_outputs.keys())
        # 如果属性集合不相等，则抛出ValueError异常
        if pt_out_attrs != tf_out_attrs:
            raise ValueError(
                f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
                f" {tf_out_attrs})"
            )

        # 2. For each output attribute, computes the difference
        # 对于每个输出属性，计算其差异
        def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):
            # 如果当前属性是一个张量，则是叶子节点，进行比较。否则，递归深入，保持属性名称。
            if isinstance(pt_out, torch.Tensor):
                # 计算张量之间的差异
                tensor_difference = np.max(np.abs(pt_out.numpy() - tf_out.numpy()))
                differences[attr_name] = tensor_difference
            else:
                root_name = attr_name
                for i, pt_item in enumerate(pt_out):
                    # 如果是命名属性，保持名称。否则，只保留索引。
                    if isinstance(pt_item, str):
                        branch_name = root_name + pt_item
                        tf_item = tf_out[pt_item]
                        pt_item = pt_out[pt_item]
                    else:
                        branch_name = root_name + f"[{i}]"
                        tf_item = tf_out[i]
                    # 递归计算差异
                    differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

            return differences

        return _find_pt_tf_differences(pt_outputs, tf_outputs, {})

    def __init__(
        self,
        model_name: str,
        local_dir: str,
        max_error: float,
        new_weights: bool,
        no_pr: bool,
        push: bool,
        extra_commit_description: str,
        override_model_class: str,
        *args,
    ):
        # 初始化日志记录器和各种属性
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        # 如果未提供本地目录，则使用默认临时目录
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._max_error = max_error
        self._new_weights = new_weights
        self._no_pr = no_pr
        self._push = push
        self._extra_commit_description = extra_commit_description
        self._override_model_class = override_model_class
```  
```