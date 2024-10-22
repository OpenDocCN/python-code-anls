# `.\diffusers\commands\fp16_safetensors.py`

```py
# 版权声明，指明代码的版权归 HuggingFace 团队所有
# 
# 根据 Apache License 2.0 版本授权，声明用户须遵守该许可
# 用户可以在以下网址获取许可副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律规定或书面同意，软件在“按原样”基础上分发，
# 不提供任何形式的担保或条件
# 请查看许可以了解特定语言的权限和限制

"""
用法示例：
    diffusers-cli fp16_safetensors --ckpt_id=openai/shap-e --fp16 --use_safetensors
"""

# 导入所需模块
import glob  # 用于文件路径匹配
import json  # 用于 JSON 数据解析
import warnings  # 用于发出警告
from argparse import ArgumentParser, Namespace  # 用于命令行参数解析
from importlib import import_module  # 动态导入模块

import huggingface_hub  # Hugging Face Hub 的接口库
import torch  # PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型的函数
from packaging import version  # 用于版本比较

from ..utils import logging  # 导入日志模块
from . import BaseDiffusersCLICommand  # 导入基本 CLI 命令类


def conversion_command_factory(args: Namespace):
    # 根据传入的命令行参数创建转换命令
    if args.use_auth_token:
        # 发出警告，提示 --use_auth_token 参数已弃用
        warnings.warn(
            "The `--use_auth_token` flag is deprecated and will be removed in a future version. Authentication is now"
            " handled automatically if user is logged in."
        )
    # 返回 FP16SafetensorsCommand 的实例
    return FP16SafetensorsCommand(args.ckpt_id, args.fp16, args.use_safetensors)


class FP16SafetensorsCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 注册子命令 fp16_safetensors
        conversion_parser = parser.add_parser("fp16_safetensors")
        # 添加 ckpt_id 参数，用于指定检查点的仓库 ID
        conversion_parser.add_argument(
            "--ckpt_id",
            type=str,
            help="Repo id of the checkpoints on which to run the conversion. Example: 'openai/shap-e'.",
        )
        # 添加 fp16 参数，指示是否以 FP16 精度序列化变量
        conversion_parser.add_argument(
            "--fp16", action="store_true", help="If serializing the variables in FP16 precision."
        )
        # 添加 use_safetensors 参数，指示是否以 safetensors 格式序列化
        conversion_parser.add_argument(
            "--use_safetensors", action="store_true", help="If serializing in the safetensors format."
        )
        # 添加 use_auth_token 参数，用于处理私有可见性的检查点
        conversion_parser.add_argument(
            "--use_auth_token",
            action="store_true",
            help="When working with checkpoints having private visibility. When used `huggingface-cli login` needs to be run beforehand.",
        )
        # 设置默认函数为 conversion_command_factory
        conversion_parser.set_defaults(func=conversion_command_factory)

    def __init__(self, ckpt_id: str, fp16: bool, use_safetensors: bool):
        # 初始化命令类，设置日志记录器和参数
        self.logger = logging.get_logger("diffusers-cli/fp16_safetensors")
        # 存储检查点 ID
        self.ckpt_id = ckpt_id
        # 定义本地检查点目录
        self.local_ckpt_dir = f"/tmp/{ckpt_id}"
        # 存储 FP16 精度设置
        self.fp16 = fp16
        # 存储 safetensors 设置
        self.use_safetensors = use_safetensors

        # 检查是否同时未使用 safetensors 和 fp16，若是则抛出异常
        if not self.use_safetensors and not self.fp16:
            raise NotImplementedError(
                "When `use_safetensors` and `fp16` both are False, then this command is of no use."
            )
    # 定义运行方法
    def run(self):
        # 检查 huggingface_hub 版本是否低于 0.9.0
        if version.parse(huggingface_hub.__version__) < version.parse("0.9.0"):
            # 如果版本低于要求，抛出导入错误
            raise ImportError(
                "The huggingface_hub version must be >= 0.9.0 to use this command. Please update your huggingface_hub"
                " installation."
            )
        else:
            # 从 huggingface_hub 导入创建提交的函数
            from huggingface_hub import create_commit
            # 从 huggingface_hub 导入提交操作类
            from huggingface_hub._commit_api import CommitOperationAdd
    
        # 下载模型索引文件
        model_index = hf_hub_download(repo_id=self.ckpt_id, filename="model_index.json")
        # 打开模型索引文件并读取内容
        with open(model_index, "r") as f:
            # 从 JSON 中提取管道类名称
            pipeline_class_name = json.load(f)["_class_name"]
        # 动态导入对应的管道类
        pipeline_class = getattr(import_module("diffusers"), pipeline_class_name)
        # 记录导入的管道类名称
        self.logger.info(f"Pipeline class imported: {pipeline_class_name}.")
    
        # 加载适当的管道
        pipeline = pipeline_class.from_pretrained(
            self.ckpt_id, torch_dtype=torch.float16 if self.fp16 else torch.float32
        )
        # 将管道保存到本地目录
        pipeline.save_pretrained(
            self.local_ckpt_dir,
            safe_serialization=True if self.use_safetensors else False,
            variant="fp16" if self.fp16 else None,
        )
        # 记录管道保存的本地目录
        self.logger.info(f"Pipeline locally saved to {self.local_ckpt_dir}.")
    
        # 获取所有的路径
        if self.fp16:
            # 获取所有 FP16 文件的路径
            modified_paths = glob.glob(f"{self.local_ckpt_dir}/*/*.fp16.*")
        elif self.use_safetensors:
            # 获取所有 Safetensors 文件的路径
            modified_paths = glob.glob(f"{self.local_ckpt_dir}/*/*.safetensors")
    
        # 准备提交请求
        commit_message = f"Serialize variables with FP16: {self.fp16} and safetensors: {self.use_safetensors}."
        operations = []
        # 遍历修改过的路径，准备提交操作
        for path in modified_paths:
            operations.append(CommitOperationAdd(path_in_repo="/".join(path.split("/")[4:]), path_or_fileobj=path))
    
        # 打开提交请求
        commit_description = (
            "Variables converted by the [`diffusers`' `fp16_safetensors`"
            " CLI](https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/fp16_safetensors.py)."
        )
        # 创建提交请求并获取其 URL
        hub_pr_url = create_commit(
            repo_id=self.ckpt_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            repo_type="model",
            create_pr=True,
        ).pr_url
        # 记录提交请求的 URL
        self.logger.info(f"PR created here: {hub_pr_url}.")
```