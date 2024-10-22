# `.\diffusers\pipelines\onnx_utils.py`

```py
# coding=utf-8  # 指定源代码的编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # HuggingFace Inc. 团队的版权声明
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.  # NVIDIA Corporation 的版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 说明该文件根据 Apache 许可证进行授权
# you may not use this file except in compliance with the License.  # 使用文件的条件
# You may obtain a copy of the License at  # 获取许可证的链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的网址
#
# Unless required by applicable law or agreed to in writing, software  # 免责声明，表示无任何担保
# distributed under the License is distributed on an "AS IS" BASIS,  # 文件按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 指定权限和限制
# limitations under the License.  # 许可证下的限制

import os  # 导入操作系统功能模块
import shutil  # 导入高级文件操作模块
from pathlib import Path  # 导入路径操作类
from typing import Optional, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库，用于数值计算
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型
from huggingface_hub.utils import validate_hf_hub_args  # 验证 Hugging Face Hub 参数

from ..utils import ONNX_EXTERNAL_WEIGHTS_NAME, ONNX_WEIGHTS_NAME, is_onnx_available, logging  # 导入实用工具

if is_onnx_available():  # 检查 ONNX 是否可用
    import onnxruntime as ort  # 导入 ONNX Runtime 库

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

ORT_TO_NP_TYPE = {  # 创建一个字典，将 ONNX 类型映射到 NumPy 类型
    "tensor(bool)": np.bool_,  # 布尔类型
    "tensor(int8)": np.int8,  # 8位整数类型
    "tensor(uint8)": np.uint8,  # 无符号8位整数类型
    "tensor(int16)": np.int16,  # 16位整数类型
    "tensor(uint16)": np.uint16,  # 无符号16位整数类型
    "tensor(int32)": np.int32,  # 32位整数类型
    "tensor(uint32)": np.uint32,  # 无符号32位整数类型
    "tensor(int64)": np.int64,  # 64位整数类型
    "tensor(uint64)": np.uint64,  # 无符号64位整数类型
    "tensor(float16)": np.float16,  # 16位浮点数类型
    "tensor(float)": np.float32,  # 32位浮点数类型
    "tensor(double)": np.float64,  # 64位浮点数类型
}

class OnnxRuntimeModel:  # 定义 OnnxRuntimeModel 类
    def __init__(self, model=None, **kwargs):  # 构造函数，初始化模型和参数
        logger.info("`diffusers.OnnxRuntimeModel` is experimental and might change in the future.")  # 记录信息，说明该模型为实验性
        self.model = model  # 保存模型实例
        self.model_save_dir = kwargs.get("model_save_dir", None)  # 获取模型保存目录
        self.latest_model_name = kwargs.get("latest_model_name", ONNX_WEIGHTS_NAME)  # 获取最新模型名称

    def __call__(self, **kwargs):  # 定义调用函数，使类实例可调用
        inputs = {k: np.array(v) for k, v in kwargs.items()}  # 将输入参数转换为 NumPy 数组
        return self.model.run(None, inputs)  # 运行模型并返回结果

    @staticmethod  # 静态方法，不需要实例化
    def load_model(path: Union[str, Path], provider=None, sess_options=None):  # 加载 ONNX 模型
        """
        Loads an ONNX Inference session with an ExecutionProvider. Default provider is `CPUExecutionProvider`

        Arguments:
            path (`str` or `Path`):  # 加载模型的路径
                Directory from which to load
            provider(`str`, *optional*):  # 执行提供者，可选参数
                Onnxruntime execution provider to use for loading the model, defaults to `CPUExecutionProvider`
        """
        if provider is None:  # 检查提供者是否为空
            logger.info("No onnxruntime provider specified, using CPUExecutionProvider")  # 记录信息，使用默认提供者
            provider = "CPUExecutionProvider"  # 设置为默认提供者

        return ort.InferenceSession(path, providers=[provider], sess_options=sess_options)  # 创建并返回推理会话
    # 定义一个保存预训练模型及其配置文件的方法
        def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
            """
            将模型及其配置文件保存到指定目录，以便可以通过
            [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] 类方法重新加载。始终保存
            latest_model_name。
    
            参数：
                save_directory (`str` 或 `Path`):
                    保存模型文件的目录。
                file_name(`str`, *可选*):
                    将默认模型文件名从 `"model.onnx"` 替换为 `file_name`。这允许使用不同的名称保存模型。
            """
            # 根据提供的文件名或默认模型名称设置模型文件名
            model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
    
            # 创建源路径，指向最新模型的保存目录
            src_path = self.model_save_dir.joinpath(self.latest_model_name)
            # 创建目标路径，指向保存目录和模型文件名
            dst_path = Path(save_directory).joinpath(model_file_name)
            try:
                # 尝试复制模型文件到目标路径
                shutil.copyfile(src_path, dst_path)
            except shutil.SameFileError:
                # 如果源文件和目标文件相同，则忽略错误
                pass
    
            # 复制外部权重（适用于大于2GB的模型）
            src_path = self.model_save_dir.joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
            # 检查外部权重文件是否存在
            if src_path.exists():
                # 创建目标路径指向外部权重文件
                dst_path = Path(save_directory).joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
                try:
                    # 尝试复制外部权重文件到目标路径
                    shutil.copyfile(src_path, dst_path)
                except shutil.SameFileError:
                    # 如果源文件和目标文件相同，则忽略错误
                    pass
    
        # 定义保存预训练模型到指定目录的方法
        def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            **kwargs,
        ):
            """
            将模型保存到指定目录，以便可以通过
            [`~OnnxModel.from_pretrained`] 类方法重新加载。:
    
            参数：
                save_directory (`str` 或 `os.PathLike`):
                    要保存的目录。如果不存在，则会创建。
            """
            # 检查提供的路径是否是文件，如果是，则记录错误并返回
            if os.path.isfile(save_directory):
                logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
                return
    
            # 创建保存目录，如果已存在则不报错
            os.makedirs(save_directory, exist_ok=True)
    
            # 保存模型权重/文件
            self._save_pretrained(save_directory, **kwargs)
    
        # 定义一个类方法，从预训练模型加载模型
        @classmethod
        @validate_hf_hub_args
        def _from_pretrained(
            cls,
            model_id: Union[str, Path],
            token: Optional[Union[bool, str, None]] = None,
            revision: Optional[Union[str, None]] = None,
            force_download: bool = False,
            cache_dir: Optional[str] = None,
            file_name: Optional[str] = None,
            provider: Optional[str] = None,
            sess_options: Optional["ort.SessionOptions"] = None,
            **kwargs,
    ):
        """
        从目录或 HF Hub 加载模型。

        参数：
            model_id (`str` 或 `Path`):
                要加载的目录
            token (`str` 或 `bool`):
                加载私有或受限库模型所需
            revision (`str`):
                具体的模型版本，可以是分支名、标签名或提交 ID
            cache_dir (`Union[str, Path]`, *可选*):
                下载的预训练模型配置应缓存的目录路径，如果不使用标准缓存。
            force_download (`bool`, *可选*, 默认值为 `False`):
                是否强制（重新）下载模型权重和配置文件，覆盖已存在的缓存版本。
            file_name(`str`):
                将默认模型文件名从 `"model.onnx"` 替换为 `file_name`。这允许从同一库或目录加载不同的模型文件。
            provider(`str`):
                ONNX 运行时提供者，例如 `CPUExecutionProvider` 或 `CUDAExecutionProvider`。
            kwargs (`Dict`, *可选*):
                初始化时将传递给模型的关键字参数
        """
        # 根据 file_name 判断模型文件名，如果为 None 则使用默认的 ONNX_WEIGHTS_NAME
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        # 检查 model_id 是否为目录
        if os.path.isdir(model_id):
            # 从本地目录加载模型
            model = OnnxRuntimeModel.load_model(
                # 使用给定模型文件名和提供者加载模型
                Path(model_id, model_file_name).as_posix(), provider=provider, sess_options=sess_options
            )
            # 将模型保存目录加入 kwargs
            kwargs["model_save_dir"] = Path(model_id)
        # 如果 model_id 不是目录，则从 hub 加载模型
        else:
            # 下载模型
            model_cache_path = hf_hub_download(
                # 从 HF Hub 下载模型，使用提供的参数
                repo_id=model_id,
                filename=model_file_name,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            # 将模型缓存路径的父目录加入 kwargs
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            # 将下载的最新模型名称加入 kwargs
            kwargs["latest_model_name"] = Path(model_cache_path).name
            # 从缓存路径加载模型
            model = OnnxRuntimeModel.load_model(model_cache_path, provider=provider, sess_options=sess_options)
        # 返回模型实例和关键字参数
        return cls(model=model, **kwargs)

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        # 要加载的模型 ID，可以是字符串或路径
        model_id: Union[str, Path],
        # 是否强制下载模型，默认值为 True
        force_download: bool = True,
        # 用于私有库的访问令牌，可选
        token: Optional[str] = None,
        # 缓存目录，可选
        cache_dir: Optional[str] = None,
        # 其他模型关键字参数
        **model_kwargs,
    # 结束函数定义
        ):
            # 初始化修订版本为 None
            revision = None
            # 如果模型 ID 以 "@" 分隔成两个部分，则分开赋值
            if len(str(model_id).split("@")) == 2:
                # 分割模型 ID 和修订版本
                model_id, revision = model_id.split("@")
    
            # 从预训练模型中加载，返回加载的模型
            return cls._from_pretrained(
                # 指定模型 ID
                model_id=model_id,
                # 指定修订版本
                revision=revision,
                # 指定缓存目录
                cache_dir=cache_dir,
                # 指定是否强制下载
                force_download=force_download,
                # 提供访问令牌
                token=token,
                # 传递额外的模型参数
                **model_kwargs,
            )
```