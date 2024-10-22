# `.\diffusers\models\modeling_flax_utils.py`

```py
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权声明，表示文件由 HuggingFace Inc. 团队拥有
# Copyright 2024 The HuggingFace Inc. team.
#
# 根据 Apache 2.0 许可证许可本文件，使用时需遵循该许可证
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵循许可证的前提下使用此文件
# you may not use this file except in compliance with the License.
# 可以在此网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，软件按“原样”提供
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定语言管理权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入操作系统模块
import os
# 导入反序列化错误类
from pickle import UnpicklingError
# 导入类型提示所需的 Any, Dict, Union 类型
from typing import Any, Dict, Union

# 导入 JAX 库及其 NumPy 子模块
import jax
import jax.numpy as jnp
# 导入 msgpack 异常
import msgpack.exceptions
# 从 flax 库导入冻结字典及其解冻方法
from flax.core.frozen_dict import FrozenDict, unfreeze
# 从 flax 库导入字节序列化与反序列化方法
from flax.serialization import from_bytes, to_bytes
# 从 flax 库导入字典扁平化与解扁平化方法
from flax.traverse_util import flatten_dict, unflatten_dict
# 从 huggingface_hub 导入创建仓库和下载方法
from huggingface_hub import create_repo, hf_hub_download
# 导入 huggingface_hub 的一些异常类
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
# 导入请求库中的 HTTP 错误类
from requests import HTTPError

# 导入当前包的版本和 PyTorch 可用性检查
from .. import __version__, is_torch_available
# 导入工具函数和常量
from ..utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    WEIGHTS_NAME,
    PushToHubMixin,
    logging,
)
# 从模型转换工具中导入 PyTorch 状态字典转换为 Flax 的方法
from .modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 FlaxModelMixin 类，继承自 PushToHubMixin
class FlaxModelMixin(PushToHubMixin):
    r"""
    所有 Flax 模型的基类。

    [`FlaxModelMixin`] 负责存储模型配置，并提供加载、下载和保存模型的方法。

        - **config_name** ([`str`]) -- 调用 [`~FlaxModelMixin.save_pretrained`] 时保存模型的文件名。
    """

    # 配置文件名常量，指定模型配置文件名
    config_name = CONFIG_NAME
    # 自动保存的参数列表
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    # Flax 内部参数列表
    _flax_internal_args = ["name", "parent", "dtype"]

    # 类方法，用于根据配置创建模型实例
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        模型初始化所需的上下文管理器在这里定义。
        """
        # 返回类的实例，传入配置和其他参数
        return cls(config, **kwargs)
    # 定义一个方法，将给定参数的浮点值转换为指定的数据类型
    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None) -> Any:
        # 帮助方法，用于将给定 PyTree 中的浮点值转换为给定的数据类型
        """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """
    
        # 条件转换函数，判断参数类型并执行转换
        # taken from https://github.com/deepmind/jmp/blob/3a8318abc3292be38582794dbf7b094e6583b192/jmp/_src/policy.py#L27
        def conditional_cast(param):
            # 检查参数是否为浮点类型的数组
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                # 将数组转换为指定的数据类型
                param = param.astype(dtype)
            # 返回转换后的参数
            return param
    
        # 如果没有提供掩码，则对所有参数应用条件转换
        if mask is None:
            # 使用 jax.tree_map 对参数树中的每个元素应用条件转换
            return jax.tree_map(conditional_cast, params)
    
        # 扁平化参数字典以便处理
        flat_params = flatten_dict(params)
        # 扁平化掩码，并丢弃结构信息
        flat_mask, _ = jax.tree_flatten(mask)
    
        # 遍历掩码和参数的扁平化键
        for masked, key in zip(flat_mask, flat_params.keys()):
            # 如果掩码为真，则执行转换
            if masked:
                param = flat_params[key]
                # 将转换后的参数重新存储回扁平化参数字典中
                flat_params[key] = conditional_cast(param)
    
        # 将扁平化的参数字典转换回原始结构
        return unflatten_dict(flat_params)
    
    # 定义一个方法，将参数转换为 bfloat16 类型
    def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        # 将浮点参数转换为 jax.numpy.bfloat16，返回新的参数树
        r"""
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.
    
        This method can be used on a TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.
    
        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans. It should be `True`
                for params you want to cast, and `False` for those you want to skip.
    
        Examples:
    
        ```python
        >>> from diffusers import FlaxUNet2DConditionModel
    
        >>> # load model
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> params = model.to_bf16(params)
        >>> # If you don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util
    
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> flat_params = traverse_util.flatten_dict(params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> params = model.to_bf16(params, mask)
        ```py"""
        # 调用内部方法，将参数转换为 bfloat16 类型
        return self._cast_floating_to(params, jnp.bfloat16, mask)
    # 将模型参数转换为浮点32位格式的方法
    def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r""" 
        将浮点数 `params` 转换为 `jax.numpy.float32`。此方法可用于显式将模型参数转换为 fp32 精度。
        返回一个新的 `params` 树，而不在原地转换 `params`。
    
        参数：
            params (`Union[Dict, FrozenDict]`):
                模型参数的 `PyTree`。
            mask (`Union[Dict, FrozenDict]`):
                与 `params` 树具有相同结构的 `PyTree`。叶子应为布尔值。应为要转换的参数设置为 `True`，为要跳过的参数设置为 `False`。
    
        示例：
    
        ```python
        >>> from diffusers import FlaxUNet2DConditionModel
    
        >>> # 从 huggingface.co 下载模型和配置
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # 默认情况下，模型参数将是 fp32，为了说明此方法的用法，
        >>> # 我们将首先转换为 fp16，然后再转换回 fp32
        >>> params = model.to_f16(params)
        >>> # 现在转换回 fp32
        >>> params = model.to_fp32(params)
        ```py"""
        # 调用私有方法，将参数转换为浮点32格式，传入参数、目标类型和掩码
        return self._cast_floating_to(params, jnp.float32, mask)
    # 定义一个将浮点数参数转换为 float16 的方法，接受参数字典和可选的掩码
    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        将浮点数 `params` 转换为 `jax.numpy.float16`。该方法返回一个新的 `params` 树，不会在原地转换 `params`。

        此方法可在 GPU 上使用，显式地将模型参数转换为 float16 精度，以进行全半精度训练，或将权重保存为 float16 以便推理，从而节省内存并提高速度。

        参数：
            params (`Union[Dict, FrozenDict]`):
                一个模型参数的 `PyTree`。
            mask (`Union[Dict, FrozenDict]`):
                具有与 `params` 树相同结构的 `PyTree`。叶子节点应为布尔值。对于要转换的参数，应为 `True`，而要跳过的参数应为 `False`。

        示例：

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # 加载模型
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # 默认情况下，模型参数将为 fp32，转换为 float16
        >>> params = model.to_fp16(params)
        >>> # 如果你不想转换某些参数（例如层归一化的偏差和尺度）
        >>> # 则可以按如下方式传递掩码
        >>> from flax import traverse_util

        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> flat_params = traverse_util.flatten_dict(params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> params = model.to_fp16(params, mask)
        ```py"""
        # 调用内部方法将参数转换为 float16 类型，传入可选的掩码
        return self._cast_floating_to(params, jnp.float16, mask)

    # 定义一个初始化权重的方法，接受随机数生成器作为参数，返回字典
    def init_weights(self, rng: jax.Array) -> Dict:
        # 抛出未实现的错误，提示此方法需要被实现
        raise NotImplementedError(f"init_weights method has to be implemented for {self}")

    # 定义一个类方法用于从预训练模型加载参数，接受模型名称或路径等参数
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs,
    # 定义一个保存预训练模型的方法，接受保存目录和参数等
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        params: Union[Dict, FrozenDict],
        is_main_process: bool = True,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        保存模型及其配置文件到指定目录，以便使用
        [`~FlaxModelMixin.from_pretrained`] 类方法重新加载。

        参数：
            save_directory (`str` 或 `os.PathLike`):
                保存模型及其配置文件的目录。如果目录不存在，将会被创建。
            params (`Union[Dict, FrozenDict]`):
                模型参数的 `PyTree`。
            is_main_process (`bool`, *可选*, 默认为 `True`):
                调用此函数的进程是否为主进程。在分布式训练中非常有用，
                需要在所有进程上调用此函数。此时，仅在主进程上将 `is_main_process=True`
                以避免竞争条件。
            push_to_hub (`bool`, *可选*, 默认为 `False`):
                保存模型后是否将其推送到 Hugging Face 模型库。可以使用 `repo_id`
                指定要推送到的库（默认为 `save_directory` 中的名称）。
            kwargs (`Dict[str, Any]`, *可选*):
                额外的关键字参数，将传递给 [`~utils.PushToHubMixin.push_to_hub`] 方法。
        """
        # 检查提供的路径是否为文件，如果是则记录错误并返回
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # 如果目录不存在则创建该目录
        os.makedirs(save_directory, exist_ok=True)

        # 如果需要推送到模型库
        if push_to_hub:
            # 从关键字参数中弹出提交信息，如果没有则为 None
            commit_message = kwargs.pop("commit_message", None)
            # 从关键字参数中弹出隐私设置，默认为 False
            private = kwargs.pop("private", False)
            # 从关键字参数中弹出创建 PR 的设置，默认为 False
            create_pr = kwargs.pop("create_pr", False)
            # 从关键字参数中弹出 token，默认为 None
            token = kwargs.pop("token", None)
            # 从关键字参数中弹出 repo_id，默认为 save_directory 的最后一部分
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            # 创建库并获取 repo_id
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        # 将当前对象赋值给 model_to_save
        model_to_save = self

        # 将模型架构附加到配置中
        # 保存配置
        if is_main_process:
            # 如果是主进程，保存模型配置到指定目录
            model_to_save.save_config(save_directory)

        # 保存模型的输出文件路径
        output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)
        # 以二进制写入模式打开模型文件
        with open(output_model_file, "wb") as f:
            # 将模型参数转换为字节
            model_bytes = to_bytes(params)
            # 将字节数据写入文件
            f.write(model_bytes)

        # 记录模型权重保存的路径信息
        logger.info(f"Model weights saved in {output_model_file}")

        # 如果需要推送到模型库
        if push_to_hub:
            # 调用上传文件夹的方法，将模型文件夹推送到模型库
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
```