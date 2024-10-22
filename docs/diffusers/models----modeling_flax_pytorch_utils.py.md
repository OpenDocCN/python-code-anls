# `.\diffusers\models\modeling_flax_pytorch_utils.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 版权信息，表明版权所有者

# Licensed under the Apache License, Version 2.0 (the "License");  # 说明该文件根据 Apache 2.0 许可证发布
# you may not use this file except in compliance with the License.  # 说明只能在遵守许可证的情况下使用此文件
# You may obtain a copy of the License at  # 提供获取许可证的地址
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的具体链接
#
# Unless required by applicable law or agreed to in writing, software  # 免责声明，除非另有规定或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 说明软件是按“现状”提供的
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 没有任何形式的明示或暗示的保证
# See the License for the specific language governing permissions and  # 指向许可证以获取具体条款
# limitations under the License.  # 以及使用限制的说明
"""PyTorch - Flax general utilities."""  # 文档字符串，描述该模块的功能

import re  # 导入正则表达式模块

import jax.numpy as jnp  # 导入 JAX 的 NumPy 库，并重命名为 jnp
from flax.traverse_util import flatten_dict, unflatten_dict  # 从 flax 导入字典扁平化和还原的工具
from jax.random import PRNGKey  # 从 jax 导入伪随机数生成器的键

from ..utils import logging  # 从父目录导入 logging 模块

logger = logging.get_logger(__name__)  # 创建一个日志记录器，记录当前模块的信息

def rename_key(key):  # 定义一个函数，用于重命名键
    regex = r"\w+[.]\d+"  # 定义一个正则表达式，匹配包含点号和数字的字符串
    pats = re.findall(regex, key)  # 使用正则表达式查找所有匹配的字符串
    for pat in pats:  # 遍历所有找到的匹配
        key = key.replace(pat, "_".join(pat.split(".")))  # 将匹配的字符串中的点替换为下划线
    return key  # 返回修改后的键

#####################
# PyTorch => Flax #
#####################  # 注释区分 PyTorch 到 Flax 的转换部分

# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69  # 说明该函数的来源链接
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py  # 说明该函数的另一来源链接
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):  # 定义函数，重命名权重并在必要时改变张量形状
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""  # 文档字符串，说明函数功能
    # conv norm or layer norm  # 注释，说明即将处理的内容是卷积归一化或层归一化
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)  # 将原键的最后一个元素替换为 "scale"

    # rename attention layers  # 注释，说明将重命名注意力层
    if len(pt_tuple_key) > 1:  # 如果元组键的长度大于 1
        for rename_from, rename_to in (  # 遍历重命名映射的元组
            ("to_out_0", "proj_attn"),  # 旧名称到新名称的映射
            ("to_k", "key"),  # 旧名称到新名称的映射
            ("to_v", "value"),  # 旧名称到新名称的映射
            ("to_q", "query"),  # 旧名称到新名称的映射
        ):
            if pt_tuple_key[-2] == rename_from:  # 如果倒数第二个元素匹配旧名称
                weight_name = pt_tuple_key[-1]  # 获取最后一个元素作为权重名称
                weight_name = "kernel" if weight_name == "weight" else weight_name  # 如果权重名称是 "weight"，则改为 "kernel"
                renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)  # 生成新的键
                if renamed_pt_tuple_key in random_flax_state_dict:  # 如果新键存在于状态字典中
                    assert random_flax_state_dict[renamed_pt_tuple_key].shape == pt_tensor.T.shape  # 断言新键的形状与转置的张量形状相同
                    return renamed_pt_tuple_key, pt_tensor.T  # 返回新的键和转置的张量

    if (  # 检查是否满足以下条件
        any("norm" in str_ for str_ in pt_tuple_key)  # 如果键中任何部分包含 "norm"
        and (pt_tuple_key[-1] == "bias")  # 并且最后一个元素是 "bias"
        and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)  # 并且去掉最后一个元素后加 "bias" 的键不在状态字典中
        and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)  # 并且去掉最后一个元素后加 "scale" 的键在状态字典中
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)  # 将键的最后一个元素替换为 "scale"
        return renamed_pt_tuple_key, pt_tensor  # 返回新的键和原张量

    elif pt_tuple_key[-1] in ["weight", "gamma"] and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict:  # 如果最后一个元素是 "weight" 或 "gamma" 并且去掉最后一个元素后加 "scale" 的键在状态字典中
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)  # 将键的最后一个元素替换为 "scale"
        return renamed_pt_tuple_key, pt_tensor  # 返回新的键和原张量

    # embedding  # 注释，表明此处将处理嵌入相关的内容
    # 检查元组的最后一个元素是否为 "weight"，并且在字典中查找相应的 "embedding" 键
    if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
        # 将元组的最后一个元素替换为 "embedding"
        pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        # 返回更新后的元组键和张量
        return renamed_pt_tuple_key, pt_tensor

    # 卷积层处理
    # 更新元组的最后一个元素为 "kernel"
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    # 检查元组的最后一个元素是否为 "weight"，并且张量的维度是否为 4
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        # 转置张量的维度顺序
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        # 返回更新后的元组键和张量
        return renamed_pt_tuple_key, pt_tensor

    # 线性层处理
    # 更新元组的最后一个元素为 "kernel"
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    # 检查元组的最后一个元素是否为 "weight"
    if pt_tuple_key[-1] == "weight":
        # 转置张量
        pt_tensor = pt_tensor.T
        # 返回更新后的元组键和张量
        return renamed_pt_tuple_key, pt_tensor

    # 旧版 PyTorch 层归一化权重处理
    # 更新元组的最后一个元素为 "weight"
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    # 检查元组的最后一个元素是否为 "gamma"
    if pt_tuple_key[-1] == "gamma":
        # 返回更新后的元组键和张量
        return renamed_pt_tuple_key, pt_tensor

    # 旧版 PyTorch 层归一化偏置处理
    # 更新元组的最后一个元素为 "bias"
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    # 检查元组的最后一个元素是否为 "beta"
    if pt_tuple_key[-1] == "beta":
        # 返回更新后的元组键和张量
        return renamed_pt_tuple_key, pt_tensor

    # 如果没有匹配的条件，则返回原始元组键和张量
    return pt_tuple_key, pt_tensor
# 将 PyTorch 的状态字典转换为 Flax 模型的参数字典
def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42):
    # 步骤 1：将 PyTorch 张量转换为 NumPy 数组
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    # 步骤 2：由于模型是无状态的，使用随机种子初始化 Flax 参数
    random_flax_params = flax_model.init_weights(PRNGKey(init_key))

    # 将随机生成的 Flax 参数展平为字典形式
    random_flax_state_dict = flatten_dict(random_flax_params)
    # 初始化一个空的 Flax 状态字典
    flax_state_dict = {}

    # 需要修改一些参数名称以匹配 Flax 的命名
    for pt_key, pt_tensor in pt_state_dict.items():
        # 重命名 PyTorch 的键
        renamed_pt_key = rename_key(pt_key)
        # 将重命名后的键分割成元组形式
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # 正确重命名权重参数并调整形状
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict)

        # 检查 Flax 键是否在随机生成的状态字典中
        if flax_key in random_flax_state_dict:
            # 如果形状不匹配，抛出错误
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # 也将意外的权重添加到字典中，以便引发警告
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    # 返回解压缩后的 Flax 状态字典
    return unflatten_dict(flax_state_dict)
```