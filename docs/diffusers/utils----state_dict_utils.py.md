# `.\diffusers\utils\state_dict_utils.py`

```py
# 版权声明，指定版权信息及保留权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，指定使用文件的条件和限制
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 许可地址，提供获取许可的链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 声明在适用情况下，软件按“原样”分发，且没有任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可以了解特定权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
"""
状态字典工具：用于轻松转换状态字典的实用方法
"""

# 导入枚举模块
import enum

# 从日志模块导入获取记录器的函数
from .logging import get_logger

# 创建一个记录器实例，用于当前模块的日志记录
logger = get_logger(__name__)

# 定义状态字典类型的枚举类
class StateDictType(enum.Enum):
    """
    用于转换状态字典时使用的模式。
    """

    # 指定不同的状态字典类型
    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    DIFFUSERS = "diffusers"

# 定义 Unet 到 Diffusers 的映射，因为它使用不同的输出键与文本编码器
# 例如：to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    # 映射 Unet 的上输出到 Diffusers 的对应输出
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
    ".to_out.lora_magnitude_vector": ".to_out.0.lora_magnitude_vector",
}

# 定义 Diffusers 到 PEFT 的映射
DIFFUSERS_TO_PEFT = {
    # 映射 Diffusers 的层到 PEFT 的对应层
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
    "text_projection.lora.down.weight": "text_projection.lora_A.weight",
    "text_projection.lora.up.weight": "text_projection.lora_B.weight",
}

# 定义 Diffusers_old 到 PEFT 的映射
DIFFUSERS_OLD_TO_PEFT = {
    # 映射旧版本 Diffusers 的层到 PEFT 的对应层
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

# 定义 PEFT 到 Diffusers 的映射
PEFT_TO_DIFFUSERS = {
    # 映射 PEFT 的层到 Diffusers 的对应层
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    # 将 lora_B 关联到 lora_linear_layer 的上层
        ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
        # 将 lora_A 关联到 lora_linear_layer 的下层
        ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
        # 将 out_proj 的 lora_B 关联到 lora_linear_layer 的上层
        ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
        # 将 out_proj 的 lora_A 关联到 lora_linear_layer 的下层
        ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
        # 将 to_k 的 lora_A 关联到 lora 的下层
        "to_k.lora_A": "to_k.lora.down",
        # 将 to_k 的 lora_B 关联到 lora 的上层
        "to_k.lora_B": "to_k.lora.up",
        # 将 to_q 的 lora_A 关联到 lora 的下层
        "to_q.lora_A": "to_q.lora.down",
        # 将 to_q 的 lora_B 关联到 lora 的上层
        "to_q.lora_B": "to_q.lora.up",
        # 将 to_v 的 lora_A 关联到 lora 的下层
        "to_v.lora_A": "to_v.lora.down",
        # 将 to_v 的 lora_B 关联到 lora 的上层
        "to_v.lora_B": "to_v.lora.up",
        # 将 to_out.0 的 lora_A 关联到 lora 的下层
        "to_out.0.lora_A": "to_out.0.lora.down",
        # 将 to_out.0 的 lora_B 关联到 lora 的上层
        "to_out.0.lora_B": "to_out.0.lora.up",
# 结束前一个代码块
}

# 定义一个字典，将旧的扩散器键映射到新的扩散器键
DIFFUSERS_OLD_TO_DIFFUSERS = {
    # 映射旧的 Q 线性层上升键到新的 Q 线性层上升键
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    # 映射旧的 Q 线性层下降键到新的 Q 线性层下降键
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    # 映射旧的 K 线性层上升键到新的 K 线性层上升键
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    # 映射旧的 K 线性层下降键到新的 K 线性层下降键
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    # 映射旧的 V 线性层上升键到新的 V 线性层上升键
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    # 映射旧的 V 线性层下降键到新的 V 线性层下降键
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    # 映射旧的输出层上升键到新的输出层上升键
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    # 映射旧的输出层下降键到新的输出层下降键
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
    # 映射旧的 K 大小向量键到新的 K 大小向量键
    ".to_k.lora_magnitude_vector": ".k_proj.lora_magnitude_vector",
    # 映射旧的 V 大小向量键到新的 V 大小向量键
    ".to_v.lora_magnitude_vector": ".v_proj.lora_magnitude_vector",
    # 映射旧的 Q 大小向量键到新的 Q 大小向量键
    ".to_q.lora_magnitude_vector": ".q_proj.lora_magnitude_vector",
    # 映射旧的输出层大小向量键到新的输出层大小向量键
    ".to_out.lora_magnitude_vector": ".out_proj.lora_magnitude_vector",
}

# 定义一个字典，将 PEFT 格式映射到 KOHYA_SS 格式
PEFT_TO_KOHYA_SS = {
    # 映射 PEFT 格式中的 A 到 KOHYA_SS 格式中的下降
    "lora_A": "lora_down",
    # 映射 PEFT 格式中的 B 到 KOHYA_SS 格式中的上升
    "lora_B": "lora_up",
    # 这不是一个全面的字典，因为 KOHYA 格式需要替换键中的 `.` 为 `_`，
    # 添加前缀和添加 alpha 值
    # 检查 `convert_state_dict_to_kohya` 以了解更多
}

# 定义一个字典，将状态字典类型映射到相应的 DIFFUSERS 映射
PEFT_STATE_DICT_MAPPINGS = {
    # 映射旧扩散器类型到 PEFT
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    # 映射新扩散器类型到 PEFT
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

# 定义一个字典，将状态字典类型映射到相应的 DIFFUSERS 映射
DIFFUSERS_STATE_DICT_MAPPINGS = {
    # 映射旧扩散器类型到旧扩散器映射
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    # 映射 PEFT 类型到 DIFFUSERS
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

# 定义一个字典，将 PEFT 类型映射到 KOHYA 状态字典映射
KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

# 定义一个字典，指定总是要替换的键模式
KEYS_TO_ALWAYS_REPLACE = {
    # 将处理器的键模式替换为基本形式
    ".processor.": ".",
}

# 定义函数，转换状态字典
def convert_state_dict(state_dict, mapping):
    r"""
    简单地遍历状态字典并用 `mapping` 中的模式替换相应的值。

    参数:
        state_dict (`dict[str, torch.Tensor]`):
            要转换的状态字典。
        mapping (`dict[str, str]`):
            用于转换的映射，映射应为以下结构的字典：
                - 键: 要替换的模式
                - 值: 要替换成的模式

    返回:
        converted_state_dict (`dict`)
            转换后的状态字典。
    """
    # 初始化一个新的转换后的状态字典
    converted_state_dict = {}
    # 遍历输入的状态字典
    for k, v in state_dict.items():
        # 首先，过滤出我们总是想替换的键
        for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
            # 如果当前键中包含模式，则替换
            if pattern in k:
                new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
                k = k.replace(pattern, new_pattern)

        # 遍历映射中的模式
        for pattern in mapping.keys():
            # 如果当前键中包含模式，则替换
            if pattern in k:
                new_pattern = mapping[pattern]
                k = k.replace(pattern, new_pattern)
                break
        # 将转换后的键值对添加到新的字典中
        converted_state_dict[k] = v
    # 返回转换后的状态字典
    return converted_state_dict

# 定义函数，将状态字典转换为 PEFT 格式
def convert_state_dict_to_peft(state_dict, original_type=None, **kwargs):
    r"""
    将状态字典转换为 PEFT 格式，状态字典可以来自旧的扩散器格式（`OLD_DIFFUSERS`）或
    新的扩散器格式（`DIFFUSERS`）。该方法目前仅支持从扩散器旧/新格式到 PEFT 的转换。
    # 参数说明部分
    Args:
        state_dict (`dict[str, torch.Tensor]`):
            # 要转换的状态字典
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            # 状态字典的原始类型，如果未提供，方法将尝试自动推断
            The original type of the state dict, if not provided, the method will try to infer it automatically.
    """
    # 如果原始类型未提供
    if original_type is None:
        # 检查状态字典的键中是否包含“to_out_lora”，用于判断类型
        # 将旧的 diffusers 类型转换为 PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        # 检查状态字典的键中是否包含“lora_linear_layer”
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS
        # 如果无法推断类型，则抛出错误
        else:
            raise ValueError("Could not automatically infer state dict type")

    # 检查推断的原始类型是否在支持的类型映射中
    if original_type not in PEFT_STATE_DICT_MAPPINGS.keys():
        # 如果不支持，则抛出错误
        raise ValueError(f"Original type {original_type} is not supported")

    # 根据原始类型获取对应的映射
    mapping = PEFT_STATE_DICT_MAPPINGS[original_type]
    # 转换状态字典并返回结果
    return convert_state_dict(state_dict, mapping)
# 将状态字典转换为新的 diffusers 格式。状态字典可以来自旧的 diffusers 格式
# (`OLD_DIFFUSERS`)、PEFT 格式 (`PEFT`) 或新的 diffusers 格式 (`DIFFUSERS`)。
# 在最后一种情况下，该方法将返回状态字典本身。
def convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs):
    # 状态字典转换为新格式的文档字符串
    r"""
    Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
    (`OLD_DIFFUSERS`), or PEFT format (`PEFT`) or new diffusers format (`DIFFUSERS`). In the last case the method will
    return the state dict as is.

    The method only supports the conversion from diffusers old, PEFT to diffusers new for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    # 从 kwargs 中获取适配器名称，如果不存在则默认为 None
    peft_adapter_name = kwargs.pop("adapter_name", None)
    # 如果适配器名称不为 None，前面添加一个点
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        # 否则适配器名称为空字符串
        peft_adapter_name = ""

    # 如果没有提供原始类型
    if original_type is None:
        # 检查状态字典的键是否包含 "to_out_lora"，若有则设置原始类型为旧 diffusers
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        # 检查键是否包含以适配器名称为前缀的权重
        elif any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT
        # 检查是否包含 "lora_linear_layer"，如果有则不需要转换，直接返回状态字典
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            # nothing to do
            return state_dict
        # 如果未能推断出原始类型，则引发值错误
        else:
            raise ValueError("Could not automatically infer state dict type")

    # 检查原始类型是否在支持的状态字典映射中
    if original_type not in DIFFUSERS_STATE_DICT_MAPPINGS.keys():
        # 如果不支持，抛出值错误
        raise ValueError(f"Original type {original_type} is not supported")

    # 获取与原始类型相对应的映射
    mapping = DIFFUSERS_STATE_DICT_MAPPINGS[original_type]
    # 使用映射转换状态字典并返回
    return convert_state_dict(state_dict, mapping)


# 将状态字典从 UNet 格式转换为 diffusers 格式，主要通过移除一些键来实现
def convert_unet_state_dict_to_peft(state_dict):
    # 状态字典转换文档字符串
    r"""
    Converts a state dict from UNet format to diffusers format - i.e. by removing some keys
    """
    # 定义 UNet 到 diffusers 的映射
    mapping = UNET_TO_DIFFUSERS
    # 使用映射转换状态字典并返回
    return convert_state_dict(state_dict, mapping)


# 尝试首先将状态字典转换为 PEFT 格式，如果没有检测到有效的 DIFFUSERS LoRA 的 "lora_linear_layer"
# 则尝试专门转换 UNet 状态字典
def convert_all_state_dict_to_peft(state_dict):
    # 状态字典转换为 PEFT 格式的文档字符串
    r"""
    Attempts to first `convert_state_dict_to_peft`, and if it doesn't detect `lora_linear_layer` for a valid
    `DIFFUSERS` LoRA for example, attempts to exclusively convert the Unet `convert_unet_state_dict_to_peft`
    """
    # 尝试转换状态字典为 PEFT 格式
    try:
        peft_dict = convert_state_dict_to_peft(state_dict)
    # 捕获异常并赋值给变量 e
        except Exception as e:
            # 检查异常信息是否为无法自动推断状态字典类型
            if str(e) == "Could not automatically infer state dict type":
                # 将 UNet 状态字典转换为 PEFT 字典
                peft_dict = convert_unet_state_dict_to_peft(state_dict)
            else:
                # 重新抛出未处理的异常
                raise
    
        # 检查 PEFT 字典中是否包含 "lora_A" 或 "lora_B" 的键
        if not any("lora_A" in key or "lora_B" in key for key in peft_dict.keys()):
            # 如果没有，则抛出值错误异常
            raise ValueError("Your LoRA was not converted to PEFT")
    
        # 返回转换后的 PEFT 字典
        return peft_dict
# 定义一个将 PEFT 状态字典转换为 Kohya 格式的函数
def convert_state_dict_to_kohya(state_dict, original_type=None, **kwargs):
    r"""
    将 `PEFT` 状态字典转换为可在 AUTOMATIC1111、ComfyUI、SD.Next、InvokeAI 等中使用的 `Kohya` 格式。
    该方法目前仅支持从 PEFT 到 Kohya 的转换。

    参数:
        state_dict (`dict[str, torch.Tensor]`):
            要转换的状态字典。
        original_type (`StateDictType`, *可选*):
            状态字典的原始类型，如果未提供，方法将尝试自动推断。
        kwargs (`dict`, *args*):
            传递给该方法的附加参数。

            - **adapter_name**: 例如，在 PEFT 的情况下，一些键会被适配器名称预先附加，
                因此需要特殊处理。默认情况下，PEFT 也会在
                `get_peft_model_state_dict` 方法中处理这一点：
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                但我们在这里添加它以防我们不想依赖该方法。
    """
    # 尝试导入 torch 库
    try:
        import torch
    # 如果导入失败，记录错误并引发异常
    except ImportError:
        logger.error("Converting PEFT state dicts to Kohya requires torch to be installed.")
        raise

    # 从 kwargs 中弹出适配器名称，如果没有提供则为 None
    peft_adapter_name = kwargs.pop("adapter_name", None)
    # 如果提供了适配器名称，则在前面加上点
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    # 如果没有适配器名称，则设置为空字符串
    else:
        peft_adapter_name = ""

    # 如果未提供原始类型，则检查状态字典中是否包含特定键
    if original_type is None:
        if any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT

    # 检查原始类型是否在支持的类型映射中
    if original_type not in KOHYA_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    # 使用适当的映射调用 convert_state_dict 函数
    kohya_ss_partial_state_dict = convert_state_dict(state_dict, KOHYA_STATE_DICT_MAPPINGS[StateDictType.PEFT])
    # 创建一个空字典以存储最终的 Kohya 状态字典
    kohya_ss_state_dict = {}

    # 额外逻辑：在所有键中替换头部、alpha 参数的 `.` 为 `_`
    for kohya_key, weight in kohya_ss_partial_state_dict.items():
        # 替换特定的键名
        if "text_encoder_2." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in kohya_key:
            kohya_key = kohya_key.replace("unet", "lora_unet")
        elif "lora_magnitude_vector" in kohya_key:
            kohya_key = kohya_key.replace("lora_magnitude_vector", "dora_scale")

        # 将所有键中的 `.` 替换为 `_`，保留最后两个 `.` 不变
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        # 移除适配器名称，Kohya 不需要这些名称
        kohya_key = kohya_key.replace(peft_adapter_name, "")
        # 将处理后的键及其权重存储到新的字典中
        kohya_ss_state_dict[kohya_key] = weight
        # 如果键名中包含 `lora_down`，则创建相应的 alpha 键
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            # 将 alpha 键的值设置为权重的长度
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))
    # 返回保存的状态字典，通常用于恢复训练或推理的状态
    return kohya_ss_state_dict
```