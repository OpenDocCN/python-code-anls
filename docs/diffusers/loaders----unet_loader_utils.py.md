# `.\diffusers\loaders\unet_loader_utils.py`

```py
# 版权声明，标识本代码的版权所有者及其保留的权利
# 
# 根据 Apache License, Version 2.0 进行许可；除非符合许可条款，否则不得使用此文件
# 可以在以下网址获取许可副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用的法律或书面协议另有规定，否则根据许可分发的软件以 "按现状" 基础提供，
# 不附加任何明示或暗示的担保或条件
# 请参阅许可协议以了解有关许可及其限制的详细信息
import copy  # 导入 copy 模块，用于对象的浅拷贝或深拷贝
from typing import TYPE_CHECKING, Dict, List, Union  # 导入类型注释支持

from ..utils import logging  # 从上级模块导入 logging 功能


if TYPE_CHECKING:
    # 在这里导入以避免循环导入问题
    from ..models import UNet2DConditionModel  # 从上级模块导入 UNet2DConditionModel

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 的命名检查


def _translate_into_actual_layer_name(name):
    """将用户友好的名称（例如 'mid'）转换为实际层名称（例如 'mid_block.attentions.0'）"""
    if name == "mid":
        return "mid_block.attentions.0"  # 如果名称是 'mid'，返回其对应的实际层名

    updown, block, attn = name.split(".")  # 将名称按 '.' 分割成上下文、块和注意力部分

    updown = updown.replace("down", "down_blocks").replace("up", "up_blocks")  # 替换上下文中的 'down' 和 'up'
    block = block.replace("block_", "")  # 去掉块名称中的 'block_' 前缀
    attn = "attentions." + attn  # 将注意力部分格式化为完整名称

    return ".".join((updown, block, attn))  # 将所有部分合并为实际层名并返回


def _maybe_expand_lora_scales(
    unet: "UNet2DConditionModel", weight_scales: List[Union[float, Dict]], default_scale=1.0
):
    # 可能扩展 LoRA 权重比例，接受 UNet 模型和权重比例列表作为参数
    blocks_with_transformer = {
        "down": [i for i, block in enumerate(unet.down_blocks) if hasattr(block, "attentions")],
        # 找到下层块中具有注意力层的块索引
        "up": [i for i, block in enumerate(unet.up_blocks) if hasattr(block, "attentions")]
        # 找到上层块中具有注意力层的块索引
    }
    transformer_per_block = {"down": unet.config.layers_per_block, "up": unet.config.layers_per_block + 1}
    # 创建字典，包含每个块的变换层数量

    expanded_weight_scales = [
        _maybe_expand_lora_scales_for_one_adapter(
            weight_for_adapter,
            blocks_with_transformer,
            transformer_per_block,
            unet.state_dict(),
            default_scale=default_scale,
        )
        # 对每个适配器的权重调用扩展函数，生成扩展后的权重比例列表
        for weight_for_adapter in weight_scales
    ]

    return expanded_weight_scales  # 返回扩展后的权重比例


def _maybe_expand_lora_scales_for_one_adapter(
    scales: Union[float, Dict],
    blocks_with_transformer: Dict[str, int],
    transformer_per_block: Dict[str, int],
    state_dict: None,
    default_scale: float = 1.0,
):
    """
    将输入扩展为更细粒度的字典。以下示例提供了更多细节。

    参数：
        scales (`Union[float, Dict]`):
            要扩展的比例字典。
        blocks_with_transformer (`Dict[str, int]`):
            包含 'up' 和 'down' 键的字典，显示哪些块具有变换层
        transformer_per_block (`Dict[str, int]`):
            包含 'up' 和 'down' 键的字典，显示每个块的变换层数量

    例如，转换
    ```python
    scales = {"down": 2, "mid": 3, "up": {"block_0": 4, "block_1": [5, 6, 7]}}
```py 
    # 定义一个字典，表示每个方向的块及其对应的编号
        blocks_with_transformer = {"down": [1, 2], "up": [0, 1]}
        # 定义一个字典，表示每个方向的块需要的变换器数量
        transformer_per_block = {"down": 2, "up": 3}
        # 如果 blocks_with_transformer 的键不是 "down" 和 "up"，则抛出错误
        if sorted(blocks_with_transformer.keys()) != ["down", "up"]:
            raise ValueError("blocks_with_transformer needs to be a dict with keys `'down' and `'up'`")
        # 如果 transformer_per_block 的键不是 "down" 和 "up"，则抛出错误
        if sorted(transformer_per_block.keys()) != ["down", "up"]:
            raise ValueError("transformer_per_block needs to be a dict with keys `'down' and `'up'`")
        # 如果 scales 不是字典类型，则直接返回其值
        if not isinstance(scales, dict):
            # don't expand if scales is a single number
            return scales
        # 复制 scales 的深拷贝，以避免修改原始数据
        scales = copy.deepcopy(scales)
        # 如果 scales 中没有 "mid"，则赋予默认比例
        if "mid" not in scales:
            scales["mid"] = default_scale
        # 如果 "mid" 是列表类型且仅有一个元素，则将其转换为该元素
        elif isinstance(scales["mid"], list):
            if len(scales["mid"]) == 1:
                scales["mid"] = scales["mid"][0]
            # 如果 "mid" 列表元素个数不为 1，则抛出错误
            else:
                raise ValueError(f"Expected 1 scales for mid, got {len(scales['mid'])}.")
        # 遍历方向 "up" 和 "down"
        for updown in ["up", "down"]:
            # 如果当前方向不在 scales 中，则赋予默认比例
            if updown not in scales:
                scales[updown] = default_scale
            # 如果当前方向的比例不是字典，则将其转换为字典格式
            if not isinstance(scales[updown], dict):
                scales[updown] = {f"block_{i}": copy.deepcopy(scales[updown]) for i in blocks_with_transformer[updown]}
            # 遍历当前方向的每个块
            for i in blocks_with_transformer[updown]:
                block = f"block_{i}"
                # 如果当前块未赋值，则设置为默认比例
                if block not in scales[updown]:
                    scales[updown][block] = default_scale
                # 如果块的比例不是列表，则转换为列表格式
                if not isinstance(scales[updown][block], list):
                    scales[updown][block] = [scales[updown][block] for _ in range(transformer_per_block[updown])]
                # 如果块的比例列表仅有一个元素，则扩展为多个元素
                elif len(scales[updown][block]) == 1:
                    scales[updown][block] = scales[updown][block] * transformer_per_block[updown]
                # 如果块的比例列表长度不匹配，则抛出错误
                elif len(scales[updown][block]) != transformer_per_block[updown]:
                    raise ValueError(
                        f"Expected {transformer_per_block[updown]} scales for {updown}.{block}, got {len(scales[updown][block])}."
                    )
            # 将 scales 中的当前方向块转换为扁平格式
            for i in blocks_with_transformer[updown]:
                block = f"block_{i}"
                for tf_idx, value in enumerate(scales[updown][block]):
                    scales[f"{updown}.{block}.{tf_idx}"] = value
            # 删除 scales 中当前方向的条目
            del scales[updown]
    # 遍历 scales 字典中的每一层
        for layer in scales.keys():
            # 检查该层是否在 state_dict 中存在
            if not any(_translate_into_actual_layer_name(layer) in module for module in state_dict.keys()):
                # 如果不存在，抛出值错误，提示该层无法设置 lora 缩放
                raise ValueError(
                    f"Can't set lora scale for layer {layer}. It either doesn't exist in this unet or it has no attentions."
                )
    
        # 返回一个字典，键为实际层名，值为对应的权重
        return {_translate_into_actual_layer_name(name): weight for name, weight in scales.items()}
```