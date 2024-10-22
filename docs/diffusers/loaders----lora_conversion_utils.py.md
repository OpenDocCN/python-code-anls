# `.\diffusers\loaders\lora_conversion_utils.py`

```py
# 版权声明，表示该代码属于 HuggingFace 团队，所有权利保留
# 授权信息，依据 Apache 许可证版本 2.0
# 用户必须遵循许可证条款使用此文件
# 许可证的获取链接
# 除非适用法律要求或书面同意，软件在“现状”基础上分发，不提供任何形式的保证或条件
# 查看许可证获取关于权限和限制的详细信息

# 导入正则表达式模块
import re

# 从 utils 模块导入 is_peft_version 和 logging 函数
from ..utils import is_peft_version, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义可能将 SGM 块映射到 diffusers 的函数
def _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter="_", block_slice_pos=5):
    # 1. 获取所有的 state_dict 键
    all_keys = list(state_dict.keys())
    # 定义 SGM 模式的列表
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    # 2. 检查是否需要重新映射，如果不需要则返回原始字典
    is_in_sgm_format = False
    # 遍历所有键以检查是否包含 SGM 模式
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break

    # 如果不在 SGM 格式中，直接返回原字典
    if not is_in_sgm_format:
        return state_dict

    # 3. 否则，根据 SGM 模式重新映射
    new_state_dict = {}
    # 定义内部块映射的列表
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    # 初始化输入、中间和输出块的 ID 集合
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

    # 遍历所有层以填充块 ID
    for layer in all_keys:
        # 如果层名称中包含 "text"，直接移到新字典中
        if "text" in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            # 从层名称中提取 ID
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
            # 根据层类型添加到相应的 ID 集合中
            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                # 如果层不支持，则抛出异常
                raise ValueError(f"Checkpoint not supported because layer {layer} not supported.")

    # 根据层 ID 获取输入块的所有键
    input_blocks = {
        layer_id: [key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key]
        for layer_id in input_block_ids
    }
    # 根据层 ID 获取中间块的所有键
    middle_blocks = {
        layer_id: [key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key]
        for layer_id in middle_block_ids
    }
    # 根据层 ID 获取输出块的所有键
    output_blocks = {
        layer_id: [key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key]
        for layer_id in output_block_ids
    }

    # 按照新的规则重命名键
    # 遍历输入块的 ID 列表
        for i in input_block_ids:
            # 计算当前块的 ID
            block_id = (i - 1) // (unet_config.layers_per_block + 1)
            # 计算当前层在块内的 ID
            layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)
    
            # 遍历当前输入块中的每个键
            for key in input_blocks[i]:
                # 从键中提取内部块的 ID
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                # 判断当前键是操作还是下采样，并获取对应的内部块键
                inner_block_key = inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
                # 确定块内层的字符串表示
                inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
                # 构造新的键
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                # 将状态字典中的旧键对应的值存入新键
                new_state_dict[new_key] = state_dict.pop(key)
    
        # 遍历中间块的 ID 列表
        for i in middle_block_ids:
            key_part = None
            # 根据中间块 ID 设置键部分
            if i == 0:
                key_part = [inner_block_map[0], "0"]
            elif i == 1:
                key_part = [inner_block_map[1], "0"]
            elif i == 2:
                key_part = [inner_block_map[0], "1"]
            else:
                # 抛出异常以防无效的中间块 ID
                raise ValueError(f"Invalid middle block id {i}.")
    
            # 遍历当前中间块中的每个键
            for key in middle_blocks[i]:
                # 构造新的键
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1] + key_part + key.split(delimiter)[block_slice_pos:]
                )
                # 将状态字典中的旧键对应的值存入新键
                new_state_dict[new_key] = state_dict.pop(key)
    
        # 遍历输出块的 ID 列表
        for i in output_block_ids:
            # 计算当前块的 ID
            block_id = i // (unet_config.layers_per_block + 1)
            # 计算当前层在块内的 ID
            layer_in_block_id = i % (unet_config.layers_per_block + 1)
    
            # 遍历当前输出块中的每个键
            for key in output_blocks[i]:
                # 从键中提取内部块的 ID
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                # 获取对应的内部块键
                inner_block_key = inner_block_map[inner_block_id]
                # 确定块内层的字符串表示
                inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
                # 构造新的键
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                # 将状态字典中的旧键对应的值存入新键
                new_state_dict[new_key] = state_dict.pop(key)
    
        # 如果状态字典还有未转换的条目，则抛出异常
        if len(state_dict) > 0:
            raise ValueError("At this point all state dict entries have to be converted.")
    
        # 返回新的状态字典
        return new_state_dict
# 将非 Diffusers 格式的 LoRA 状态字典转换为兼容 Diffusers 格式的状态字典
def _convert_non_diffusers_lora_to_diffusers(state_dict, unet_name="unet", text_encoder_name="text_encoder"):
    # 初始化 U-Net 模块的状态字典
    unet_state_dict = {}
    # 初始化文本编码器的状态字典
    te_state_dict = {}
    # 初始化第二文本编码器的状态字典
    te2_state_dict = {}
    # 初始化网络 alphas 字典
    network_alphas = {}

    # 检查是否存在 DoRA 支持的 LoRA
    dora_present_in_unet = any("dora_scale" in k and "lora_unet_" in k for k in state_dict)
    dora_present_in_te = any("dora_scale" in k and ("lora_te_" in k or "lora_te1_" in k) for k in state_dict)
    dora_present_in_te2 = any("dora_scale" in k and "lora_te2_" in k for k in state_dict)
    # 如果存在 DoRA，则检查 peft 版本是否满足要求
    if dora_present_in_unet or dora_present_in_te or dora_present_in_te2:
        if is_peft_version("<", "0.9.0"):
            # 抛出错误提示需要更新 peft 版本
            raise ValueError(
                "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
            )

    # 遍历所有 LoRA 权重的键
    all_lora_keys = list(state_dict.keys())
    # 遍历所有 LoRA 键
    for key in all_lora_keys:
        # 如果键不以 "lora_down.weight" 结尾，则跳过
        if not key.endswith("lora_down.weight"):
            continue

        # 提取 LoRA 名称
        lora_name = key.split(".")[0]

        # 找到对应的上升权重和 alpha
        lora_name_up = lora_name + ".lora_up.weight"
        lora_name_alpha = lora_name + ".alpha"

        # 处理 U-Net 的 LoRAs
        if lora_name.startswith("lora_unet_"):
            # 转换为 Diffusers 格式的 U-Net LoRA 键
            diffusers_name = _convert_unet_lora_key(key)

            # 存储下权重和上权重
            unet_state_dict[diffusers_name] = state_dict.pop(key)
            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # 如果存在 DoRA 规模，则存储
            if dora_present_in_unet:
                # 替换相应的 DoRA 规模键
                dora_scale_key_to_replace = "_lora.down." if "_lora.down." in diffusers_name else ".lora.down."
                unet_state_dict[
                    diffusers_name.replace(dora_scale_key_to_replace, ".lora_magnitude_vector.")
                ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))

        # 处理文本编码器的 LoRAs
        elif lora_name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            # 转换为 Diffusers 格式的文本编码器 LoRA 键
            diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

            # 存储 te 或 te2 的下权重和上权重
            if lora_name.startswith(("lora_te_", "lora_te1_")):
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            else:
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # 如果存在 DoRA 规模，则存储
            if dora_present_in_te or dora_present_in_te2:
                # 替换相应的 DoRA 规模键
                dora_scale_key_to_replace_te = (
                    "_lora.down." if "_lora.down." in diffusers_name else ".lora_linear_layer."
                )
                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    te_state_dict[
                        diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                elif lora_name.startswith("lora_te2_"):
                    te2_state_dict[
                        diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))

        # 如果存在 alpha，则存储
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            # 更新网络中的 alpha 名称
            network_alphas.update(_get_alpha_name(lora_name_alpha, diffusers_name, alpha))

    # 检查是否还有剩余的键
    if len(state_dict) > 0:
        # 如果有未重命名的键，则引发错误
        raise ValueError(f"The following keys have not been correctly renamed: \n\n {', '.join(state_dict.keys())}")
    # 记录日志，提示检测到非扩散模型的检查点
        logger.info("Non-diffusers checkpoint detected.")
    
        # 构造最终的状态字典
        # 为 UNet 的状态字典添加前缀
        unet_state_dict = {f"{unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
        # 为文本编码器的状态字典添加前缀
        te_state_dict = {f"{text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()}
        # 检查第二个文本编码器状态字典是否非空，如果非空则添加前缀
        te2_state_dict = (
            {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
            if len(te2_state_dict) > 0
            else None
        )
        # 如果第二个文本编码器状态字典存在，则更新第一个状态字典
        if te2_state_dict is not None:
            te_state_dict.update(te2_state_dict)
    
        # 合并 UNet 和文本编码器的状态字典
        new_state_dict = {**unet_state_dict, **te_state_dict}
        # 返回合并后的状态字典和网络的 alpha 值
        return new_state_dict, network_alphas
# 定义一个将 U-Net LoRA 键转换为 Diffusers 兼容键的函数
def _convert_unet_lora_key(key):
    """
    转换 U-Net LoRA 键为 Diffusers 兼容的键。
    """
    # 将键中的前缀替换为更通用的格式
    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

    # 替换常见的 U-Net 命名模式
    diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("middle.block", "mid_block")
    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
    diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("proj.in", "proj_in")
    diffusers_name = diffusers_name.replace("proj.out", "proj_out")
    diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

    # 针对 SDXL 特定的转换
    if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
        # 使用正则表达式移除最后一个数字
        pattern = r"\.\d+(?=\D*$)"
        diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
    if ".in." in diffusers_name:
        # 将指定层替换为新的名称
        diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
    if ".out." in diffusers_name:
        # 将指定层替换为新的名称
        diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
    if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
        # 将操作符替换为卷积层
        diffusers_name = diffusers_name.replace("op", "conv")
    if "skip" in diffusers_name:
        # 将跳跃连接替换为卷积捷径
        diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

    # 针对 LyCORIS 特定的转换
    if "time.emb.proj" in diffusers_name:
        # 替换时间嵌入投影的名称
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
    if "conv.shortcut" in diffusers_name:
        # 替换卷积捷径的名称
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

    # 一般性的转换
    if "transformer_blocks" in diffusers_name:
        if "attn1" in diffusers_name or "attn2" in diffusers_name:
            # 将注意力层的名称进行处理
            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
        elif "ff" in diffusers_name:
            pass  # 如果包含前馈层，什么都不做
    elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
        pass  # 如果包含投影层，什么都不做
    else:
        pass  # 其他情况，什么都不做

    # 返回转换后的键
    return diffusers_name


# 定义一个将文本编码器 LoRA 键转换为 Diffusers 兼容键的函数
def _convert_text_encoder_lora_key(key, lora_name):
    """
    转换文本编码器 LoRA 键为 Diffusers 兼容的键。
    """
    # 检查 LoRA 名称的前缀
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        # 根据前缀设置要替换的键
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        # 如果条件不满足，则设置替换的键为 "lora_te2_"
        key_to_replace = "lora_te2_"

    # 将原始键中的指定键去除，并将下划线替换为点，以形成 diffusers_name
    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    # 将 "text.model" 替换为 "text_model"
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    # 将 "self.attn" 替换为 "self_attn"
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    # 将 "q.proj.lora" 替换为 "to_q_lora"
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    # 将 "k.proj.lora" 替换为 "to_k_lora"
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    # 将 "v.proj.lora" 替换为 "to_v_lora"
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    # 将 "out.proj.lora" 替换为 "to_out_lora"
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    # 将 "text.projection" 替换为 "text_projection"
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    # 检查 diffusers_name 是否包含 "self_attn" 或 "text_projection"
    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        # 如果包含，则不进行任何操作，直接跳过
        pass
    # 检查 diffusers_name 是否包含 "mlp"
    elif "mlp" in diffusers_name:
        # 注意这是新的 diffusers 约定，其余代码可能尚未使用此约定
        # 将 ".lora." 替换为 ".lora_linear_layer."
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    # 返回最终的 diffusers_name
    return diffusers_name
# 获取 Diffusers 模型的正确 alpha 名称
def _get_alpha_name(lora_name_alpha, diffusers_name, alpha):
    # 文档字符串，说明函数的功能
    """
    Gets the correct alpha name for the Diffusers model.
    """
    # 检查 lora_name_alpha 是否以 "lora_unet_" 开头
    if lora_name_alpha.startswith("lora_unet_"):
        # 设置前缀为 "unet."
        prefix = "unet."
    # 检查 lora_name_alpha 是否以 "lora_te_" 或 "lora_te1_" 开头
    elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
        # 设置前缀为 "text_encoder."
        prefix = "text_encoder."
    else:
        # 其他情况设置前缀为 "text_encoder_2."
        prefix = "text_encoder_2."
    # 生成新的名称，组合前缀和 diffusers_name 的部分，并加上 ".alpha"
    new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
    # 返回一个字典，包含新名称和 alpha 值
    return {new_name: alpha}
```