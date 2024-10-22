# `.\cogvideo-finetune\tools\convert_weight_sat2hf.py`

```
"""
# 此脚本演示如何从文本提示转换和生成视频
# 使用 CogVideoX 和 🤗Huggingface Diffusers Pipeline。
# 此脚本需要安装 `diffusers>=0.30.2` 库。

# 函数列表：
#     - reassign_query_key_value_inplace: 就地重新分配查询、键和值的权重。
#     - reassign_query_key_layernorm_inplace: 就地重新分配查询和键的层归一化。
#     - reassign_adaln_norm_inplace: 就地重新分配自适应层归一化。
#     - remove_keys_inplace: 就地移除状态字典中指定的键。
#     - replace_up_keys_inplace: 就地替换“up”块中的键。
#     - get_state_dict: 从保存的检查点中提取状态字典。
#     - update_state_dict_inplace: 就地更新状态字典以进行新的键分配。
#     - convert_transformer: 将变换器检查点转换为 CogVideoX 格式。
#     - convert_vae: 将 VAE 检查点转换为 CogVideoX 格式。
#     - get_args: 解析脚本的命令行参数。
#     - generate_video: 使用 CogVideoX 管道从文本提示生成视频。
"""

# 导入 argparse 模块用于解析命令行参数
import argparse
# 从 typing 导入 Any 和 Dict 类型
from typing import Any, Dict

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 T5EncoderModel 和 T5Tokenizer
from transformers import T5EncoderModel, T5Tokenizer

# 从 diffusers 库导入多个类
from diffusers import (
    AutoencoderKLCogVideoX,  # 自动编码器类
    CogVideoXDDIMScheduler,   # 调度器类
    CogVideoXImageToVideoPipeline,  # 图像到视频的管道类
    CogVideoXPipeline,        # 主管道类
    CogVideoXTransformer3DModel,  # 3D 变换器模型类
)

# 函数：就地重新分配查询、键和值的权重
def reassign_query_key_value_inplace(key: str, state_dict: Dict[str, Any]):
    # 根据原始键生成新的键，替换查询键值
    to_q_key = key.replace("query_key_value", "to_q")
    to_k_key = key.replace("query_key_value", "to_k")
    to_v_key = key.replace("query_key_value", "to_v")
    # 将状态字典中该键的值分割成三部分（查询、键和值）
    to_q, to_k, to_v = torch.chunk(state_dict[key], chunks=3, dim=0)
    # 将分割后的查询、键和值添加到状态字典中
    state_dict[to_q_key] = to_q
    state_dict[to_k_key] = to_k
    state_dict[to_v_key] = to_v
    # 从状态字典中移除原始键
    state_dict.pop(key)

# 函数：就地重新分配查询和键的层归一化
def reassign_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
    # 从键中提取层 ID 和权重或偏差类型
    layer_id, weight_or_bias = key.split(".")[-2:]

    # 根据键名确定新键名
    if "query" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_q.{weight_or_bias}"
    elif "key" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_k.{weight_or_bias}"

    # 将状态字典中原键的值移到新键中
    state_dict[new_key] = state_dict.pop(key)

# 函数：就地重新分配自适应层归一化
def reassign_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
    # 从键中提取层 ID 和权重或偏差类型
    layer_id, _, weight_or_bias = key.split(".")[-3:]

    # 将状态字典中该键的值分割为 12 部分
    weights_or_biases = state_dict[key].chunk(12, dim=0)
    # 合并特定部分形成新的权重或偏差
    norm1_weights_or_biases = torch.cat(weights_or_biases[0:3] + weights_or_biases[6:9])
    norm2_weights_or_biases = torch.cat(weights_or_biases[3:6] + weights_or_biases[9:12])

    # 构建新键名并更新状态字典
    norm1_key = f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}"
    state_dict[norm1_key] = norm1_weights_or_biases

    norm2_key = f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}"
    state_dict[norm2_key] = norm2_weights_or_biases

    # 从状态字典中移除原始键
    state_dict.pop(key)

# 函数：就地移除状态字典中的指定键
def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
    # 从状态字典中移除指定的键
    state_dict.pop(key)
# 定义一个函数，替换状态字典中的特定键，直接在字典中修改
def replace_up_keys_inplace(key: str, state_dict: Dict[str, Any]):
    # 将键字符串按点分割成列表
    key_split = key.split(".")
    # 获取指定层的索引，假设索引在第三个位置
    layer_index = int(key_split[2])
    # 计算替换后的层索引
    replace_layer_index = 4 - 1 - layer_index

    # 将分割后的键更新为 "up_blocks" 作为新的第二层
    key_split[1] = "up_blocks"
    # 更新层索引为计算后的新索引
    key_split[2] = str(replace_layer_index)
    # 将分割的键重新拼接为字符串
    new_key = ".".join(key_split)

    # 在状态字典中用新键替换旧键对应的值
    state_dict[new_key] = state_dict.pop(key)


# 定义一个字典，用于重命名 Transformer 模型的键
TRANSFORMER_KEYS_RENAME_DICT = {
    # 重命名 final_layernorm 键为 norm_final
    "transformer.final_layernorm": "norm_final",
    # 将 transformer 键重命名为 transformer_blocks
    "transformer": "transformer_blocks",
    # 重命名注意力层的键
    "attention": "attn1",
    # 重命名 MLP 层的键
    "mlp": "ff.net",
    # 重命名密集层的键
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    # 处理 layers 键的重命名
    ".layers": "",
    # 将 dense 键重命名为 to_out.0
    "dense": "to_out.0",
    # 处理输入层归一化的重命名
    "input_layernorm": "norm1.norm",
    # 处理后注意力层归一化的重命名
    "post_attn1_layernorm": "norm2.norm",
    # 重命名时间嵌入的层
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    # 处理 Patch 嵌入的重命名
    "mixins.patch_embed": "patch_embed",
    # 处理最终层的重命名
    "mixins.final_layer.norm_final": "norm_out.norm",
    "mixins.final_layer.linear": "proj_out",
    # 处理 ADA LN 调制层的重命名
    "mixins.final_layer.adaLN_modulation.1": "norm_out.linear",
    # 处理特定于 CogVideoX-5b-I2V 的重命名
    "mixins.pos_embed.pos_embedding": "patch_embed.pos_embedding",  # Specific to CogVideoX-5b-I2V
}

# 定义一个字典，用于特殊键的重映射
TRANSFORMER_SPECIAL_KEYS_REMAP = {
    # 映射特定的查询键值处理函数
    "query_key_value": reassign_query_key_value_inplace,
    # 映射查询层归一化列表的处理函数
    "query_layernorm_list": reassign_query_key_layernorm_inplace,
    # 映射键层归一化列表的处理函数
    "key_layernorm_list": reassign_query_key_layernorm_inplace,
    # 映射 ADA LN 调制层的处理函数
    "adaln_layer.adaLN_modulations": reassign_adaln_norm_inplace,
    # 映射嵌入令牌的处理函数
    "embed_tokens": remove_keys_inplace,
    # 映射频率正弦的处理函数
    "freqs_sin": remove_keys_inplace,
    # 映射频率余弦的处理函数
    "freqs_cos": remove_keys_inplace,
    # 映射位置嵌入的处理函数
    "position_embedding": remove_keys_inplace,
}

# 定义一个字典，用于重命名 VAE 模型的键
VAE_KEYS_RENAME_DICT = {
    # 将块的键重命名为 resnets. 
    "block.": "resnets.",
    # 将 down 的键重命名为 down_blocks.
    "down.": "down_blocks.",
    # 将 downsample 的键重命名为 downsamplers.0
    "downsample": "downsamplers.0",
    # 将 upsample 的键重命名为 upsamplers.0
    "upsample": "upsamplers.0",
    # 将 nin_shortcut 的键重命名为 conv_shortcut
    "nin_shortcut": "conv_shortcut",
    # 将编码器的块重命名
    "encoder.mid.block_1": "encoder.mid_block.resnets.0",
    "encoder.mid.block_2": "encoder.mid_block.resnets.1",
    # 将解码器的块重命名
    "decoder.mid.block_1": "decoder.mid_block.resnets.0",
    "decoder.mid.block_2": "decoder.mid_block.resnets.1",
}

# 定义一个字典，用于特殊键的重映射，适用于 VAE
VAE_SPECIAL_KEYS_REMAP = {
    # 映射损失的处理函数
    "loss": remove_keys_inplace,
    # 映射 up 的处理函数
    "up.": replace_up_keys_inplace,
}

# 定义一个常量，表示标记器的最大长度
TOKENIZER_MAX_LENGTH = 226


# 定义一个函数，从保存的字典中获取状态字典
def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 默认状态字典为保存的字典
    state_dict = saved_dict
    # 如果保存的字典中包含 "model" 键，则提取模型部分
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    # 如果保存的字典中包含 "module" 键，则提取模块部分
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    # 如果保存的字典中包含 "state_dict" 键，则提取状态字典
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    # 返回最终提取的状态字典
    return state_dict


# 定义一个函数，直接在状态字典中更新键
def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    # 用新键替换旧键在字典中的值
    state_dict[new_key] = state_dict.pop(old_key)


# 定义一个函数，用于转换 Transformer 模型
def convert_transformer(
    ckpt_path: str,
    num_layers: int,
    num_attention_heads: int,
    use_rotary_positional_embeddings: bool,
    i2v: bool,
    dtype: torch.dtype,
):
    # 定义一个前缀键，表示模型的前缀部分
    PREFIX_KEY = "model.diffusion_model."

    # 从指定路径加载原始状态字典，设置 map_location 为 "cpu" 和 mmap 为 True
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    # 创建一个 CogVideoXTransformer3DModel 实例，设置输入通道、层数、注意力头数等参数
    transformer = CogVideoXTransformer3DModel(
        # 根据 i2v 的值决定输入通道数
        in_channels=32 if i2v else 16,
        # 设置模型的层数
        num_layers=num_layers,
        # 设置注意力头的数量
        num_attention_heads=num_attention_heads,
        # 是否使用旋转位置嵌入
        use_rotary_positional_embeddings=use_rotary_positional_embeddings,
        # 是否使用学习到的位置嵌入
        use_learned_positional_embeddings=i2v,
    ).to(dtype=dtype)  # 将模型转换为指定的数据类型

    # 遍历原始状态字典的键列表
    for key in list(original_state_dict.keys()):
        # 从键中去掉前缀，以获得新的键名
        new_key = key[len(PREFIX_KEY) :]
        # 遍历重命名字典，替换键名中的特定部分
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        # 更新原始状态字典中的键值对
        update_state_dict_inplace(original_state_dict, key, new_key)

    # 再次遍历原始状态字典的键列表
    for key in list(original_state_dict.keys()):
        # 遍历特殊键的映射字典
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            # 如果特殊键不在当前键中，则继续下一个键
            if special_key not in key:
                continue
            # 调用处理函数以更新状态字典
            handler_fn_inplace(key, original_state_dict)
    
    # 加载更新后的状态字典到 transformer 中，严格匹配键
    transformer.load_state_dict(original_state_dict, strict=True)
    # 返回 transformer 实例
    return transformer
# 定义一个函数，将 VAE 模型从检查点路径转换
def convert_vae(ckpt_path: str, scaling_factor: float, dtype: torch.dtype):
    # 从指定路径加载原始状态字典，使用 CPU 映射
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    # 创建一个新的 VAE 对象，并将其数据类型设置为指定的 dtype
    vae = AutoencoderKLCogVideoX(scaling_factor=scaling_factor).to(dtype=dtype)

    # 遍历原始状态字典的所有键
    for key in list(original_state_dict.keys()):
        # 复制当前键以便修改
        new_key = key[:]
        # 遍历重命名字典，将旧键替换为新键
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        # 更新原始状态字典中的键
        update_state_dict_inplace(original_state_dict, key, new_key)

    # 再次遍历原始状态字典的所有键
    for key in list(original_state_dict.keys()):
        # 遍历特殊键映射字典
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            # 如果特殊键不在当前键中，则跳过
            if special_key not in key:
                continue
            # 使用处理函数处理原始状态字典
            handler_fn_inplace(key, original_state_dict)

    # 加载更新后的状态字典到 VAE 模型中，严格匹配
    vae.load_state_dict(original_state_dict, strict=True)
    # 返回转换后的 VAE 对象
    return vae


# 定义获取命令行参数的函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加原始变换器检查点路径参数
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint")
    # 添加原始 VAE 检查点路径参数
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    # 添加输出路径参数，作为必需参数
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    # 添加是否以 fp16 格式保存模型权重的布尔参数
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to save the model weights in fp16")
    # 添加是否以 bf16 格式保存模型权重的布尔参数
    parser.add_argument("--bf16", action="store_true", default=False, help="Whether to save the model weights in bf16")
    # 添加是否在保存后推送到 HF Hub 的布尔参数
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving"
    )
    # 添加文本编码器缓存目录路径参数
    parser.add_argument(
        "--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory"
    )
    # 添加变换器块数量参数，默认值为 30
    parser.add_argument("--num_layers", type=int, default=30, help="Number of transformer blocks")
    # 添加注意力头数量参数，默认值为 30
    parser.add_argument("--num_attention_heads", type=int, default=30, help="Number of attention heads")
    # 添加是否使用旋转位置嵌入的布尔参数
    parser.add_argument(
        "--use_rotary_positional_embeddings", action="store_true", default=False, help="Whether to use RoPE or not"
    )
    # 添加 VAE 的缩放因子参数，默认值为 1.15258426
    parser.add_argument("--scaling_factor", type=float, default=1.15258426, help="Scaling factor in the VAE")
    # 添加 SNR 偏移比例参数，默认值为 3.0
    parser.add_argument("--snr_shift_scale", type=float, default=3.0, help="Scaling factor in the VAE")
    # 添加是否以 fp16 格式保存模型权重的布尔参数
    parser.add_argument("--i2v", action="store_true", default=False, help="Whether to save the model weights in fp16")
    # 解析命令行参数并返回
    return parser.parse_args()


# 如果脚本作为主程序执行
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()

    # 初始化 transformer 和 vae 为 None
    transformer = None
    vae = None
    # 检查是否同时传递了 --fp16 和 --bf16 参数
    if args.fp16 and args.bf16:
        # 如果同时存在则抛出值错误
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    # 根据输入参数选择数据类型
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    # 如果提供了变换器检查点路径，则转换变换器
    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(
            # 传递变换器检查点路径及相关参数
            args.transformer_ckpt_path,
            args.num_layers,
            args.num_attention_heads,
            args.use_rotary_positional_embeddings,
            args.i2v,
            dtype,
        )
    # 如果提供了 VAE 检查点路径，则转换 VAE
    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path, args.scaling_factor, dtype)

    # 设置文本编码器的模型 ID
    text_encoder_id = "google/t5-v1_1-xxl"
    # 从预训练模型中加载分词器
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    # 从预训练模型中加载文本编码器
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)
    # 处理参数以确保数据连续性
    for param in text_encoder.parameters():
        # 使参数数据连续
        param.data = param.data.contiguous()

    # 从配置中创建调度器
    scheduler = CogVideoXDDIMScheduler.from_config(
        {
            # 设置调度器的超参数
            "snr_shift_scale": args.snr_shift_scale,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "set_alpha_to_one": True,
            "timestep_spacing": "trailing",
        }
    )
    # 根据 i2v 参数选择管道类
    if args.i2v:
        pipeline_cls = CogVideoXImageToVideoPipeline
    else:
        pipeline_cls = CogVideoXPipeline

    # 实例化管道
    pipe = pipeline_cls(
        # 传递所需的组件
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    # 如果选择 fp16 则将管道转为 fp16
    if args.fp16:
        pipe = pipe.to(dtype=torch.float16)
    # 如果选择 bf16 则将管道转为 bf16
    if args.bf16:
        pipe = pipe.to(dtype=torch.bfloat16)

    # 保存预训练的管道到指定路径
    pipe.save_pretrained(args.output_path, safe_serialization=True, push_to_hub=args.push_to_hub)
```