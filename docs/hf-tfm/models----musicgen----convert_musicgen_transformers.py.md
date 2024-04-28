# `.\transformers\models\musicgen\convert_musicgen_transformers.py`

```
# 设置编码格式为 UTF-8
# 版权声明：版权归 2023 年 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可 2.0 版本（"许可证"）获得许可；
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 不提供任何形式的保证或条件
# 有关详细信息，请参阅许可证
"""从原始存储库转换 MusicGen 检查点"""
# 导入必要的库
import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import torch
# 从 audiocraft 库中导入 MusicGen 模型
from audiocraft.models import MusicGen
# 导入 HuggingFace 库中的相关模块
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    EncoderModel,
    MusicgenDecoderConfig,
    MusicgenForConditionalGeneration,
    MusicgenProcessor,
    T5EncoderModel,
)
# 导入 MusicGen 模型的相关组件
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
# 导入日志记录工具
from transformers.utils import logging

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预期丢失的键列表
EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]

# 定义重命名函数
def rename_keys(name):
    # 如果名称中包含 "emb"，则替换为 "model.decoder.embed_tokens"
    if "emb" in name:
        name = name.replace("emb", "model.decoder.embed_tokens")
    # 如果名称中包含 "transformer"，则替换为 "model.decoder"
    if "transformer" in name:
        name = name.replace("transformer", "model.decoder")
    # 如果名称中包含 "cross_attention"，则替换为 "encoder_attn"
    if "cross_attention" in name:
        name = name.replace("cross_attention", "encoder_attn")
    # 如果名称中包含 "linear1"，则替换为 "fc1"
    if "linear1" in name:
        name = name.replace("linear1", "fc1")
    # 如果名称中包含 "linear2"，则替换为 "fc2"
    if "linear2" in name:
        name = name.replace("linear2", "fc2")
    # 如果名称中包含 "norm1"，则替换为 "self_attn_layer_norm"
    if "norm1" in name:
        name = name.replace("norm1", "self_attn_layer_norm")
    # 如果名称中包含 "norm_cross"，则替换为 "encoder_attn_layer_norm"
    if "norm_cross" in name:
        name = name.replace("norm_cross", "encoder_attn_layer_norm")
    # 如果名称中包含 "norm2"，则替换为 "final_layer_norm"
    if "norm2" in name:
        name = name.replace("norm2", "final_layer_norm")
    # 如果名称中包含 "out_norm"，则替换为 "model.decoder.layer_norm"
    if "out_norm" in name:
        name = name.replace("out_norm", "model.decoder.layer_norm")
    # 如果名称中包含 "linears"，则替换为 "lm_heads"
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    # 如果名称中包含特定字符串，进行替换
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "enc_to_dec_proj")
    # 返回修改后的名称
    return name

# 定义函数，对给定的 state_dict 进行重命名处理
def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> Tuple[Dict, Dict]:
    """函数用于根据 HF 模块名称对 fairseq Musicgen state_dict 进行重命名处理。
    它进一步将状态字典分为解码器（LM）状态字典和编码器-解码器投影的状态字典。"""
    # 获取状态字典的键列表
    keys = list(state_dict.keys())
    # 初始化编码器-解码器投影的状态字典
    enc_dec_proj_state_dict = {}
    # 遍历给定的键列表
    for key in keys:
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 重命名当前键
        key = rename_keys(key)
        # 检查是否包含 "in_proj_weight" 字符串
        if "in_proj_weight" in key:
            # 拆分融合的 qkv 投影
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        # 检查是否包含 "enc_to_dec_proj" 字符串
        elif "enc_to_dec_proj" in key:
            # 将当前键值对添加到新字典中
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            # 否则将当前键值对添加到原字典中
            state_dict[key] = val
    # 返回经过处理的字典和额外的字典
    return state_dict, enc_dec_proj_state_dict
# 从检查点名称中解析出 Musicgen 解码器的配置信息，并返回 MusicgenDecoderConfig 对象
def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenDecoderConfig:
    # 根据不同的检查点名称设置不同的隐藏层大小、隐藏层数量和注意力头数量
    if checkpoint == "small" or checkpoint == "facebook/musicgen-stereo-small":
        hidden_size = 1024
        num_hidden_layers = 24
        num_attention_heads = 16
    elif checkpoint == "medium" or checkpoint == "facebook/musicgen-stereo-medium":
        hidden_size = 1536
        num_hidden_layers = 48
        num_attention_heads = 24
    elif checkpoint == "large" or checkpoint == "facebook/musicgen-stereo-large":
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 32
    else:
        raise ValueError(
            "Checkpoint should be one of `['small', 'medium', 'large']` for the mono checkpoints, "
            "or `['facebook/musicgen-stereo-small', 'facebook/musicgen-stereo-medium', 'facebook/musicgen-stereo-large']` "
            f"for the stereo checkpoints, got {checkpoint}."
        )

    # 根据检查点名称中是否包含 "stereo" 设置音频通道和码本数量
    if "stereo" in checkpoint:
        audio_channels = 2
        num_codebooks = 8
    else:
        audio_channels = 1
        num_codebooks = 4

    # 创建 MusicgenDecoderConfig 对象并返回
    config = MusicgenDecoderConfig(
        hidden_size=hidden_size,
        ffn_dim=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_codebooks=num_codebooks,
        audio_channels=audio_channels,
    )
    return config


@torch.no_grad()
def convert_musicgen_checkpoint(
    checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu", safe_serialization=False
):
    # 获取预训练的 MusicGen 模型
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    # 根据检查点信息获取 MusicgenDecoderConfig 对象
    decoder_config = decoder_config_from_checkpoint(checkpoint)

    # 获取解码器的状态字典
    decoder_state_dict = fairseq_model.lm.state_dict()
    # 重命名解码器的状态字典中的特定键，与隐藏层大小对应
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(
        decoder_state_dict, hidden_size=decoder_config.hidden_size
    )

    # 加载 T5 文本编码器和 32KHz 音频编码器模型
    text_encoder = T5EncoderModel.from_pretrained("t5-base")
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    # 初始化 Musicgen 解码器
    decoder = MusicgenForCausalLM(decoder_config).eval()

    # 加载解码器权重，可能会缺少嵌入词和编码器-��码器投影
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)

    # 移除特定键后的缺失键
    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # 初始化综合模型
    model = MusicgenForConditionalGeneration(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder)

    # 加载预训练的编码器-解码器投影
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)

    # 检查是否可以进行正向传播
    # 生成一个长为2*decoder_config.num_codebooks的torch长整型张量，再reshape为2行，-1列的形状
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1)
    # 将input_ids reshape为2*decoder_config.num_codebooks行，-1列的形状
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1)

    # 使用torch.no_grad()上下文，计算模型生成的logits，输入为input_ids和decoder_input_ids
    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    # 如果logits的形状不符合预期，抛出ValueError异常
    if logits.shape != (2 * decoder_config.num_codebooks, 1, 2048):
        raise ValueError("Incorrect shape for logits")

    # 现在构建处理器
    # 从预训练模型"t5-base"加载分词器
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # 从预训练模型"facebook/encodec_32khz"加载特征提取器，设置padding_side为"left"，feature_size为decoder_config.audio_channels
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/encodec_32khz", padding_side="left", feature_size=decoder_config.audio_channels
    )

    # 设置音乐生成处理器，其中包括特征提取器和分词器
    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 设置适当的bos/pad令牌id
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048

    # 设置其他默认的生成配置参数
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0

    # 如果pytorch_dump_folder不为空
    if pytorch_dump_folder is not None:
        # 如果pytorch_dump_folder不存在，则创建它
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        # 输出保存模型的信息，将模型保存到pytorch_dump_folder，并使用安全序列化方式
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        model.save_pretrained(pytorch_dump_folder, safe_serialization=safe_serialization)
        # 将处理器保存到pytorch_dump_folder
        processor.save_pretrained(pytorch_dump_folder)

    # 如果repo_id不为空
    if repo_id:
        # 输出推送模型的信息，将模型推送到repo_id，并使用安全序列化方式
        logger.info(f"Pushing model {checkpoint} to {repo_id}")
        model.push_to_hub(repo_id, safe_serialization=safe_serialization)
        # 将处理器推送到repo_id
        processor.push_to_hub(repo_id)
# 如果当前脚本是主脚本入口
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 定义必需的参数
    # 参数名称: --checkpoint
    # 默认值: "small"
    # 参数类型: 字符串
    # 帮助信息: 指定要转换的 MusicGen 模型的检查点大小。可以是以下之一:"['small', 'medium', 'large']" 用于单声道检查点，或 "['facebook/musicgen-stereo-small', 'facebook/musicgen-stereo-medium', 'facebook/musicgen-stereo-large']" 用于立体声检查点。
    parser.add_argument(
        "--checkpoint",
        default="small",
        type=str,
        help="Checkpoint size of the MusicGen model you'd like to convert. Can be one of: "
        "`['small', 'medium', 'large']` for the mono checkpoints, or "
        "`['facebook/musicgen-stereo-small', 'facebook/musicgen-stereo-medium', 'facebook/musicgen-stereo-large']` "
        "for the stereo checkpoints.",
    )

    # 参数名称: --pytorch_dump_folder
    # 是否必需: 是
    # 默认值: None
    # 参数类型: 字符串
    # 帮助信息: 输出 PyTorch 模型的目录路径。
    parser.add_argument(
        "--pytorch_dump_folder",
        required=True,
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    # 参数名称: --push_to_hub
    # 默认值: None
    # 参数类型: 字符串
    # 帮助信息: 在 🤗 hub 上上传转换后的模型的位置。
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 参数名称: --device
    # 默认值: "cpu"
    # 参数类型: 字符串
    # 帮助信息: 用于运行转换的 Torch 设备, 可以是 cpu 或 cuda。
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )

    # 参数名称: --safe_serialization
    # 是否为标志参数: 是
    # 帮助信息: 是否使用 `safetensors` 保存模型, 或使用传统的 PyTorch 方式 (使用 `pickle`)。
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).",
    )

    # 解析命令行参数并赋值给 args
    args = parser.parse_args()

    # 调用 convert_musicgen_checkpoint 函数, 使用命令行参数作为参数
    convert_musicgen_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
```