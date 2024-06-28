# `.\models\musicgen\convert_musicgen_transformers.py`

```py
# 设置文件编码为 UTF-8，确保支持中文等非 ASCII 字符
# 版权声明和许可信息，指明此代码的版权归属和使用许可
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
from pathlib import Path  # 用于处理文件路径的类
from typing import Dict, OrderedDict, Tuple  # 引入类型提示，用于静态类型检查

import torch  # 引入 PyTorch 库
from audiocraft.models import MusicGen  # 导入本地定义的 MusicGen 模型

# 从 transformers 库中导入必要的类和函数
from transformers import (
    AutoFeatureExtractor,  # 自动特征提取器
    AutoTokenizer,  # 自动分词器
    EncodecModel,  # 编码模型（可能是拼写错误，应为 EncoderModel）
    MusicgenDecoderConfig,  # Musicgen 解码器配置
    MusicgenForConditionalGeneration,  # 用于条件生成的 Musicgen 模型
    MusicgenProcessor,  # Musicgen 处理器
    T5EncoderModel,  # T5 编码模型
)
# 从 transformers 库的 musicgen 模块中导入特定的类
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM  # 用于因果语言模型的 Musicgen
from transformers.utils import logging  # 导入日志记录工具

# 设置日志的详细程度为 info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预期缺失的模型键列表
EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]


def rename_keys(name):
    """根据预定义规则重命名模型状态字典中的键名。

    Args:
        name (str): 原始的键名字符串。

    Returns:
        str: 重命名后的键名字符串。
    """
    if "emb" in name:
        name = name.replace("emb", "model.decoder.embed_tokens")
    if "transformer" in name:
        name = name.replace("transformer", "model.decoder")
    if "cross_attention" in name:
        name = name.replace("cross_attention", "encoder_attn")
    if "linear1" in name:
        name = name.replace("linear1", "fc1")
    if "linear2" in name:
        name = name.replace("linear2", "fc2")
    if "norm1" in name:
        name = name.replace("norm1", "self_attn_layer_norm")
    if "norm_cross" in name:
        name = name.replace("norm_cross", "encoder_attn_layer_norm")
    if "norm2" in name:
        name = name.replace("norm2", "final_layer_norm")
    if "out_norm" in name:
        name = name.replace("out_norm", "model.decoder.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "enc_to_dec_proj")
    return name


def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> Tuple[Dict, Dict]:
    """根据 Hugging Face 模块名称规则重命名 fairseq Musicgen 的状态字典，并将其分成解码器（LM）状态字典和编码器-解码器投影的状态字典。

    Args:
        state_dict (OrderedDict): 原始的 fairseq Musicgen 状态字典。
        hidden_size (int): 隐藏层大小。

    Returns:
        Tuple[Dict, Dict]: 重命名后的解码器状态字典和编码器-解码器投影状态字典的元组。
    """
    keys = list(state_dict.keys())
    enc_dec_proj_state_dict = {}  # 用于存储编码器-解码器投影的状态字典
    # 对于给定的每个键进行迭代处理
    for key in keys:
        # 弹出当前状态字典中的键，并将其对应的值赋给变量val
        val = state_dict.pop(key)
        # 使用指定函数重命名当前的键值
        key = rename_keys(key)
        # 如果当前键名包含'in_proj_weight'字符串
        if "in_proj_weight" in key:
            # 拆分融合的qkv投影权重
            # 更新状态字典，替换键名中的'in_proj_weight'为'q_proj.weight'，并赋予对应的值
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            # 更新状态字典，替换键名中的'in_proj_weight'为'k_proj.weight'，并赋予对应的值
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            # 更新状态字典，替换键名中的'in_proj_weight'为'v_proj.weight'，并赋予对应的值
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        # 如果当前键名包含'enc_to_dec_proj'字符串
        elif "enc_to_dec_proj" in key:
            # 将当前键值对存入enc_dec_proj_state_dict字典中，去除键名中'enc_to_dec_proj.'部分
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            # 否则，直接将当前键值对存回状态字典中
            state_dict[key] = val
    # 返回更新后的状态字典及enc_dec_proj_state_dict字典
    return state_dict, enc_dec_proj_state_dict
# 根据给定的检查点名称返回MusicgenDecoderConfig配置对象
def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenDecoderConfig:
    # 根据不同的检查点名称设置不同的隐藏层大小、隐藏层数和注意力头数
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
        # 如果检查点名称不符合预期，则抛出数值错误异常
        raise ValueError(
            "Checkpoint should be one of `['small', 'medium', 'large']` for the mono checkpoints, "
            "or `['facebook/musicgen-stereo-small', 'facebook/musicgen-stereo-medium', 'facebook/musicgen-stereo-large']` "
            f"for the stereo checkpoints, got {checkpoint}."
        )

    # 根据检查点名称中是否包含"stereo"关键词设置音频通道数和码书数
    if "stereo" in checkpoint:
        audio_channels = 2
        num_codebooks = 8
    else:
        audio_channels = 1
        num_codebooks = 4

    # 创建MusicgenDecoderConfig对象，使用之前设置的参数
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
# 从Fairseq模型的预训练检查点转换MusicGen模型的函数
def convert_musicgen_checkpoint(
    checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu", safe_serialization=False
):
    # 从Fairseq库中获取预训练的MusicGen模型
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    # 根据检查点名称获取解码器的配置信息
    decoder_config = decoder_config_from_checkpoint(checkpoint)

    # 获取Fairseq模型的语言模型状态字典
    decoder_state_dict = fairseq_model.lm.state_dict()
    # 重命名解码器的状态字典，同时获取编码器到解码器投影的状态字典
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(
        decoder_state_dict, hidden_size=decoder_config.hidden_size
    )

    # 从预训练模型中加载T5文本编码器和32kHz音频编码器
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-base")
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    # 创建MusicgenForCausalLM对象作为解码器
    decoder = MusicgenForCausalLM(decoder_config).eval()

    # 加载解码器的所有权重，但可能缺少嵌入层和编码器到解码器的投影
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)

    # 对于符合预期缺失的键，移除其在缺失列表中
    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    # 如果仍有缺失的键存在，则抛出数值错误异常
    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    # 如果存在不预期的键，则抛出数值错误异常
    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # 初始化组合模型，包括文本编码器、音频编码器和解码器
    model = MusicgenForConditionalGeneration(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder)

    # 加载预训练的编码器到解码器投影权重
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)
    # 检查是否可以进行前向传播

    # 创建一个长为 2*decoder_config.num_codebooks 的长整型张量，并重塑为形状为 (2, -1)
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1)
    # 将 input_ids 重塑为形状为 (2*decoder_config.num_codebooks, -1) 的张量作为 decoder_input_ids
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1)

    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型进行推断，传入 input_ids 和 decoder_input_ids，获取logits
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    # 检查 logits 的形状是否为 (2*decoder_config.num_codebooks, 1, 2048)，否则引发 ValueError 异常
    if logits.shape != (2 * decoder_config.num_codebooks, 1, 2048):
        raise ValueError("Incorrect shape for logits")

    # 实例化一个 T5 tokenizer，从预训练模型 "google-t5/t5-base" 加载
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    # 实例化一个特征提取器，从预训练模型 "facebook/encodec_32khz" 加载，设置填充在左侧，特征大小为 decoder_config.audio_channels
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/encodec_32khz", padding_side="left", feature_size=decoder_config.audio_channels
    )

    # 实例化一个音乐生成处理器，传入特征提取器和 tokenizer
    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 设置适当的开始和填充标记的 ID
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048

    # 设置其他默认的生成配置参数
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0

    # 如果指定了 pytorch_dump_folder，则保存模型和处理器到该文件夹
    if pytorch_dump_folder is not None:
        # 创建目录，如果已存在则不做任何操作
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        # 记录日志，显示正在保存模型到指定目录
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        # 保存模型到 pytorch_dump_folder，使用安全序列化进行保存
        model.save_pretrained(pytorch_dump_folder, safe_serialization=safe_serialization)
        # 保存处理器到 pytorch_dump_folder
        processor.save_pretrained(pytorch_dump_folder)

    # 如果提供了 repo_id，则推送模型到指定的 Hub 仓库
    if repo_id:
        # 记录日志，显示正在推送模型到指定 repo_id
        logger.info(f"Pushing model {checkpoint} to {repo_id}")
        # 将模型推送到指定的 repo_id，使用安全序列化进行保存
        model.push_to_hub(repo_id, safe_serialization=safe_serialization)
        # 将处理器推送到指定的 repo_id
        processor.push_to_hub(repo_id)
if __name__ == "__main__":
    # 如果脚本直接执行而非被导入，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必填参数
    parser.add_argument(
        "--checkpoint",
        default="small",
        type=str,
        help="Checkpoint size of the MusicGen model you'd like to convert. Can be one of: "
             "`['small', 'medium', 'large']` for the mono checkpoints, or "
             "`['facebook/musicgen-stereo-small', 'facebook/musicgen-stereo-medium', 'facebook/musicgen-stereo-large']` "
             "for the stereo checkpoints.",
    )

    # 必填参数
    parser.add_argument(
        "--pytorch_dump_folder",
        required=True,
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )

    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行音乐生成模型的转换
    convert_musicgen_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
```