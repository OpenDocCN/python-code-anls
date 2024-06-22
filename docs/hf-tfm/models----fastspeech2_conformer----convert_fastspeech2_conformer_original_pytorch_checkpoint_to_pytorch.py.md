# `.\models\fastspeech2_conformer\convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 该代码版权归 2023 年的 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 进行许可
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 无任何形式的担保或条件，包括但不限于任何隐含的担保
# 或适销性、特定用途适用性和非侵权性的担保
"""将 FastSpeech2Conformer 检查点转换为其他格式。"""

# 导入模块
import argparse  # 用于解析命令行参数的库
import json  # 用于处理 JSON 格式的数据的库
import re  # 用于正则表达式匹配的库
from pathlib import Path  # 用于处理文件路径的库
from tempfile import TemporaryDirectory  # 用于创建临时目录的库

import torch  # PyTorch 深度学习库

import yaml  # 用于处理 YAML 格式的数据的库

from transformers import (  # 从 transformers 库中导入以下类和函数
    FastSpeech2ConformerConfig,  # FastSpeech2Conformer 模型配置类
    FastSpeech2ConformerModel,  # FastSpeech2Conformer 模型类
    FastSpeech2ConformerTokenizer,  # FastSpeech2Conformer 模型分词器类
    logging,  # transformers 库中的日志记录模块
)

# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")

# 定义配置映射关系，将旧配置名称映射到新配置名称
CONFIG_MAPPING = {
    "adim": "hidden_size",  # 隐藏层大小
    "aheads": "num_attention_heads",  # 注意力头数
    "conformer_dec_kernel_size": "decoder_kernel_size",  # 解码器内核大小
    "conformer_enc_kernel_size": "encoder_kernel_size",  # 编码器内核大小
    "decoder_normalize_before": "decoder_normalize_before",  # 解码器前归一化
    "dlayers": "decoder_layers",  # 解码器层数
    "dunits": "decoder_linear_units",  # 解码器线性单元数
    "duration_predictor_chans": "duration_predictor_channels",  # 时长预测器通道数
    "duration_predictor_kernel_size": "duration_predictor_kernel_size",  # 时长预测器内核大小
    "duration_predictor_layers": "duration_predictor_layers",  # 时长预测器层数
    "elayers": "encoder_layers",  # 编码器层数
    "encoder_normalize_before": "encoder_normalize_before",  # 编码器前归一化
    "energy_embed_dropout": "energy_embed_dropout",  # 能量嵌入丢失率
    "energy_embed_kernel_size": "energy_embed_kernel_size",  # 能量嵌入内核大小
    "energy_predictor_chans": "energy_predictor_channels",  # 能量预测器通道数
    "energy_predictor_dropout": "energy_predictor_dropout",  # 能量预测器丢失率
    "energy_predictor_kernel_size": "energy_predictor_kernel_size",  # 能量预测器内核大小
    "energy_predictor_layers": "energy_predictor_layers",  # 能量预测器层数
    "eunits": "encoder_linear_units",  # 编码器线性单元数
    "pitch_embed_dropout": "pitch_embed_dropout",  # 音高嵌入丢失率
    "pitch_embed_kernel_size": "pitch_embed_kernel_size",  # 音高嵌入内核大小
    "pitch_predictor_chans": "pitch_predictor_channels",  # 音高预测器通道数
    "pitch_predictor_dropout": "pitch_predictor_dropout",  # 音高预测器丢失率
    "pitch_predictor_kernel_size": "pitch_predictor_kernel_size",  # 音高预测器内核大小
    "pitch_predictor_layers": "pitch_predictor_layers",  # 音高预测器层数
    "positionwise_conv_kernel_size": "positionwise_conv_kernel_size",  # 位置卷积内核大小
    "postnet_chans": "speech_decoder_postnet_units",  # 后网络通道数
    "postnet_filts": "speech_decoder_postnet_kernel",  # 后网络内核大小
    "postnet_layers": "speech_decoder_postnet_layers",  # 后网络层数
    "reduction_factor": "reduction_factor",  # 缩小因子
    "stop_gradient_from_energy_predictor": "stop_gradient_from_energy_predictor",  # 从能量预测器停止梯度
    "stop_gradient_from_pitch_predictor": "stop_gradient_from_pitch_predictor",  # 从音高预测器停止梯度
    "transformer_dec_attn_dropout_rate": "decoder_attention_dropout_rate",  # 解码器注意力丢失率
    "transformer_dec_dropout_rate": "decoder_dropout_rate",  # 解码器丢失率
}
    "transformer_dec_positional_dropout_rate": "decoder_positional_dropout_rate",  
    # 将配置参数中的"transformer_dec_positional_dropout_rate"映射为"decoder_positional_dropout_rate"
    
    "transformer_enc_attn_dropout_rate": "encoder_attention_dropout_rate",  
    # 将配置参数中的"transformer_enc_attn_dropout_rate"映射为"encoder_attention_dropout_rate"
    
    "transformer_enc_dropout_rate": "encoder_dropout_rate",  
    # 将配置参数中的"transformer_enc_dropout_rate"映射为"encoder_dropout_rate"
    
    "transformer_enc_positional_dropout_rate": "encoder_positional_dropout_rate",  
    # 将配置参数中的"transformer_enc_positional_dropout_rate"映射为"encoder_positional_dropout_rate"
    
    "use_cnn_in_conformer": "use_cnn_in_conformer",  
    # 将配置参数中的"use_cnn_in_conformer"映射为"use_cnn_in_conformer"
    
    "use_macaron_style_in_conformer": "use_macaron_style_in_conformer",  
    # 将配置参数中的"use_macaron_style_in_conformer"映射为"use_macaron_style_in_conformer"
    
    "use_masking": "use_masking",  
    # 将配置参数中的"use_masking"映射为"use_masking"
    
    "use_weighted_masking": "use_weighted_masking",  
    # 将配置参数中的"use_weighted_masking"映射为"use_weighted_masking"
    
    "idim": "input_dim",  
    # 将配置参数中的"idim"映射为"input_dim"
    
    "odim": "num_mel_bins",  
    # 将配置参数中的"odim"映射为"num_mel_bins"
    
    "spk_embed_dim": "speaker_embed_dim",  
    # 将配置参数中的"spk_embed_dim"映射为"speaker_embed_dim"
    
    "langs": "num_languages",  
    # 将配置参数中的"langs"映射为"num_languages"
    
    "spks": "num_speakers",  
    # 将配置参数中的"spks"映射为"num_speakers"
# 重新映射 ESPnet 模型的 YAML 配置文件
def remap_model_yaml_config(yaml_config_path):
    # 使用 UTF-8 编码打开 YAML 配置文件
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        # 使用 PyYAML 加载 YAML 文件内容，并转换为字典类型的参数
        args = yaml.safe_load(f)
        # 将参数字典转换为 argparse.Namespace 对象
        args = argparse.Namespace(**args)

    # 初始化一个空字典来存储重新映射后的配置
    remapped_config = {}

    # 从参数中获取 TTS 配置中的 text2mel_params
    model_params = args.tts_conf["text2mel_params"]
    # 遍历 CONFIG_MAPPING 中的映射关系，将 ESPnet 配置键映射到 Hugging Face 配置键
    for espnet_config_key, hf_config_key in CONFIG_MAPPING.items():
        # 如果 ESPnet 配置键在模型参数中存在
        if espnet_config_key in model_params:
            # 将 ESPnet 的配置键及其对应值映射到 Hugging Face 的配置键及其对应值
            remapped_config[hf_config_key] = model_params[espnet_config_key]

    # 返回重新映射后的配置、args.g2p 和 args.token_list
    return remapped_config, args.g2p, args.token_list


# 将 ESPnet 的状态字典转换为 Hugging Face 的状态字典
def convert_espnet_state_dict_to_hf(state_dict):
    # 初始化一个空字典来存储新的状态字典
    new_state_dict = {}
    # 遍历给定的 state_dict 字典
    for key in state_dict:
        # 检查当前 key 是否包含特定字符串
        if "tts.generator.text2mel." in key:
            # 替换 key 中的特定字符串
            new_key = key.replace("tts.generator.text2mel.", "")
            # 如果 key 中包含 "postnet"
            if "postnet" in key:
                # 替换 key 中的特定字符串，并调整格式
                new_key = new_key.replace("postnet.postnet", "speech_decoder_postnet.layers")
                new_key = new_key.replace(".0.weight", ".conv.weight")
                new_key = new_key.replace(".1.weight", ".batch_norm.weight")
                new_key = new_key.replace(".1.bias", ".batch_norm.bias")
                new_key = new_key.replace(".1.running_mean", ".batch_norm.running_mean")
                new_key = new_key.replace(".1.running_var", ".batch_norm.running_var")
                new_key = new_key.replace(".1.num_batches_tracked", ".batch_norm.num_batches_tracked")
            # 如果 key 中包含 "feat_out"
            if "feat_out" in key:
                # 根据 key 中的特定文本设置新的 key
                if "weight" in key:
                    new_key = "speech_decoder_postnet.feat_out.weight"
                if "bias" in key:
                    new_key = "speech_decoder_postnet.feat_out.bias"
            # 如果 key 中包含 "encoder.embed.0.weight"
            if "encoder.embed.0.weight" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("0.", "")
            # 如果 key 中包含 "w_1"
            if "w_1" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("w_1", "conv1")
            # 如果 key 中包含 "w_2"
            if "w_2" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("w_2", "conv2")
            # 如果 key 中包含 "predictor.conv"
            if "predictor.conv" in key:
                # 替换 key 中的特定字符串，并根据条件选择替换格式
                new_key = new_key.replace(".conv", ".conv_layers")
                pattern = r"(\d)\.(\d)"
                replacement = (
                    r"\1.conv" if ("2.weight" not in new_key) and ("2.bias" not in new_key) else r"\1.layer_norm"
                )
                new_key = re.sub(pattern, replacement, new_key)
            # 如果 key 中包含 "pitch_embed" 或 "energy_embed"
            if "pitch_embed" in key or "energy_embed" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("0", "conv")
            # 如果 key 中包含 "encoders"
            if "encoders" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("encoders", "conformer_layers")
                new_key = new_key.replace("norm_final", "final_layer_norm")
                new_key = new_key.replace("norm_mha", "self_attn_layer_norm")
                new_key = new_key.replace("norm_ff_macaron", "ff_macaron_layer_norm")
                new_key = new_key.replace("norm_ff", "ff_layer_norm")
                new_key = new_key.replace("norm_conv", "conv_layer_norm")
            # 如果 key 中包含 "lid_emb"
            if "lid_emb" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("lid_emb", "language_id_embedding")
            # 如果 key 中包含 "sid_emb"
            if "sid_emb" in key:
                # 替换 key 中的特定字符串
                new_key = new_key.replace("sid_emb", "speaker_id_embedding")

            # 将新的 key 和对应的数值存入新的 state_dict 中
            new_state_dict[new_key] = state_dict[key]

    # 返回处理后的新 state_dict
    return new_state_dict
# 导入相关的库
@torch.no_grad()
# 忽略梯度计算
def convert_FastSpeech2ConformerModel_checkpoint(
    checkpoint_path, # 要转换的原始检查点路径
    yaml_config_path, # 要转换的模型的配置文件路径
    pytorch_dump_folder_path, # 输出 PyTorch 模型的文件夹路径
    repo_id=None, # 要上传到模型库的 ID，可选
):
    # 基于模型配置文件重建模型参数、分词器名称和词汇表
    model_params, tokenizer_name, vocab = remap_model_yaml_config(yaml_config_path)
    # 基于模型参数创建 FastSpeech2ConformerConfig 对象
    config = FastSpeech2ConformerConfig(**model_params)

    # 准备模型
    model = FastSpeech2ConformerModel(config)

    # 加载 ESPnet 检查点
    espnet_checkpoint = torch.load(checkpoint_path)
    # 将 ESPnet 检查点的参数转换为适配 Hugging Face 模型的格式
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)

    # 加载模型参数
    model.load_state_dict(hf_compatible_state_dict)

    # 保存 PyTorch 模型
    model.save_pretrained(pytorch_dump_folder_path)

    # 准备分词器
    with TemporaryDirectory() as tempdir:
        # 将词汇表转换为 ID 到标记的映射
        vocab = {token: id for id, token in enumerate(vocab)}
        # 在临时目录下创建词汇表文件
        vocab_file = Path(tempdir) / "vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        # 根据词汇表文件和分词器名称创建分词器
        should_strip_spaces = "no_space" in tokenizer_name
        tokenizer = FastSpeech2ConformerTokenizer(str(vocab_file), should_strip_spaces=should_strip_spaces)

    # 保存分词器
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 如果有提供 repo_id，则将模型和分词器上传到模型库
    if repo_id:
        # 打印提示信息
        print("Pushing to the hub...")
        # 将模型上传到模型库
        model.push_to_hub(repo_id)
        # 将分词器上传到模型库
        tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：原始检查点路径
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数：要转换的模型的配置文件路径
    parser.add_argument(
        "--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert"
    )
    # 添加命令行参数：输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数：上传转换后的模型到模型库的位置
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数进行模型转换
    convert_FastSpeech2ConformerModel_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
```