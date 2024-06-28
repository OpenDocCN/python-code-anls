# `.\models\fastspeech2_conformer\convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置代码文件的编码格式为 UTF-8

# 导入 argparse 模块，用于处理命令行参数
import argparse

# 导入 json 模块，用于处理 JSON 格式数据
import json

# 导入 re 模块，用于正则表达式操作
import re

# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path

# 从 tempfile 模块中导入 TemporaryDirectory 类，用于创建临时目录
from tempfile import TemporaryDirectory

# 导入 torch 库，用于处理 PyTorch 相关功能
import torch

# 导入 yaml 模块，用于处理 YAML 格式数据
import yaml

# 从 transformers 库中导入以下类和函数
from transformers import (
    FastSpeech2ConformerConfig,        # FastSpeech2ConformerConfig 类，用于配置 FastSpeech2Conformer 模型
    FastSpeech2ConformerModel,        # FastSpeech2ConformerModel 类，FastSpeech2Conformer 模型
    FastSpeech2ConformerTokenizer,    # FastSpeech2ConformerTokenizer 类，FastSpeech2Conformer 模型的分词器
    logging                           # logging 模块，用于日志记录
)

# 设置 logging 模块的详细程度为 info
logging.set_verbosity_info()

# 获取日志记录器，用于记录 FastSpeech2Conformer 模型相关的日志信息
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")

# 定义一个映射表，将配置参数映射到新的命名格式，用于兼容旧的配置
CONFIG_MAPPING = {
    "adim": "hidden_size",                                    # adim 映射到 hidden_size
    "aheads": "num_attention_heads",                          # aheads 映射到 num_attention_heads
    "conformer_dec_kernel_size": "decoder_kernel_size",       # conformer_dec_kernel_size 映射到 decoder_kernel_size
    "conformer_enc_kernel_size": "encoder_kernel_size",       # conformer_enc_kernel_size 映射到 encoder_kernel_size
    "decoder_normalize_before": "decoder_normalize_before",   # decoder_normalize_before 映射到 decoder_normalize_before
    "dlayers": "decoder_layers",                              # dlayers 映射到 decoder_layers
    "dunits": "decoder_linear_units",                         # dunits 映射到 decoder_linear_units
    "duration_predictor_chans": "duration_predictor_channels",# duration_predictor_chans 映射到 duration_predictor_channels
    "duration_predictor_kernel_size": "duration_predictor_kernel_size",  # duration_predictor_kernel_size 映射到 duration_predictor_kernel_size
    "duration_predictor_layers": "duration_predictor_layers",# duration_predictor_layers 映射到 duration_predictor_layers
    "elayers": "encoder_layers",                              # elayers 映射到 encoder_layers
    "encoder_normalize_before": "encoder_normalize_before",   # encoder_normalize_before 映射到 encoder_normalize_before
    "energy_embed_dropout": "energy_embed_dropout",           # energy_embed_dropout 映射到 energy_embed_dropout
    "energy_embed_kernel_size": "energy_embed_kernel_size",   # energy_embed_kernel_size 映射到 energy_embed_kernel_size
    "energy_predictor_chans": "energy_predictor_channels",    # energy_predictor_chans 映射到 energy_predictor_channels
    "energy_predictor_dropout": "energy_predictor_dropout",   # energy_predictor_dropout 映射到 energy_predictor_dropout
    "energy_predictor_kernel_size": "energy_predictor_kernel_size",  # energy_predictor_kernel_size 映射到 energy_predictor_kernel_size
    "energy_predictor_layers": "energy_predictor_layers",     # energy_predictor_layers 映射到 energy_predictor_layers
    "eunits": "encoder_linear_units",                         # eunits 映射到 encoder_linear_units
    "pitch_embed_dropout": "pitch_embed_dropout",             # pitch_embed_dropout 映射到 pitch_embed_dropout
    "pitch_embed_kernel_size": "pitch_embed_kernel_size",     # pitch_embed_kernel_size 映射到 pitch_embed_kernel_size
    "pitch_predictor_chans": "pitch_predictor_channels",      # pitch_predictor_chans 映射到 pitch_predictor_channels
    "pitch_predictor_dropout": "pitch_predictor_dropout",     # pitch_predictor_dropout 映射到 pitch_predictor_dropout
    "pitch_predictor_kernel_size": "pitch_predictor_kernel_size",  # pitch_predictor_kernel_size 映射到 pitch_predictor_kernel_size
    "pitch_predictor_layers": "pitch_predictor_layers",       # pitch_predictor_layers 映射到 pitch_predictor_layers
    "positionwise_conv_kernel_size": "positionwise_conv_kernel_size",  # positionwise_conv_kernel_size 映射到 positionwise_conv_kernel_size
    "postnet_chans": "speech_decoder_postnet_units",          # postnet_chans 映射到 speech_decoder_postnet_units
    "postnet_filts": "speech_decoder_postnet_kernel",         # postnet_filts 映射到 speech_decoder_postnet_kernel
    "postnet_layers": "speech_decoder_postnet_layers",        # postnet_layers 映射到 speech_decoder_postnet_layers
    "reduction_factor": "reduction_factor",                   # reduction_factor 映射到 reduction_factor
    "stop_gradient_from_energy_predictor": "stop_gradient_from_energy_predictor",  # stop_gradient_from_energy_predictor 映射到 stop_gradient_from_energy_predictor
    "stop_gradient_from_pitch_predictor": "stop_gradient_from_pitch_predictor",    # stop_gradient_from_pitch_predictor 映射到 stop_gradient_from_pitch_predictor
    "transformer_dec_attn_dropout_rate": "decoder_attention_dropout_rate",  # transformer_dec_attn_dropout_rate 映射到 decoder_attention_dropout_rate
    "transformer_dec_dropout_rate": "decoder_dropout_rate",   # transformer_dec_dropout_rate 映射到 decoder_dropout_rate
    "transformer_dec_positional_dropout_rate": "decoder_positional_dropout_rate",
    # 将配置中的 "transformer_dec_positional_dropout_rate" 映射为 "decoder_positional_dropout_rate"

    "transformer_enc_attn_dropout_rate": "encoder_attention_dropout_rate",
    # 将配置中的 "transformer_enc_attn_dropout_rate" 映射为 "encoder_attention_dropout_rate"

    "transformer_enc_dropout_rate": "encoder_dropout_rate",
    # 将配置中的 "transformer_enc_dropout_rate" 映射为 "encoder_dropout_rate"

    "transformer_enc_positional_dropout_rate": "encoder_positional_dropout_rate",
    # 将配置中的 "transformer_enc_positional_dropout_rate" 映射为 "encoder_positional_dropout_rate"

    "use_cnn_in_conformer": "use_cnn_in_conformer",
    # 指示是否在 Conformer 模型中使用 CNN

    "use_macaron_style_in_conformer": "use_macaron_style_in_conformer",
    # 指示是否在 Conformer 模型中使用 Macaron 风格的结构

    "use_masking": "use_masking",
    # 指示是否使用掩码来进行模型训练

    "use_weighted_masking": "use_weighted_masking",
    # 指示是否使用加权掩码进行模型训练

    "idim": "input_dim",
    # 输入数据的维度

    "odim": "num_mel_bins",
    # 梅尔频谱图的频道数

    "spk_embed_dim": "speaker_embed_dim",
    # 说话人嵌入向量的维度

    "langs": "num_languages",
    # 语言的数量

    "spks": "num_speakers",
    # 说话人的数量
}

# 重新映射模型的 YAML 配置文件
def remap_model_yaml_config(yaml_config_path):
    # 打开并读取 YAML 配置文件
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        # 使用 yaml.safe_load 将 YAML 文件内容加载为 Python 对象
        args = yaml.safe_load(f)
        # 将加载的参数转换为 argparse.Namespace 对象
        args = argparse.Namespace(**args)

    # 初始化一个空的重新映射配置字典
    remapped_config = {}

    # 获取模型参数中的文本到语音转换器参数
    model_params = args.tts_conf["text2mel_params"]
    # 使用 CONFIG_MAPPING 字典进行参数重映射，未包含的键会被忽略
    for espnet_config_key, hf_config_key in CONFIG_MAPPING.items():
        # 如果 espnet_config_key 存在于模型参数中
        if espnet_config_key in model_params:
            # 将映射后的参数加入到 remapped_config 字典中
            remapped_config[hf_config_key] = model_params[espnet_config_key]

    # 返回重新映射后的配置字典，以及 args 对象中的 g2p 和 token_list 属性
    return remapped_config, args.g2p, args.token_list


def convert_espnet_state_dict_to_hf(state_dict):
    # 初始化一个空的新状态字典
    new_state_dict = {}
    # 遍历给定的状态字典（state_dict）中的每个键
    for key in state_dict:
        # 如果键名包含特定子串 "tts.generator.text2mel."
        if "tts.generator.text2mel." in key:
            # 去除键名中的 "tts.generator.text2mel."，得到新的键名
            new_key = key.replace("tts.generator.text2mel.", "")
            
            # 如果键名包含 "postnet"
            if "postnet" in key:
                # 修改新键名，将 "postnet.postnet" 替换为 "speech_decoder_postnet.layers"
                new_key = new_key.replace("postnet.postnet", "speech_decoder_postnet.layers")
                # 根据键名结尾不同的后缀，调整为特定的命名格式
                new_key = new_key.replace(".0.weight", ".conv.weight")
                new_key = new_key.replace(".1.weight", ".batch_norm.weight")
                new_key = new_key.replace(".1.bias", ".batch_norm.bias")
                new_key = new_key.replace(".1.running_mean", ".batch_norm.running_mean")
                new_key = new_key.replace(".1.running_var", ".batch_norm.running_var")
                new_key = new_key.replace(".1.num_batches_tracked", ".batch_norm.num_batches_tracked")
            
            # 如果键名包含 "feat_out"
            if "feat_out" in key:
                # 根据键名是否包含 "weight" 或 "bias"，确定新键名
                if "weight" in key:
                    new_key = "speech_decoder_postnet.feat_out.weight"
                if "bias" in key:
                    new_key = "speech_decoder_postnet.feat_out.bias"
            
            # 如果键名为 "encoder.embed.0.weight"
            if "encoder.embed.0.weight" in key:
                # 将 "0." 替换为空字符串，得到新键名
                new_key = new_key.replace("0.", "")
            
            # 如果键名包含 "w_1"
            if "w_1" in key:
                # 将 "w_1" 替换为 "conv1"
                new_key = new_key.replace("w_1", "conv1")
            
            # 如果键名包含 "w_2"
            if "w_2" in key:
                # 将 "w_2" 替换为 "conv2"
                new_key = new_key.replace("w_2", "conv2")
            
            # 如果键名包含 "predictor.conv"
            if "predictor.conv" in key:
                # 将 ".conv" 替换为 ".conv_layers"
                new_key = new_key.replace(".conv", ".conv_layers")
                # 使用正则表达式模式和替换规则来调整新键名的格式
                pattern = r"(\d)\.(\d)"
                replacement = (
                    r"\1.conv" if ("2.weight" not in new_key) and ("2.bias" not in new_key) else r"\1.layer_norm"
                )
                new_key = re.sub(pattern, replacement, new_key)
            
            # 如果键名中包含 "pitch_embed" 或 "energy_embed"
            if "pitch_embed" in key or "energy_embed" in key:
                # 将 "0" 替换为 "conv"
                new_key = new_key.replace("0", "conv")
            
            # 如果键名中包含 "encoders"
            if "encoders" in key:
                # 替换键名中的 "encoders" 为 "conformer_layers"
                new_key = new_key.replace("encoders", "conformer_layers")
                # 替换其他特定的后缀部分为对应的新命名格式
                new_key = new_key.replace("norm_final", "final_layer_norm")
                new_key = new_key.replace("norm_mha", "self_attn_layer_norm")
                new_key = new_key.replace("norm_ff_macaron", "ff_macaron_layer_norm")
                new_key = new_key.replace("norm_ff", "ff_layer_norm")
                new_key = new_key.replace("norm_conv", "conv_layer_norm")
            
            # 如果键名中包含 "lid_emb"
            if "lid_emb" in key:
                # 将 "lid_emb" 替换为 "language_id_embedding"
                new_key = new_key.replace("lid_emb", "language_id_embedding")
            
            # 如果键名中包含 "sid_emb"
            if "sid_emb" in key:
                # 将 "sid_emb" 替换为 "speaker_id_embedding"
                new_key = new_key.replace("sid_emb", "speaker_id_embedding")
            
            # 将新的键名与原始状态字典中的值对应起来，添加到新的状态字典中
            new_state_dict[new_key] = state_dict[key]

    # 返回经过修改后的新状态字典
    return new_state_dict
# 使用装饰器 @torch.no_grad() 来确保在此函数中不进行梯度计算
@torch.no_grad()
# 定义函数，将 FastSpeech2Conformer 模型的检查点转换为 PyTorch 模型
def convert_FastSpeech2ConformerModel_checkpoint(
    checkpoint_path,  # 原始检查点文件的路径
    yaml_config_path,  # 模型配置文件 config.yaml 的路径
    pytorch_dump_folder_path,  # 输出的 PyTorch 模型文件夹路径
    repo_id=None,  # 可选参数，用于指定上传到 🤗 hub 的 repo ID
):
    # 调用函数 remap_model_yaml_config 读取模型参数、分词器名称及词汇表
    model_params, tokenizer_name, vocab = remap_model_yaml_config(yaml_config_path)
    
    # 根据读取的模型参数创建 FastSpeech2ConformerConfig 配置对象
    config = FastSpeech2ConformerConfig(**model_params)

    # 根据配置对象创建 FastSpeech2ConformerModel 模型
    model = FastSpeech2ConformerModel(config)

    # 加载 ESPnet 模型的检查点文件
    espnet_checkpoint = torch.load(checkpoint_path)
    # 将 ESPnet 模型的状态字典转换为适用于 Hugging Face 的状态字典格式
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)

    # 将转换后的状态字典加载到模型中
    model.load_state_dict(hf_compatible_state_dict)

    # 将模型保存到指定的 PyTorch 模型文件夹路径中
    model.save_pretrained(pytorch_dump_folder_path)

    # 准备分词器
    with TemporaryDirectory() as tempdir:
        # 创建词汇表的索引映射
        vocab = {token: id for id, token in enumerate(vocab)}
        # 创建词汇表文件的路径
        vocab_file = Path(tempdir) / "vocab.json"
        # 将词汇表写入到 JSON 文件中
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        
        # 确定是否需要去除空格
        should_strip_spaces = "no_space" in tokenizer_name
        # 使用 FastSpeech2ConformerTokenizer 创建分词器对象
        tokenizer = FastSpeech2ConformerTokenizer(str(vocab_file), should_strip_spaces=should_strip_spaces)

    # 将分词器保存到指定的 PyTorch 模型文件夹路径中
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了 repo_id，将模型和分词器推送到 🤗 hub 上
    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：原始检查点文件的路径
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数：模型配置文件 config.yaml 的路径
    parser.add_argument("--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert")
    # 添加命令行参数：输出的 PyTorch 模型文件夹路径
    parser.add_argument("--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数：可选参数，用于指定上传到 🤗 hub 的 repo ID
    parser.add_argument("--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub.")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入解析后的命令行参数
    convert_FastSpeech2ConformerModel_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
```