# `.\models\musicgen_melody\convert_musicgen_melody_transformers.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明和许可信息，指明此代码受 Apache License, Version 2.0 的保护
# 该脚本用于将原始存储库中的 Musicgen Melody 检查点转换
"""Convert Musicgen Melody checkpoints from the original repository."""
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
from pathlib import Path  # 提供处理文件路径的类和方法
from typing import Dict, OrderedDict, Tuple  # 引入类型提示

import torch  # PyTorch 深度学习库
from audiocraft.models import MusicGen  # 导入自定义的音乐生成模型

# 从 Transformers 库中导入相关模块和类
from transformers import (
    AutoTokenizer,  # 自动模型令牌化
    EncodecModel,  # 编码器模型（可能是拼写错误，应为EncoderModel）
    T5EncoderModel,  # T5 编码器模型
)
# 导入 Musicgen Melody 的配置、特征提取、模型和处理模块
from transformers.models.musicgen_melody.configuration_musicgen_melody import MusicgenMelodyDecoderConfig
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
    MusicgenMelodyForCausalLM,  # Musicgen Melody 的因果语言建模模型
    MusicgenMelodyForConditionalGeneration,  # Musicgen Melody 的条件生成模型
)
from transformers.models.musicgen_melody.processing_musicgen_melody import MusicgenMelodyProcessor  # 处理 Musicgen Melody 相关任务的模块
from transformers.utils import logging  # 导入日志记录模块

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预期缺失的模型键列表
EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]
# 预期额外的模型键列表
EXPECTED_ADDITIONAL_KEYS = ["condition_provider.conditioners.self_wav.chroma.spec.window"]


# 定义一个函数用于重命名模型参数名
def rename_keys(name):
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
    if "condition_provider.conditioners.self_wav.output_proj" in name:
        name = name.replace("condition_provider.conditioners.self_wav.output_proj", "audio_enc_to_dec_proj")
    return name
# 定义一个函数，用于重命名给定的状态字典，并按照特定的模块名称重新命名。
def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> Tuple[Dict, Dict]:
    """Function that takes the fairseq MusicgenMelody state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that for the
    text encoder projection and for the audio encoder projection."""
    
    # 获取状态字典的所有键
    keys = list(state_dict.keys())
    # 初始化空字典，用于存储编码器-解码器投影和音频编码器到解码器投影之间的状态字典
    enc_dec_proj_state_dict = {}
    audio_enc_to_dec_proj_state_dict = {}
    
    # 遍历状态字典的每个键
    for key in keys:
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 使用自定义函数重命名当前键
        key = rename_keys(key)
        
        # 如果当前键包含 "in_proj_weight"，则拆分融合的qkv投影
        if "in_proj_weight" in key:
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        # 如果当前键包含 "audio_enc_to_dec_proj"，则将其添加到音频编码器到解码器投影状态字典中
        elif "audio_enc_to_dec_proj" in key:
            audio_enc_to_dec_proj_state_dict[key[len("audio_enc_to_dec_proj.") :]] = val
        # 如果当前键包含 "enc_to_dec_proj"，则将其添加到编码器到解码器投影状态字典中
        elif "enc_to_dec_proj" in key:
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        # 否则，将当前键和对应的值添加回状态字典中
        else:
            state_dict[key] = val
    
    # 返回重命名后的状态字典，编码器-解码器投影状态字典和音频编码器到解码器投影状态字典
    return state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict


# 定义一个函数，从给定的检查点加载配置信息并返回 MusicgenMelodyDecoderConfig 对象
def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenMelodyDecoderConfig:
    # 根据给定的检查点名称，设置隐藏大小、隐藏层数、注意力头数等参数
    if checkpoint == "facebook/musicgen-melody" or checkpoint == "facebook/musicgen-stereo-melody":
        hidden_size = 1536
        num_hidden_layers = 48
        num_attention_heads = 24
    elif checkpoint == "facebook/musicgen-melody-large" or checkpoint == "facebook/musicgen-stereo-melody-large":
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 32
    else:
        # 如果检查点名称不在预期范围内，抛出 ValueError 异常
        raise ValueError(
            "Checkpoint should be one of `['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, "
            "or `['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` "
            f"for the stereo checkpoints, got {checkpoint}."
        )
    
    # 根据检查点名称中是否包含 "stereo" 设置音频通道数和码本数
    if "stereo" in checkpoint:
        audio_channels = 2
        num_codebooks = 8
    else:
        audio_channels = 1
        num_codebooks = 4
    
    # 创建并返回一个 MusicgenMelodyDecoderConfig 对象，包含从检查点加载的配置信息
    config = MusicgenMelodyDecoderConfig(
        hidden_size=hidden_size,
        ffn_dim=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_codebooks=num_codebooks,
        audio_channels=audio_channels,
    )
    return config


# 定义一个装饰器，用于声明一个无需计算梯度的函数
@torch.no_grad()
def convert_musicgen_melody_checkpoint(
    checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu", test_same_output=False
):
    # 从预训练模型加载指定的检查点，并将模型移至指定的设备上
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=args.device)
    # 从加载的模型中获取语言模型的状态字典
    decoder_state_dict = fairseq_model.lm.state_dict()
    # 重命名解码器的状态字典，并根据隐藏层大小调整编码-解码投影的状态字典
    decoder_state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict = rename_state_dict(
        decoder_state_dict, hidden_size=decoder_config.hidden_size
    )

    # 使用预训练的T5模型初始化文本编码器
    text_encoder = T5EncoderModel.from_pretrained("t5-base")
    
    # 使用预训练的音频编码器初始化音频编码器
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    
    # 根据给定的解码器配置初始化音乐生成的Melody解码器，并设为评估模式
    decoder = MusicgenMelodyForCausalLM(decoder_config).eval()

    # 加载解码器权重，允许缺少嵌入和编码-解码投影
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)

    # 移除与文本编码器或音频编码器相关的缺失键及期望的缺失键
    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    # 移除与期望的额外键相对应的意外键
    for key in unexpected_keys.copy():
        if key in EXPECTED_ADDITIONAL_KEYS:
            unexpected_keys.remove(key)

    # 如果存在缺失的键，则引发值错误
    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    # 如果存在意外的键，则引发值错误
    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # 初始化组合模型，包括文本编码器、音频编码器和解码器
    model = MusicgenMelodyForConditionalGeneration(
        text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder
    ).to(args.device)

    # 加载预训练的编码-解码投影（从解码器状态字典中）
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)

    # 加载预训练的音频编码器投影（从解码器状态字典中）
    model.audio_enc_to_dec_proj.load_state_dict(audio_enc_to_dec_proj_state_dict)

    # 检查是否可以进行前向传播
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1).to(device)
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1).to(device)

    # 使用torch.no_grad()上下文管理器执行前向传播，获取logits
    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    # 计算预期的输出长度，并检查logits的形状是否符合预期
    output_length = 1 + input_ids.shape[1] + model.config.chroma_length
    if logits.shape != (2 * decoder_config.num_codebooks, output_length, 2048):
        raise ValueError("Incorrect shape for logits")

    # 初始化tokenizer，使用T5-base模型的自动tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # 初始化特征提取器为音乐生成Melody的特征提取器
    feature_extractor = MusicgenMelodyFeatureExtractor()

    # 初始化processor，使用特征提取器和tokenizer
    processor = MusicgenMelodyProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 设置适当的开始/填充token id
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048

    # 设置其他默认的生成配置参数
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0
    # 如果需要测试输出是否与原始模型相同
    if test_same_output:
        # 准备用于解码的输入张量，全部填充为模型的填充标记ID
        decoder_input_ids = torch.ones_like(decoder_input_ids).to(device) * model.generation_config.pad_token_id
        
        # 禁止梯度计算的上下文
        with torch.no_grad():
            # 限制解码器输入的长度，仅保留前 decoder_config.num_codebooks 个位置
            decoder_input_ids = decoder_input_ids[: decoder_config.num_codebooks]
            
            # 使用processor对文本进行处理，返回PyTorch张量格式的输入数据
            inputs = processor(text=["gen"], return_tensors="pt", padding=True).to(device)
            
            # 使用模型生成logits，给定解码器的输入张量
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits

            # 准备fairseq模型的tokens和attributes用于生成
            attributes, prompt_tokens = fairseq_model._prepare_tokens_and_attributes(["gen"], None)
            
            # 使用fairseq模型进行前向推断，计算原始模型的logits
            original_logits = fairseq_model.lm.forward(
                decoder_input_ids.reshape(1, decoder_config.num_codebooks, -1), attributes
            )

            # 使用torch的测试工具断言，检查生成的logits与原始模型的logits在数值上的接近度
            torch.testing.assert_close(
                original_logits.squeeze(2).reshape(decoder_config.num_codebooks, -1),
                logits[:, -1],
                rtol=1e-5,
                atol=5e-5,
            )

    # 如果提供了pytorch_dump_folder路径，则保存模型和processor的配置到指定目录
    if pytorch_dump_folder is not None:
        # 如果路径不存在，则创建该目录
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        
        # 记录日志，指示将模型保存到指定目录
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        
        # 保存模型的预训练配置到指定目录
        model.save_pretrained(pytorch_dump_folder)
        
        # 保存processor的配置到指定目录
        processor.save_pretrained(pytorch_dump_folder)

    # 如果提供了repo_id，则将模型和processor推送到指定的Hub repo中
    if repo_id:
        # 记录日志，指示将模型推送到指定的Hub repo中
        logger.info(f"Pushing model {checkpoint} to {repo_id}")
        
        # 将模型推送到指定的Hub repo中，并创建pull request
        model.push_to_hub(repo_id, create_pr=True)
        
        # 将processor推送到指定的Hub repo中，并创建pull request
        processor.push_to_hub(repo_id, create_pr=True)
if __name__ == "__main__":
    # 如果脚本作为主程序执行，进入主程序入口

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint",
        default="facebook/musicgen-melody",
        type=str,
        help="Checkpoint size of the Musicgen Melody model you'd like to convert. Can be one of: "
        "`['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, or "
        "`['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` "
        "for the stereo checkpoints.",
    )
    # 添加必选参数--checkpoint，指定要转换的 Musicgen Melody 模型的检查点位置

    parser.add_argument(
        "--pytorch_dump_folder",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加可选参数--pytorch_dump_folder，指定输出的 PyTorch 模型保存路径

    parser.add_argument(
        "--push_to_hub",
        default="musicgen-melody",
        type=str,
        help="Where to upload the converted model on the 🤗 hub.",
    )
    # 添加可选参数--push_to_hub，指定在 🤗 hub 上上传转换后的模型的位置标识

    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )
    # 添加可选参数--device，指定转换过程中使用的 Torch 设备，可以是 cpu 或 cuda

    parser.add_argument("--test_same_output", default=False, type=bool, help="If `True`, test if same output logits.")
    # 添加可选参数--test_same_output，如果设置为 True，则测试是否输出相同的 logits

    args = parser.parse_args()
    # 解析命令行参数并返回解析后的参数对象 args

    convert_musicgen_melody_checkpoint(
        args.checkpoint, args.pytorch_dump_folder, args.push_to_hub, args.device, args.test_same_output
    )
    # 调用函数 convert_musicgen_melody_checkpoint，传入解析后的参数，执行 Musicgen Melody 模型的转换操作
```