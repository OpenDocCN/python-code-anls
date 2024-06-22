# `.\transformers\models\audio_spectrogram_transformer\convert_audio_spectrogram_transformer_original_to_pytorch.py`

```
# 设置脚本编码为 UTF-8
# 版权声明和许可信息
# Copyright 2022 The HuggingFace Inc. team.
# 根据 Apache License, Version 2.0 许可证授权，除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获得许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，本软件是基于“按原样”提供的，无任何明示或暗示的保证或条件。
# 请参阅许可证了解特定语言下的权限和限制。
"""从原始存储库转换音频频谱变换器检查点。URL: https://github.com/YuanGongND/ast"""

# 导入模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式的数据
from pathlib import Path  # 用于处理文件路径

import torch  # PyTorch 深度学习库
import torchaudio  # 用于音频处理的 PyTorch 扩展
from datasets import load_dataset  # 加载数据集的函数
from huggingface_hub import hf_hub_download  # 用于从 Hugging Face Hub 下载文件的函数

from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification  # 用于音频分类的 AST 模型
from transformers.utils import logging  # 用于日志记录的工具模块

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 函数：根据模型名称获取音频频谱变换器的配置信息
def get_audio_spectrogram_transformer_config(model_name):
    # 创建一个 ASTConfig 实例对象
    config = ASTConfig()

    # 根据模型名称设置配置参数
    if "10-10" in model_name:
        pass
    elif "speech-commands" in model_name:
        config.max_length = 128  # 设置最大长度为 128
    elif "12-12" in model_name:
        config.time_stride = 12  # 设置时间步长为 12
        config.frequency_stride = 12  # 设置频率步长为 12
    elif "14-14" in model_name:
        config.time_stride = 14  # 设置时间步长为 14
        config.frequency_stride = 14  # 设置频率步长为 14
    elif "16-16" in model_name:
        config.time_stride = 16  # 设置时间步长为 16
        config.frequency_stride = 16  # 设置频率步长为 16
    else:
        raise ValueError("Model not supported")  # 抛出值错误异常，模型不受支持

    # 用于从 Hugging Face Hub 下载文件的存储库 ID
    repo_id = "huggingface/label-files"
    # 根据模型名称设置配置参数
    if "speech-commands" in model_name:
        config.num_labels = 35  # 设置标签数量为 35
        filename = "speech-commands-v2-id2label.json"  # 设置文件名
    else:
        config.num_labels = 527  # 设置标签数量为 527
        filename = "audioset-id2label.json"  # 设置文件名

    # 从 Hugging Face Hub 下载标签文件，并加载为 JSON 数据
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将标签 ID 转换为整数，并构建 ID 到标签的映射字典
    id2label = {int(k): v for k, v in id2label.items()}
    # 设置配置对象中的 ID 到标签的映射字典
    config.id2label = id2label
    # 设置配置对象中的标签到 ID 的映射字典
    config.label2id = {v: k for k, v in id2label.items()}

    # 返回配置对象
    return config

# 函数：重命名键名
def rename_key(name):
    # 替换键名中的字符串
    if "module.v" in name:
        name = name.replace("module.v", "audio_spectrogram_transformer")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "dist_token" in name:
        name = name.replace("dist_token", "embeddings.distillation_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # 替换键名中的字符串，用于转换器块
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    # 如果文件名中包含"norm1"，则替换为"layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果文件名中包含"norm2"，则替换为"layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果文件名中包含"mlp.fc1"，则替换为"intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果文件名中包含"mlp.fc2"，则替换为"output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果文件名中包含"audio_spectrogram_transformer.norm"，则替换为"audio_spectrogram_transformer.layernorm"
    if "audio_spectrogram_transformer.norm" in name:
        name = name.replace("audio_spectrogram_transformer.norm", "audio_spectrogram_transformer.layernorm")
    # 如果文件名中包含"module.mlp_head.0"，则替换为"classifier.layernorm"
    if "module.mlp_head.0" in name:
        name = name.replace("module.mlp_head.0", "classifier.layernorm")
    # 如果文件名中包含"module.mlp_head.1"，则替换为"classifier.dense"
    if "module.mlp_head.1" in name:
        name = name.replace("module.mlp_head.1", "classifier.dense")

    # 返回处理后的文件名
    return name
# 将原始状态字典转换为适用于新模型结构的状态字典
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的键的副本
    for key in orig_state_dict.copy().keys():
        # 弹出键对应的值
        val = orig_state_dict.pop(key)

        # 如果键中包含"qkv"
        if "qkv" in key:
            # 拆分键名
            key_split = key.split(".")
            # 获取层编号
            layer_num = int(key_split[3])
            # 获取隐藏层大小
            dim = config.hidden_size
            # 如果键中包含"weight"
            if "weight" in key:
                # 更新新键值对应的值
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新新键值对应的值
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.bias"
                ] = val[-dim:]
        else:
            # 更新新键值对应的值
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的状态字典
    return orig_state_dict


# 移除指定键的值
def remove_keys(state_dict):
    # 需要忽略的键列表
    ignore_keys = [
        "module.v.head.weight",
        "module.v.head.bias",
        "module.v.head_dist.weight",
        "module.v.head_dist.bias",
    ]
    # 遍历忽略的键列表
    for k in ignore_keys:
        # 弹出指定键的值
        state_dict.pop(k, None)


# 无需梯度计算
@torch.no_grad()
def convert_audio_spectrogram_transformer_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Audio Spectrogram Transformer structure.
    """
    # 获取音频频谱变换器的配置
    config = get_audio_spectrogram_transformer_config(model_name)
    # 定义一个字典，将模型名称映射到对应的下载链接
    model_name_to_url = {
        "ast-finetuned-audioset-10-10-0.4593": (
            "https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1"
        ),
        "ast-finetuned-audioset-10-10-0.450": (
            "https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1"
        ),
        "ast-finetuned-audioset-10-10-0.448": (
            "https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1"
        ),
        "ast-finetuned-audioset-10-10-0.448-v2": (
            "https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1"
        ),
        "ast-finetuned-audioset-12-12-0.447": (
            "https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1"
        ),
        "ast-finetuned-audioset-14-14-0.443": (
            "https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1"
        ),
        "ast-finetuned-audioset-16-16-0.442": (
            "https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1"
        ),
        "ast-finetuned-speech-commands-v2": (
            "https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1"
        ),
    }

    # 加载原始状态字典
    checkpoint_url = model_name_to_url[model_name]
    # 从指定 URL 加载 PyTorch 模型的状态字典，并指定映射到 CPU
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 移除一些键
    remove_keys(state_dict)
    # 重命名一些键
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载 🤗 模型
    model = ASTForAudioClassification(config)
    # 将模型设置为评估模式
    model.eval()

    # 加载模型的状态字典
    model.load_state_dict(new_state_dict)

    # 在虚拟输入上验证输出
    # 来源: https://github.com/YuanGongND/ast/blob/79e873b8a54d0a3b330dd522584ff2b9926cd581/src/run.py#L62
    # 如果模型名称中不包含"speech-commands"，则设置平均值为-4.2677393，否则设置为-6.845978
    mean = -4.2677393 if "speech-commands" not in model_name else -6.845978
    # 如果模型名称中不包含"speech-commands"，则设置标准差为4.5689974，否则设置为5.5654526
    std = 4.5689974 if "speech-commands" not in model_name else 5.5654526
    # 如果模型名称中不包含"speech-commands"，则设置最大长度为1024，否则设置为128
    max_length = 1024 if "speech-commands" not in model_name else 128
    # 创建一个 AST 特征提取器对象，设置均值、标准差和最大长度
    feature_extractor = ASTFeatureExtractor(mean=mean, std=std, max_length=max_length)

    # 如果模型名称中包含"speech-commands"，则加载 speech_commands 数据集的验证集，获取第一个音频文件的波形
    if "speech-commands" in model_name:
        dataset = load_dataset("speech_commands", "v0.02", split="validation")
        waveform = dataset[0]["audio"]["array"]
    else:
        # 否则，下载 nielsr/audio-spectogram-transformer-checkpoint 仓库中的样本音频文件
        filepath = hf_hub_download(
            repo_id="nielsr/audio-spectogram-transformer-checkpoint",
            filename="sample_audio.flac",
            repo_type="dataset",
        )
        # 加载音频文件，返回音频数据和采样率，并将数据转换为 NumPy 数组
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.squeeze().numpy()

    # 使用特征提取器处理音频数据，返回 PyTorch 张量格式的输入
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    # 前向传播
    outputs = model(**inputs)
    # 获取模型输出的 logits
    logits = outputs.logits

    # 如果模型名称为"ast-finetuned-audioset-10-10-0.4593"，则设置预期切片为[-0.8760, -7.0042, -8.6602]
    if model_name == "ast-finetuned-audioset-10-10-0.4593":
        expected_slice = torch.tensor([-0.8760, -7.0042, -8.6602])
    # 如果模型名称为"ast-finetuned-audioset-10-10-0.450"，则设置预期切片为[-1.1986, -7.0903, -8.2718]
    elif model_name == "ast-finetuned-audioset-10-10-0.450":
        expected_slice = torch.tensor([-1.1986, -7.0903, -8.2718])
    # 如果模型名称为"ast-finetuned-audioset-10-10-0.448"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-audioset-10-10-0.448":
        expected_slice = torch.tensor([-2.6128, -8.0080, -9.4344])
    # 如果模型名称为"ast-finetuned-audioset-10-10-0.448-v2"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-audioset-10-10-0.448-v2":
        expected_slice = torch.tensor([-1.5080, -7.4534, -8.8917])
    # 如果模型名称为"ast-finetuned-audioset-12-12-0.447"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-audioset-12-12-0.447":
        expected_slice = torch.tensor([-0.5050, -6.5833, -8.0843])
    # 如果模型名称为"ast-finetuned-audioset-14-14-0.443"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-audioset-14-14-0.443":
        expected_slice = torch.tensor([-0.3826, -7.0336, -8.2413])
    # 如果模型名称为"ast-finetuned-audioset-16-16-0.442"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-audioset-16-16-0.442":
        expected_slice = torch.tensor([-1.2113, -6.9101, -8.3470])
    # 如果模型名称为"ast-finetuned-speech-commands-v2"，设置预期切片为给定张量
    elif model_name == "ast-finetuned-speech-commands-v2":
        expected_slice = torch.tensor([6.1589, -8.0566, -8.7984])
    else:
        # 如果模型名称未知，则引发值错误异常
        raise ValueError("Unknown model name")
    # 如果模型输出的前三个值不与预期切片接近（绝对误差小于等于1e-4），则引发值错误异常
    if not torch.allclose(logits[0, :3], expected_slice, atol=1e-4):
        raise ValueError("Logits don't match")
    # 输出提示信息
    print("Looks ok!")

    # 如果 PyTorch 模型文件夹路径不为空，则执行以下操作
    if pytorch_dump_folder_path is not None:
        # 创建 PyTorch 模型文件夹路径，如果已存在则不做任何操作
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 输出保存模型的提示信息
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 输出保存特征提取器的提示信息
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # 将特征提取器保存到指定路径
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub，则执行以下操作
    if push_to_hub:
        # 输出推送模型和特征提取器到 Hub 的提示信息
        print("Pushing model and feature extractor to the hub...")
        # 将模型推送到 Hub
        model.push_to_hub(f"MIT/{model_name}")
        # 将特征提取器推送到 Hub
        feature_extractor.push_to_hub(f"MIT/{model_name}")
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_name",  # 模型名称参数
        default="ast-finetuned-audioset-10-10-0.4593",  # 默认模型名称
        type=str,  # 参数类型为字符串
        help="Name of the Audio Spectrogram Transformer model you'd like to convert."  # 参数说明
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch 模型输出文件夹路径参数
        default=None,  # 默认为空
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 参数说明
    )
    parser.add_argument(
        "--push_to_hub",  # 推送至🤗 hub 参数
        action="store_true",  # 设置为 True 时触发该参数
        help="Whether or not to push the converted model to the 🤗 hub."  # 参数说明
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将音频频谱变换器检查点转换为 PyTorch 模型
    convert_audio_spectrogram_transformer_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```