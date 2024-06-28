# `.\models\audio_spectrogram_transformer\convert_audio_spectrogram_transformer_original_to_pytorch.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，声明代码归 HuggingFace Inc. 团队所有，采用 Apache License 2.0 版本进行许可
# 除非符合许可，否则不得使用该文件
# 可在以下链接获取许可协议内容：http://www.apache.org/licenses/LICENSE-2.0

"""从原始仓库转换音频频谱变换器检查点。URL: https://github.com/YuanGongND/ast"""

# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式数据的模块
from pathlib import Path  # 导入处理路径操作的模块

import torch  # 导入 PyTorch 深度学习库
import torchaudio  # 导入处理音频数据的 PyTorch 扩展模块
from datasets import load_dataset  # 导入加载数据集的函数
from huggingface_hub import hf_hub_download  # 导入下载 Hugging Face Hub 模型的函数

from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification  # 导入音频频谱变换器相关的类
from transformers.utils import logging  # 导入日志记录工具

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)


def get_audio_spectrogram_transformer_config(model_name):
    # 创建一个音频频谱变换器配置对象
    config = ASTConfig()

    # 根据模型名称设置不同的配置参数
    if "10-10" in model_name:
        pass  # 如果模型名称包含 "10-10"，则不修改配置
    elif "speech-commands" in model_name:
        config.max_length = 128  # 如果模型名称包含 "speech-commands"，设置最大长度为 128
    elif "12-12" in model_name:
        config.time_stride = 12  # 如果模型名称包含 "12-12"，设置时间步长为 12
        config.frequency_stride = 12  # 设置频率步长为 12
    elif "14-14" in model_name:
        config.time_stride = 14  # 如果模型名称包含 "14-14"，设置时间步长为 14
        config.frequency_stride = 14  # 设置频率步长为 14
    elif "16-16" in model_name:
        config.time_stride = 16  # 如果模型名称包含 "16-16"，设置时间步长为 16
        config.frequency_stride = 16  # 设置频率步长为 16
    else:
        raise ValueError("Model not supported")  # 如果模型名称不在支持列表中，抛出数值错误异常

    # 设置仓库 ID 用于下载标签文件
    repo_id = "huggingface/label-files"

    # 根据模型名称进一步设置配置对象的属性
    if "speech-commands" in model_name:
        config.num_labels = 35  # 如果模型名称包含 "speech-commands"，设置标签数量为 35
        filename = "speech-commands-v2-id2label.json"  # 设置要下载的标签文件名
    else:
        config.num_labels = 527  # 否则，设置标签数量为 527
        filename = "audioset-id2label.json"  # 设置要下载的标签文件名

    # 使用 Hugging Face Hub 下载指定仓库 ID 和文件名的 JSON 文件，并加载为 Python 字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将加载的标签字典中的键转换为整数类型，值保持不变
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label  # 将转换后的标签字典赋值给配置对象的 id2label 属性
    config.label2id = {v: k for k, v in id2label.items()}  # 创建标签到 ID 的反向映射字典

    return config  # 返回配置对象


def rename_key(name):
    # 根据特定规则重命名输入的键名字符串，并返回重命名后的结果

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
    # 替换 transformer blocks 相关的键名
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")

    # 返回修改后的键名字符串
    return name
    # 如果变量 name 中包含字符串 "norm1"，则将其替换为 "layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果变量 name 中包含字符串 "norm2"，则将其替换为 "layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果变量 name 中包含字符串 "mlp.fc1"，则将其替换为 "intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果变量 name 中包含字符串 "mlp.fc2"，则将其替换为 "output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果变量 name 中包含字符串 "audio_spectrogram_transformer.norm"，则将其替换为 "audio_spectrogram_transformer.layernorm"
    # 这一步是为了兼容不同命名规范下的模型参数
    if "audio_spectrogram_transformer.norm" in name:
        name = name.replace("audio_spectrogram_transformer.norm", "audio_spectrogram_transformer.layernorm")
    # 如果变量 name 中包含字符串 "module.mlp_head.0"，则将其替换为 "classifier.layernorm"
    # 这一步是为了重命名分类器头部的层归一化层
    if "module.mlp_head.0" in name:
        name = name.replace("module.mlp_head.0", "classifier.layernorm")
    # 如果变量 name 中包含字符串 "module.mlp_head.1"，则将其替换为 "classifier.dense"
    # 这一步是为了重命名分类器头部的全连接层
    if "module.mlp_head.1" in name:
        name = name.replace("module.mlp_head.1", "classifier.dense")

    # 返回经过处理的最终变量 name
    return name
# 将原始状态字典转换为新配置的状态字典
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的拷贝中的每个键
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名包含 "qkv"
        if "qkv" in key:
            # 根据 "." 分割键名
            key_split = key.split(".")
            # 获取层号，这里假设层号在第4个位置
            layer_num = int(key_split[3])
            # 获取隐藏层大小
            dim = config.hidden_size
            # 如果键名包含 "weight"
            if "weight" in key:
                # 更新状态字典中的 query、key、value 的权重参数
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
                # 更新状态字典中的 query、key、value 的偏置参数
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
            # 如果键名不包含 "qkv"，则重命名键并保留其对应的值
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict


# 从状态字典中移除指定的键
def remove_keys(state_dict):
    # 需要忽略的键列表
    ignore_keys = [
        "module.v.head.weight",
        "module.v.head.bias",
        "module.v.head_dist.weight",
        "module.v.head_dist.bias",
    ]
    # 遍历忽略键列表，从状态字典中移除对应的键
    for k in ignore_keys:
        state_dict.pop(k, None)


# 在没有梯度更新的情况下执行函数
@torch.no_grad()
def convert_audio_spectrogram_transformer_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    将模型的权重复制/粘贴/调整到我们的音频频谱变换器结构中。
    """
    # 获取音频频谱变换器的配置
    config = get_audio_spectrogram_transformer_config(model_name)
    # 模型名称到预训练模型权重文件下载链接的映射字典
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
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 移除部分键
    remove_keys(state_dict)
    # 重命名部分键
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载 🤗 模型
    model = ASTForAudioClassification(config)
    model.eval()

    model.load_state_dict(new_state_dict)

    # 在虚拟输入上验证输出
    # 来源：https://github.com/YuanGongND/ast/blob/79e873b8a54d0a3b330dd522584ff2b9926cd581/src/run.py#L62
    mean = -4.2677393 if "speech-commands" not in model_name else -6.845978
    std = 4.5689974 if "speech-commands" not in model_name else 5.5654526
    max_length = 1024 if "speech-commands" not in model_name else 128
    feature_extractor = ASTFeatureExtractor(mean=mean, std=std, max_length=max_length)

    if "speech-commands" in model_name:
        # 加载 "speech-commands" 数据集的验证集
        dataset = load_dataset("speech_commands", "v0.02", split="validation")
        waveform = dataset[0]["audio"]["array"]
    else:
        # 下载样本音频文件
        filepath = hf_hub_download(
            repo_id="nielsr/audio-spectogram-transformer-checkpoint",
            filename="sample_audio.flac",
            repo_type="dataset",
        )

        # 加载音频文件并转换为 NumPy 数组
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.squeeze().numpy()

    # 使用特征提取器处理波形数据
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    # 前向传播
    outputs = model(**inputs)
    logits = outputs.logits

    if model_name == "ast-finetuned-audioset-10-10-0.4593":
        expected_slice = torch.tensor([-0.8760, -7.0042, -8.6602])
    elif model_name == "ast-finetuned-audioset-10-10-0.450":
        expected_slice = torch.tensor([-1.1986, -7.0903, -8.2718])
    # 根据模型名称设置预期的输出向量片段
    elif model_name == "ast-finetuned-audioset-10-10-0.448":
        expected_slice = torch.tensor([-2.6128, -8.0080, -9.4344])
    elif model_name == "ast-finetuned-audioset-10-10-0.448-v2":
        expected_slice = torch.tensor([-1.5080, -7.4534, -8.8917])
    elif model_name == "ast-finetuned-audioset-12-12-0.447":
        expected_slice = torch.tensor([-0.5050, -6.5833, -8.0843])
    elif model_name == "ast-finetuned-audioset-14-14-0.443":
        expected_slice = torch.tensor([-0.3826, -7.0336, -8.2413])
    elif model_name == "ast-finetuned-audioset-16-16-0.442":
        expected_slice = torch.tensor([-1.2113, -6.9101, -8.3470])
    elif model_name == "ast-finetuned-speech-commands-v2":
        expected_slice = torch.tensor([6.1589, -8.0566, -8.7984])
    else:
        # 如果模型名称未知，则引发值错误异常
        raise ValueError("Unknown model name")
    
    # 检查模型输出的前三个元素是否与预期的向量片段非常接近，如果不是，则引发值错误异常
    if not torch.allclose(logits[0, :3], expected_slice, atol=1e-4):
        raise ValueError("Logits don't match")
    
    # 打印提示信息，表示检查通过
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径，则执行以下操作
    if pytorch_dump_folder_path is not None:
        # 确保指定路径存在，如果不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印模型保存的信息，包括模型名称和保存路径
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印特征提取器保存的信息，包括保存路径
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # 将特征提取器保存到指定路径
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 打印推送模型和特征提取器到 Hub 的提示信息
        print("Pushing model and feature extractor to the hub...")
        # 将模型推送到指定 Hub 路径
        model.push_to_hub(f"MIT/{model_name}")
        # 将特征提取器推送到指定 Hub 路径
        feature_extractor.push_to_hub(f"MIT/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default="ast-finetuned-audioset-10-10-0.4593",
        type=str,
        help="Name of the Audio Spectrogram Transformer model you'd like to convert."
    )
    # 添加参数：模型名称，指定默认值和帮助信息

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数：PyTorch 模型输出目录的路径，支持默认值和帮助信息

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加参数：是否将转换后的模型推送到 🤗 hub，采用布尔型标志

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_audio_spectrogram_transformer_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_audio_spectrogram_transformer_checkpoint，传递解析得到的参数
```