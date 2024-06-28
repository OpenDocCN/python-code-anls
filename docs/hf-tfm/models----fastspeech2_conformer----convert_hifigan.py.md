# `.\models\fastspeech2_conformer\convert_hifigan.py`

```py
# coding=utf-8
# 版权 2023 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证版本 2.0 授权使用此文件；除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“原样”分发，无任何明示或暗示的担保或条件。
# 请查阅许可证了解详细的许可条件及限制。
"""将 FastSpeech2Conformer HiFi-GAN 的检查点转换为模型。"""

import argparse  # 导入命令行参数解析模块
from pathlib import Path  # 导入处理路径的模块

import torch  # 导入 PyTorch 深度学习框架
import yaml  # 导入处理 YAML 格式的模块

from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, logging  # 导入模型相关类和日志记录

logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")  # 获取模型日志记录器


def load_weights(checkpoint, hf_model, config):
    """加载权重到模型中。

    Args:
        checkpoint (dict): 检查点中的权重字典
        hf_model (FastSpeech2ConformerHifiGan): 需要加载权重的模型实例
        config (FastSpeech2ConformerHifiGanConfig): 模型的配置信息
    """
    vocoder_key_prefix = "tts.generator.vocoder."
    checkpoint = {k.replace(vocoder_key_prefix, ""): v for k, v in checkpoint.items() if vocoder_key_prefix in k}

    hf_model.apply_weight_norm()

    hf_model.conv_pre.weight_g.data = checkpoint["input_conv.weight_g"]
    hf_model.conv_pre.weight_v.data = checkpoint["input_conv.weight_v"]
    hf_model.conv_pre.bias.data = checkpoint["input_conv.bias"]

    for i in range(len(config.upsample_rates)):
        hf_model.upsampler[i].weight_g.data = checkpoint[f"upsamples.{i}.1.weight_g"]
        hf_model.upsampler[i].weight_v.data = checkpoint[f"upsamples.{i}.1.weight_v"]
        hf_model.upsampler[i].bias.data = checkpoint[f"upsamples.{i}.1.bias"]

    for i in range(len(config.upsample_rates) * len(config.resblock_kernel_sizes)):
        for j in range(len(config.resblock_dilation_sizes)):
            hf_model.resblocks[i].convs1[j].weight_g.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_g"]
            hf_model.resblocks[i].convs1[j].weight_v.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_v"]
            hf_model.resblocks[i].convs1[j].bias.data = checkpoint[f"blocks.{i}.convs1.{j}.1.bias"]

            hf_model.resblocks[i].convs2[j].weight_g.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_g"]
            hf_model.resblocks[i].convs2[j].weight_v.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_v"]
            hf_model.resblocks[i].convs2[j].bias.data = checkpoint[f"blocks.{i}.convs2.{j}.1.bias"]

    hf_model.conv_post.weight_g.data = checkpoint["output_conv.1.weight_g"]
    hf_model.conv_post.weight_v.data = checkpoint["output_conv.1.weight_v"]
    hf_model.conv_post.bias.data = checkpoint["output_conv.1.bias"]

    hf_model.remove_weight_norm()


def remap_hifigan_yaml_config(yaml_config_path):
    """重新映射 HiFi-GAN 的 YAML 配置。

    Args:
        yaml_config_path (str): YAML 配置文件的路径
    """
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        args = argparse.Namespace(**args)

    vocoder_type = args.tts_conf["vocoder_type"]
    # 检查声码器类型是否为 "hifigan_generator"，如果不是则引发类型错误并提供详细信息
    if vocoder_type != "hifigan_generator":
        raise TypeError(f"Vocoder config must be for `hifigan_generator`, but got {vocoder_type}")

    # 创建一个空的重映射字典
    remapped_dict = {}

    # 获取声码器参数字典
    vocoder_params = args.tts_conf["vocoder_params"]

    # 定义键映射关系字典，将 espnet 配置键映射到 huggingface 配置键
    key_mappings = {
        "channels": "upsample_initial_channel",
        "in_channels": "model_in_dim",
        "resblock_dilations": "resblock_dilation_sizes",
        "resblock_kernel_sizes": "resblock_kernel_sizes",
        "upsample_kernel_sizes": "upsample_kernel_sizes",
        "upsample_scales": "upsample_rates",
    }

    # 遍历键映射字典，将 espnet 配置中对应键的值映射到 remapped_dict 中的对应 huggingface 键
    for espnet_config_key, hf_config_key in key_mappings.items():
        remapped_dict[hf_config_key] = vocoder_params[espnet_config_key]

    # 将采样率从参数中的 TTS 配置复制到 remapped_dict
    remapped_dict["sampling_rate"] = args.tts_conf["sampling_rate"]
    
    # 设置 normalize_before 为 False
    remapped_dict["normalize_before"] = False
    
    # 从声码器参数中的非线性激活参数中获取 leaky ReLU 的负斜率，并设置到 remapped_dict
    remapped_dict["leaky_relu_slope"] = vocoder_params["nonlinear_activation_params"]["negative_slope"]

    # 返回重映射后的配置字典
    return remapped_dict
# 使用装饰器 @torch.no_grad() 来确保在此函数中不会计算梯度
@torch.no_grad()
# 定义函数 convert_hifigan_checkpoint，用于转换 HiFi-GAN 模型的检查点
def convert_hifigan_checkpoint(
    checkpoint_path,  # 输入参数：原始检查点的文件路径
    pytorch_dump_folder_path,  # 输入参数：输出 PyTorch 模型的文件夹路径
    yaml_config_path=None,  # 输入参数：可选的模型配置文件（YAML）路径，默认为 None
    repo_id=None,  # 输入参数：可选的 🤗 hub 上模型的 repo_id，默认为 None
):
    # 如果提供了 yaml_config_path，则使用 remap_hifigan_yaml_config 函数处理配置文件并创建配置对象
    if yaml_config_path is not None:
        config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
        config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)
    else:
        # 否则，使用默认配置创建配置对象
        config = FastSpeech2ConformerHifiGanConfig()

    # 使用配置对象创建 FastSpeech2ConformerHifiGan 模型
    model = FastSpeech2ConformerHifiGan(config)

    # 加载原始检查点文件内容到 orig_checkpoint
    orig_checkpoint = torch.load(checkpoint_path)
    # 调用 load_weights 函数，将 orig_checkpoint 中的权重加载到模型中
    load_weights(orig_checkpoint, model, config)

    # 将模型保存为 PyTorch 模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了 repo_id，则打印消息并将模型推送到 🤗 hub 上的指定 repo_id
    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


# 如果当前脚本作为主程序运行，则执行以下内容
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：原始检查点文件路径，必需参数
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数：模型配置文件（YAML）路径，可选参数
    parser.add_argument("--yaml_config_path", default=None, type=str, help="Path to config.yaml of model to convert")
    # 添加命令行参数：输出 PyTorch 模型的文件夹路径，必需参数
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数：是否推送到 🤗 hub 的 repo_id，可选参数
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_hifigan_checkpoint 函数，传入命令行解析得到的参数
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.yaml_config_path,
        args.push_to_hub,
    )
```