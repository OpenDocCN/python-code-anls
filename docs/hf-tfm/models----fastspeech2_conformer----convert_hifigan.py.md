# `.\models\fastspeech2_conformer\convert_hifigan.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
"""Convert FastSpeech2Conformer HiFi-GAN checkpoint."""

# 导入必要的库和模块
import argparse
from pathlib import Path

# 导入 PyTorch 库
import torch
import yaml

# 导入 Hugging Face 的 Transformers 库
from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, logging

# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取记录器
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")

# 定义函数以加载权重
def load_weights(checkpoint, hf_model, config):
    # 定义声码器的键前缀
    vocoder_key_prefix = "tts.generator.vocoder."
    # 从检查点中提取声码器相关键值对，并去除前缀
    checkpoint = {k.replace(vocoder_key_prefix, ""): v for k, v in checkpoint.items() if vocoder_key_prefix in k}

    # 对 HF 模型应用权重归一化
    hf_model.apply_weight_norm()

    # 设置输入卷积层的权重和偏置
    hf_model.conv_pre.weight_g.data = checkpoint["input_conv.weight_g"]
    hf_model.conv_pre.weight_v.data = checkpoint["input_conv.weight_v"]
    hf_model.conv_pre.bias.data = checkpoint["input_conv.bias"]

    # 遍历并设置上采样层的权重和偏置
    for i in range(len(config.upsample_rates)):
        hf_model.upsampler[i].weight_g.data = checkpoint[f"upsamples.{i}.1.weight_g"]
        hf_model.upsampler[i].weight_v.data = checkpoint[f"upsamples.{i}.1.weight_v"]
        hf_model.upsampler[i].bias.data = checkpoint[f"upsamples.{i}.1.bias"]

    # 遍历并设置残差块的权重和偏置
    for i in range(len(config.upsample_rates) * len(config.resblock_kernel_sizes)):
        for j in range(len(config.resblock_dilation_sizes)):
            hf_model.resblocks[i].convs1[j].weight_g.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_g"]
            hf_model.resblocks[i].convs1[j].weight_v.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_v"]
            hf_model.resblocks[i].convs1[j].bias.data = checkpoint[f"blocks.{i}.convs1.{j}.1.bias"]

            hf_model.resblocks[i].convs2[j].weight_g.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_g"]
            hf_model.resblocks[i].convs2[j].weight_v.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_v"]
            hf_model.resblocks[i].convs2[j].bias.data = checkpoint[f"blocks.{i}.convs2.{j}.1.bias"]

    # 设置输出卷积层的权重和偏置
    hf_model.conv_post.weight_g.data = checkpoint["output_conv.1.weight_g"]
    hf_model.conv_post.weight_v.data = checkpoint["output_conv.1.weight_v"]
    hf_model.conv_post.bias.data = checkpoint["output_conv.1.bias"]

    # 移除 HF 模型的权重归一化
    hf_model.remove_weight_norm()

# 重新映射 Hifigan 的 YAML 配置
def remap_hifigan_yaml_config(yaml_config_path):
    # 从 YAML 文件中加载配置参数
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        args = argparse.Namespace(**args)

    # 获取声码器类型
    vocoder_type = args.tts_conf["vocoder_type"]
    # 如果 vocoder_type 不是 "hifigan_generator"，则抛出类型错误异常，指示必须使用 `hifigan_generator` 的声码器配置
    if vocoder_type != "hifigan_generator":
        raise TypeError(f"Vocoder config must be for `hifigan_generator`, but got {vocoder_type}")

    # 创建一个空字典，用于存储重新映射后的参数
    remapped_dict = {}
    # 获取 vocoder_params
    vocoder_params = args.tts_conf["vocoder_params"]

    # 定义字典，指定参数之间的键映射关系
    # espnet_config_key -> hf_config_key
    key_mappings = {
        "channels": "upsample_initial_channel",
        "in_channels": "model_in_dim",
        "resblock_dilations": "resblock_dilation_sizes",
        "resblock_kernel_sizes": "resblock_kernel_sizes",
        "upsample_kernel_sizes": "upsample_kernel_sizes",
        "upsample_scales": "upsample_rates",
    }
    # 遍历键映射关系字典，将参数从 espnet_config_key 映射到 hf_config_key
    for espnet_config_key, hf_config_key in key_mappings.items():
        remapped_dict[hf_config_key] = vocoder_params[espnet_config_key]
    # 将采样率添加到重新映射的字典中
    remapped_dict["sampling_rate"] = args.tts_conf["sampling_rate"]
    # 设置 normalize_before 参数为 False
    remapped_dict["normalize_before"] = False
    # 从 vocoder_params 中获取非线性激活参数中的 negative_slope，并添加到重新映射的字典中
    remapped_dict["leaky_relu_slope"] = vocoder_params["nonlinear_activation_params"]["negative_slope"]

    # 返回重新映射后的字典
    return remapped_dict
# 禁用梯度更新上下文管理器，以确保在推理过程中不会进行梯度更新
@torch.no_grad()
# 将 HiFi-GAN 模型检查点转换为 PyTorch 模型
def convert_hifigan_checkpoint(
    # 模型检查点路径
    checkpoint_path,
    # 转换后的 PyTorch 模型保存路径
    pytorch_dump_folder_path,
    # 可选参数，HiFi-GAN 模型的配置文件路径
    yaml_config_path=None,
    # 可选参数，模型在 🤗 hub 中的存储库 ID
    repo_id=None,
):
    # 如果提供了配置文件路径，则根据配置文件创建 HiFi-GAN 模型配置对象
    if yaml_config_path is not None:
        # 通过配置文件路径获取配置参数
        config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
        # 使用配置参数初始化 FastSpeech2ConformerHifiGanConfig 配置对象
        config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)
    # 如果未提供配置文件路径，则使用默认参数初始化 HiFi-GAN 模型配置对象
    else:
        config = FastSpeech2ConformerHifiGanConfig()
    
    # 根据配置对象初始化 FastSpeech2ConformerHifiGan 模型
    model = FastSpeech2ConformerHifiGan(config)

    # 加载原始检查点
    orig_checkpoint = torch.load(checkpoint_path)
    # 将原始检查点加载到模型中
    load_weights(orig_checkpoint, model, config)

    # 保存转换后的模型为 PyTorch 模型
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了存储库 ID，将模型推送到 🤗 hub
    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：原始检查点路径
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数：配置文件路径
    parser.add_argument("--yaml_config_path", default=None, type=str, help="Path to config.yaml of model to convert")
    # 添加命令行参数：转换后的 PyTorch 模型保存路径
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数：模型推送到 🤗 hub 的存储库 ID
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 HiFi-GAN 模型检查点转换为 PyTorch 模型
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.yaml_config_path,
        args.push_to_hub,
    )
```