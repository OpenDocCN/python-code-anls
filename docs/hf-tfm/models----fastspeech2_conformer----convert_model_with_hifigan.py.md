# `.\models\fastspeech2_conformer\convert_model_with_hifigan.py`

```
# coding=utf-8
# 设置文件编码为UTF-8，确保支持中文和其他特殊字符的正确处理

# 导入必要的库和模块
import argparse  # 导入用于解析命令行参数的模块 argparse

import torch  # 导入 PyTorch 库

from transformers import (  # 从 transformers 库中导入以下模块和类
    FastSpeech2ConformerConfig,  # FastSpeech2ConformerConfig 类，用于配置 FastSpeech2Conformer 模型
    FastSpeech2ConformerHifiGan,  # FastSpeech2ConformerHifiGan 类，用于 FastSpeech2Conformer 和 HifiGan 的结合
    FastSpeech2ConformerHifiGanConfig,  # FastSpeech2ConformerHifiGanConfig 类，配置 FastSpeech2ConformerHifiGan 模型
    FastSpeech2ConformerModel,  # FastSpeech2ConformerModel 类，FastSpeech2Conformer 模型
    FastSpeech2ConformerWithHifiGan,  # FastSpeech2ConformerWithHifiGan 类，结合 FastSpeech2Conformer 和 HifiGan 的模型
    FastSpeech2ConformerWithHifiGanConfig,  # FastSpeech2ConformerWithHifiGanConfig 类，配置 FastSpeech2ConformerWithHifiGan 模型
    logging,  # logging 模块，用于记录日志
)

from .convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch import (  # 导入本地的模块和函数
    convert_espnet_state_dict_to_hf,  # convert_espnet_state_dict_to_hf 函数，将 espnet 模型的状态字典转换为 HF 兼容格式
    remap_model_yaml_config,  # remap_model_yaml_config 函数，重映射模型的 YAML 配置
)

from .convert_hifigan import load_weights, remap_hifigan_yaml_config  # 导入本地的 load_weights 和 remap_hifigan_yaml_config 函数

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 获取 logger 对象
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


def convert_FastSpeech2ConformerWithHifiGan_checkpoint(
    checkpoint_path,
    yaml_config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    # 准备模型
    model_params, *_ = remap_model_yaml_config(yaml_config_path)
    model_config = FastSpeech2ConformerConfig(**model_params)  # 使用从 YAML 文件中提取的参数配置 FastSpeech2ConformerConfig

    model = FastSpeech2ConformerModel(model_config)  # 基于配置创建 FastSpeech2ConformerModel 对象

    espnet_checkpoint = torch.load(checkpoint_path)  # 加载原始 ESPnet 模型的检查点
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)  # 将 ESPnet 模型的状态字典转换为 HF 兼容格式
    model.load_state_dict(hf_compatible_state_dict)  # 加载 HF 兼容的状态字典到模型中

    # 准备声码器
    config_kwargs = remap_hifigan_yaml_config(yaml_config_path)  # 从 YAML 文件中获取声码器配置参数
    vocoder_config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)  # 使用配置参数创建 FastSpeech2ConformerHifiGanConfig

    vocoder = FastSpeech2ConformerHifiGan(vocoder_config)  # 基于配置创建 FastSpeech2ConformerHifiGan 声码器
    load_weights(espnet_checkpoint, vocoder, vocoder_config)  # 加载权重到声码器中

    # 准备模型 + 声码器组合
    config = FastSpeech2ConformerWithHifiGanConfig.from_sub_model_configs(model_config, vocoder_config)
    with_hifigan_model = FastSpeech2ConformerWithHifiGan(config)  # 基于组合配置创建 FastSpeech2ConformerWithHifiGan 模型
    with_hifigan_model.model = model  # 将 FastSpeech2Conformer 模型赋给组合模型的成员变量
    with_hifigan_model.vocoder = vocoder  # 将声码器赋给组合模型的声码器成员变量

    with_hifigan_model.save_pretrained(pytorch_dump_folder_path)  # 保存组合模型到指定路径

    if repo_id:
        print("Pushing to the hub...")
        with_hifigan_model.push_to_hub(repo_id)  # 将模型推送到模型中心（hub）
    # 配置解析器添加参数
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Script for converting FastSpeech2Conformer with HifiGAN Model"
        )
    
        # 必须参数: 输出的 PyTorch 模型路径
        parser.add_argument(
            "--pytorch_dump_folder_path",
            required=True,
            default=None,
            type=str,
            help="Path to the output `FastSpeech2ConformerModel` PyTorch model.",
        )
        # 选择参数: 将模型上传到 🤗 hub 的选项
        parser.add_argument(
            "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
        )
    
        # 解析参数
        args = parser.parse_args()
    
        # 主函数调用
        convert_FastSpeech2ConformerWithHifiGan_checkpoint(
            args.checkpoint_path,
            args.yaml_config_path,
            args.pytorch_dump_folder_path,
            args.push_to_hub,
        )
```