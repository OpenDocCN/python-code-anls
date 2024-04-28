# `.\models\fastspeech2_conformer\convert_model_with_hifigan.py`

```
# 指定编码为 UTF-8
# 版权声明为 HuggingFace Inc. 团队所有
# 使用 Apache 许可证 2.0 版本，需要遵守相关条款
# 获取许可证副本的链接
# 除非适用法律要求或书面同意，否则在合规的情况下才能使用该文件
# 根据其特定语言规定的条件分发软件，采用 "按原样提供" 的方式分发，没有任何明示或暗示的保证
# 详见许可证，管理权限和限制条件
"""将 FastSpeech2Conformer 检查点转换为新的格式"""

import argparse

import torch

# 导入所需的模块和类
from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerWithHifiGan,
    FastSpeech2ConformerWithHifiGanConfig,
    logging,
)

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


# 定义函数，用于转换 FastSpeech2ConformerWithHifiGan 检查点
def convert_FastSpeech2ConformerWithHifiGan_checkpoint(
    checkpoint_path,
    yaml_config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    # 准备模型参数
    model_params, *_ = remap_model_yaml_config(yaml_config_path)
    model_config = FastSpeech2ConformerConfig(**model_params)

    # 创建 FastSpeech2ConformerModel 对象
    model = FastSpeech2ConformerModel(model_config)

    # 加载原始 ESPnet 检查点
    espnet_checkpoint = torch.load(checkpoint_path)
    # 将 ESPnet 检查点转换为适用于 HF 的状态字典
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)
    model.load_state_dict(hf_compatible_state_dict)

    # 准备声码器参数
    config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
    vocoder_config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)

    # 创建 FastSpeech2ConformerHifiGan 对象
    vocoder = FastSpeech2ConformerHifiGan(vocoder_config)
    # 加载权重到声码器
    load_weights(espnet_checkpoint, vocoder, vocoder_config)

    # 准备模型 + 声码器
    config = FastSpeech2ConformerWithHifiGanConfig.from_sub_model_configs(model_config, vocoder_config)
    # 创建 FastSpeech2ConformerWithHifiGan 对象
    with_hifigan_model = FastSpeech2ConformerWithHifiGan(config)
    with_hifigan_model.model = model
    with_hifigan_model.vocoder = vocoder

    # 保存转换后的检查点
    with_hifigan_model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        # 将转换后的模型推送到中心仓库
        print("Pushing to the hub...")
        with_hifigan_model.push_to_hub()


# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加命令行参��，指定原始检查点路径
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数，指定需要转换的模型配置文件路径
    parser.add_argument(
        "--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert"
    )
    # 添加命令行参数解析器的一个参数：用于指定 PyTorch 模型的输出路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,  # 参数是必须的
        default=None,   # 默认值为 None
        type=str,       # 参数类型为字符串
        help="Path to the output `FastSpeech2ConformerModel` PyTorch model.",  # 参数的帮助信息
    )
    
    # 添加命令行参数解析器的一个参数：用于指定是否上传转换后的模型到 🤗 hub
    parser.add_argument(
        "--push_to_hub",
        default=None,   # 默认值为 None
        type=str,       # 参数类型为字符串
        help="Where to upload the converted model on the 🤗 hub.",  # 参数的帮助信息
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数将 FastSpeech2Conformer 模型与 HiFi-GAN 模型合并并转换为 PyTorch 模型
    convert_FastSpeech2ConformerWithHifiGan_checkpoint(
        args.checkpoint_path,       # 原始模型的检查点路径
        args.yaml_config_path,      # YAML 配置文件的路径
        args.pytorch_dump_folder_path,  # PyTorch 模型的输出路径
        args.push_to_hub           # 是否上传模型到 🤗 hub 的指示
    )
```