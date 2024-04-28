# `.\transformers\models\speecht5\convert_hifigan.py`

```
# 设置代码文件的编码格式为utf-8
# 版权声明，保留所有权利
# 根据 Apache 许可证版本 2.0 进行许可
# 在遵守许可证的情况下，可以使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发
# 不提供任何明示或暗示的担保或条件
# 请参阅许可证以了解特定语言规定的权限和
# 限制

"""将 SpeechT5 HiFi-GAN 检查点转换为模型权重"""

import argparse  # 导入用于解析命令行参数的模块

import numpy as np  # 导入 numpy 模块并将其命名为 np
import torch  # 导入 torch 模块

from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging  # 从 transformers 库中导入 SpeechT5HifiGan、SpeechT5HifiGanConfig 和 logging

logging.set_verbosity_info()  # 设置日志级别为 info
logger = logging.get_logger("transformers.models.speecht5")  # 获取 transformers 模块中 speecht5 模型的日志记录器


def load_weights(checkpoint, hf_model, config):
    hf_model.apply_weight_norm()  # 应用权重规范化

    hf_model.conv_pre.weight_g.data = checkpoint["input_conv.weight_g"]  # 设置模型预卷积权重的数据
    hf_model.conv_pre.weight_v.data = checkpoint["input_conv.weight_v"]  # 设置模型预卷积权重的数据
    hf_model.conv_pre.bias.data = checkpoint["input_conv.bias"]  # 设置模型预卷积偏差的数据

    for i in range(len(config.upsample_rates)):
        hf_model.upsampler[i].weight_g.data = checkpoint[f"upsamples.{i}.1.weight_g"]  # 设置模型上采样权重的数据
        hf_model.upsampler[i].weight_v.data = checkpoint[f"upsamples.{i}.1.weight_v"]  # 设置模型上采样权重的数据
        hf_model.upsampler[i].bias.data = checkpoint[f"upsamples.{i}.1.bias"]  # 设置模型上采样偏差的数据

    for i in range(len(config.upsample_rates) * len(config.resblock_kernel_sizes)):
        for j in range(len(config.resblock_dilation_sizes)):
            hf_model.resblocks[i].convs1[j].weight_g.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_g"]  # 设置模型残差块权重的数据
            hf_model.resblocks[i].convs1[j].weight_v.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_v"]  # 设置模型残差块权重的数据
            hf_model.resblocks[i].convs1[j].bias.data = checkpoint[f"blocks.{i}.convs1.{j}.1.bias"]  # 设置模型残差块偏差的数据

            hf_model.resblocks[i].convs2[j].weight_g.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_g"]  # 设置模型残差块权重的数据
            hf_model.resblocks[i].convs2[j].weight_v.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_v"]  # 设置模型残差块权重的数据
            hf_model.resblocks[i].convs2[j].bias.data = checkpoint[f"blocks.{i}.convs2.{j}.1.bias"]  # 设置模型残差块偏差的数据

    hf_model.conv_post.weight_g.data = checkpoint["output_conv.1.weight_g"]  # 设置模型后卷积权重的数据
    hf_model.conv_post.weight_v.data = checkpoint["output_conv.1.weight_v"]  # 设置模型后卷积权重的数据
    hf_model.conv_post.bias.data = checkpoint["output_conv.1.bias"]  # 设置模型后卷积偏差的数据

    hf_model.remove_weight_norm()  # 移除权重规范化


@torch.no_grad()  # 禁用梯度计算
def convert_hifigan_checkpoint(
    checkpoint_path,
    stats_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    if config_path is not None:
        config = SpeechT5HifiGanConfig.from_pretrained(config_path)  # 如果提供了配置文件路径，则从预训练模型创建配置
    else:
        config = SpeechT5HifiGanConfig()  # 否则，使用默认配置

    model = SpeechT5HifiGan(config)  # 创建 SpeechT5HifiGan 模型对象

    orig_checkpoint = torch.load(checkpoint_path)  # 加载原始检查点
    load_weights(orig_checkpoint["model"]["generator"], model, config)  # 转换权重
    # 从指定路径加载统计数据文件
    stats = np.load(stats_path)
    # 获取统计数据中的平均值，并将其重塑为一维数组
    mean = stats[0].reshape(-1)
    # 获取统计数据中的标准差，并将其重塑为一维数组
    scale = stats[1].reshape(-1)
    # 将平均值转换为 Torch 张量并设置为模型的平均值
    model.mean = torch.from_numpy(mean).float()
    # 将标准差转换为 Torch 张量并设置为模型的标准差
    model.scale = torch.from_numpy(scale).float()

    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果有 repo_id 提供，则将模型推送到平台
    if repo_id:
        print("Pushing to the hub...")
        # 推送模型到平台
        model.push_to_hub(repo_id)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定原始检查点的路径，必需参数
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数，指定stats.npy文件的路径，必需参数
    parser.add_argument("--stats_path", required=True, default=None, type=str, help="Path to stats.npy file")
    # 添加命令行参数，指定模型转换的配置文件config.json的路径，可选参数
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，指定输出PyTorch模型的文件夹路径，必需参数
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数，指定是否将转换后的模型上传到🤗 hub的位置，可选参数
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 HiFi-GAN 检查点转换为 PyTorch 模型
    convert_hifigan_checkpoint(
        args.checkpoint_path,  # 原始检查点路径
        args.stats_path,  # stats.npy 文件路径
        args.pytorch_dump_folder_path,  # 输出 PyTorch 模型文件夹路径
        args.config_path,  # 模型配置文件路径
        args.push_to_hub,  # 是否上传至🤗 hub的位置
    )
```