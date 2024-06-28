# `.\models\speecht5\convert_hifigan.py`

```py
# 设置编码格式为 UTF-8

# 导入必要的库和模块
import argparse  # 用于解析命令行参数

import numpy as np  # 用于数值计算
import torch  # PyTorch 深度学习框架

from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging  # 导入 Transformers 库中的相关模块和类

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 获取名为 "transformers.models.speecht5" 的日志记录器
logger = logging.get_logger("transformers.models.speecht5")


def load_weights(checkpoint, hf_model, config):
    # 对模型应用权重归一化操作
    hf_model.apply_weight_norm()

    # 加载输入卷积层的权重和偏置
    hf_model.conv_pre.weight_g.data = checkpoint["input_conv.weight_g"]
    hf_model.conv_pre.weight_v.data = checkpoint["input_conv.weight_v"]
    hf_model.conv_pre.bias.data = checkpoint["input_conv.bias"]

    # 加载每个上采样层的权重和偏置
    for i in range(len(config.upsample_rates)):
        hf_model.upsampler[i].weight_g.data = checkpoint[f"upsamples.{i}.1.weight_g"]
        hf_model.upsampler[i].weight_v.data = checkpoint[f"upsamples.{i}.1.weight_v"]
        hf_model.upsampler[i].bias.data = checkpoint[f"upsamples.{i}.1.bias"]

    # 加载每个残差块的权重和偏置
    for i in range(len(config.upsample_rates) * len(config.resblock_kernel_sizes)):
        for j in range(len(config.resblock_dilation_sizes)):
            hf_model.resblocks[i].convs1[j].weight_g.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_g"]
            hf_model.resblocks[i].convs1[j].weight_v.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_v"]
            hf_model.resblocks[i].convs1[j].bias.data = checkpoint[f"blocks.{i}.convs1.{j}.1.bias"]

            hf_model.resblocks[i].convs2[j].weight_g.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_g"]
            hf_model.resblocks[i].convs2[j].weight_v.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_v"]
            hf_model.resblocks[i].convs2[j].bias.data = checkpoint[f"blocks.{i}.convs2.{j}.1.bias"]

    # 加载输出卷积层的权重和偏置
    hf_model.conv_post.weight_g.data = checkpoint["output_conv.1.weight_g"]
    hf_model.conv_post.weight_v.data = checkpoint["output_conv.1.weight_v"]
    hf_model.conv_post.bias.data = checkpoint["output_conv.1.bias"]

    # 移除模型的权重归一化
    hf_model.remove_weight_norm()


@torch.no_grad()
def convert_hifigan_checkpoint(
    checkpoint_path,
    stats_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    # 如果提供了配置文件路径，则使用预训练配置创建配置对象，否则使用默认配置
    if config_path is not None:
        config = SpeechT5HifiGanConfig.from_pretrained(config_path)
    else:
        config = SpeechT5HifiGanConfig()

    # 创建 SpeechT5HifiGan 模型对象
    model = SpeechT5HifiGan(config)

    # 加载原始检查点文件
    orig_checkpoint = torch.load(checkpoint_path)

    # 加载权重到模型中
    load_weights(orig_checkpoint["model"]["generator"], model, config)
    # 加载保存的统计信息，这里假设 stats_path 是保存的 numpy 数组的文件路径
    stats = np.load(stats_path)
    
    # 从统计信息中提取平均值，并重塑为一维数组
    mean = stats[0].reshape(-1)
    
    # 从统计信息中提取标度，并重塑为一维数组
    scale = stats[1].reshape(-1)
    
    # 将平均值转换为 PyTorch 的 float 张量，并设置为模型的平均值属性
    model.mean = torch.from_numpy(mean).float()
    
    # 将标度转换为 PyTorch 的 float 张量，并设置为模型的标度属性
    model.scale = torch.from_numpy(scale).float()
    
    # 将模型保存到指定的 PyTorch 转储文件夹路径
    model.save_pretrained(pytorch_dump_folder_path)
    
    # 如果 repo_id 存在，则将模型推送到指定的存储库
    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
# 如果当前脚本被作为主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：原始检查点的路径，必填参数
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    # 添加命令行参数：stats.npy 文件的路径，必填参数
    parser.add_argument("--stats_path", required=True, default=None, type=str, help="Path to stats.npy file")
    # 添加命令行参数：待转换模型的 HF 配置文件（config.json）的路径，可选参数
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数：输出 PyTorch 模型的文件夹路径，必填参数
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数：指定是否将转换后的模型上传到 🤗 hub 的路径，可选参数
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    
    # 调用函数 convert_hifigan_checkpoint 进行模型检查点的转换
    convert_hifigan_checkpoint(
        args.checkpoint_path,     # 原始检查点的路径
        args.stats_path,          # stats.npy 文件的路径
        args.pytorch_dump_folder_path,   # 输出 PyTorch 模型的文件夹路径
        args.config_path,         # HF 配置文件的路径
        args.push_to_hub          # 是否上传到 🤗 hub 的路径
    )
```