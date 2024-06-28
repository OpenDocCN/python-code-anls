# `.\models\convnext\convert_convnext_to_pytorch.py`

```py
# 指定文件编码为UTF-8，确保代码中的中文等字符能正确处理
# 版权声明，声明代码版权归HuggingFace Inc.团队所有
#
# 根据Apache许可证2.0版，只有在符合许可证条件下才能使用此文件
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何形式的明示或暗示保证
# 有关许可证的详细信息，请参阅许可证文本
"""从原始存储库中转换ConvNext检查点。

URL: https://github.com/facebookresearch/ConvNeXt
"""


import argparse  # 导入解析命令行参数的模块
import json  # 导入处理JSON格式数据的模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入发送HTTP请求的模块
import torch  # 导入PyTorch深度学习框架
from huggingface_hub import hf_hub_download  # 导入从Hugging Face Hub下载模型和数据的功能
from PIL import Image  # 导入处理图像的模块

from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor  # 导入ConvNext模型及其相关组件
from transformers.utils import logging  # 导入用于记录日志的模块


logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_convnext_config(checkpoint_url):
    config = ConvNextConfig()  # 创建ConvNext模型配置对象

    # 根据checkpoint_url的内容设置模型深度和隐藏层大小
    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "small" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if "xlarge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]

    # 根据checkpoint_url的内容设置标签数、标签映射文件名及期望形状
    if "1k" in checkpoint_url:
        num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)
    else:
        num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        expected_shape = (1, 21841)

    repo_id = "huggingface/label-files"
    # 加载并解析标签映射文件，转换为ID到标签的映射字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    
    if "1k" not in checkpoint_url:
        # 删除包含21843个标签的数据集中模型没有的类别
        # 参考：https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]
    
    # 设置模型配置对象的ID到标签和标签到ID映射
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    return config, expected_shape


def rename_key(name):
    if "downsample_layers.0.0" in name:
        # 将指定模型权重的键名更改为新的键名以匹配新的模型结构
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")
    if "downsample_layers.0.1" in name:
        # 将指定模型权重的键名更改为新的键名以匹配新的模型结构
        name = name.replace("downsample_layers.0.1", "embeddings.norm")  # 后续会重命名为layernorm
    # 检查字符串中是否包含特定子串 "downsample_layers.1.0"，如果是，则替换为 "stages.1.downsampling_layer.0"
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")
    # 检查字符串中是否包含特定子串 "downsample_layers.1.1"，如果是，则替换为 "stages.1.downsampling_layer.1"
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")
    # 检查字符串中是否包含特定子串 "downsample_layers.2.0"，如果是，则替换为 "stages.2.downsampling_layer.0"
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")
    # 检查字符串中是否包含特定子串 "downsample_layers.2.1"，如果是，则替换为 "stages.2.downsampling_layer.1"
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")
    # 检查字符串中是否包含特定子串 "downsample_layers.3.0"，如果是，则替换为 "stages.3.downsampling_layer.0"
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")
    # 检查字符串中是否包含特定子串 "downsample_layers.3.1"，如果是，则替换为 "stages.3.downsampling_layer.1"
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")
    # 如果字符串中包含 "stages" 但不包含 "downsampling_layer"，将其修改为 "stages.layers"
    if "stages" in name and "downsampling_layer" not in name:
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    # 如果字符串中包含 "stages"，将其替换为 "encoder.stages"
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    # 如果字符串中包含 "norm"，将其替换为 "layernorm"
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    # 如果字符串中包含 "gamma"，将其替换为 "layer_scale_parameter"
    if "gamma" in name:
        name = name.replace("gamma", "layer_scale_parameter")
    # 如果字符串中包含 "head"，将其替换为 "classifier"
    if "head" in name:
        name = name.replace("head", "classifier")

    # 返回修改后的字符串 name
    return name
# 定义一个函数，用于准备一张可爱猫咪的图片作为数据处理的基础
def prepare_img():
    # 定义图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 HTTP 请求获取图像的二进制数据流，并使用 PIL 库打开为图像对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
# 定义一个函数，用于将预训练模型的权重转换到我们的 ConvNext 结构中
def convert_convnext_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型权重以适应我们的 ConvNext 结构。
    """

    # 根据 URL 获取 ConvNext 的配置和预期形状
    config, expected_shape = get_convnext_config(checkpoint_url)
    
    # 从 URL 加载原始的 state_dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    
    # 重命名 state_dict 的键值对
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    
    # 为除了分类器头部外的所有键值对添加前缀
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnext." + key
        state_dict[key] = val
    
    # 加载 HuggingFace 的 ConvNext 图像分类模型
    model = ConvNextForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 准备一个 ConvNextImageProcessor 对象，用于处理图像
    size = 224 if "224" in checkpoint_url else 384
    image_processor = ConvNextImageProcessor(size=size)
    # 准备图像数据，返回张量格式的像素值
    pixel_values = image_processor(images=prepare_img(), return_tensors="pt").pixel_values
    
    # 使用模型预测图像的 logits
    logits = model(pixel_values).logits
    
    # 注意：以下的 logits 是在没有中心裁剪的情况下获得的
    # 根据不同的 checkpoint_url，设置期望的 logits 值
    if checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth":
        expected_logits = torch.tensor([-0.1210, -0.6605, 0.1918])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth":
        expected_logits = torch.tensor([-0.4473, -0.1847, -0.6365])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth":
        expected_logits = torch.tensor([0.4525, 0.7539, 0.0308])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth":
        expected_logits = torch.tensor([0.3561, 0.6350, -0.0384])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth":
        expected_logits = torch.tensor([0.4174, -0.0989, 0.1489])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth":
        expected_logits = torch.tensor([0.2513, -0.1349, -0.1613])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth":
        expected_logits = torch.tensor([1.2980, 0.3631, -0.1198])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth":
        expected_logits = torch.tensor([1.2963, 0.1227, 0.1723])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth":
        expected_logits = torch.tensor([1.7956, 0.8390, 0.2820])
    # 检查模型下载链接是否为指定的预训练模型，设置对应的预期输出结果
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth":
        expected_logits = torch.tensor([-0.2822, -0.0502, -0.0878])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth":
        expected_logits = torch.tensor([-0.5672, -0.0730, -0.4348])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth":
        expected_logits = torch.tensor([0.2681, 0.2365, 0.6246])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth":
        expected_logits = torch.tensor([-0.2642, 0.3931, 0.5116])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth":
        expected_logits = torch.tensor([-0.6677, -0.1873, -0.8379])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth":
        expected_logits = torch.tensor([-0.7749, -0.2967, -0.6444])
    else:
        # 如果链接不在预期列表中，抛出值错误异常
        raise ValueError(f"Unknown URL: {checkpoint_url}")

    # 断言模型输出的前三个元素与预期输出接近，允许的误差为1e-3
    assert torch.allclose(logits[0, :3], expected_logits, atol=1e-3)
    # 断言模型输出的形状与预期形状相同
    assert logits.shape == expected_shape

    # 创建目录以保存 PyTorch 模型和图像处理器
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 提示将模型推送到模型中心
    print("Pushing model to the hub...")
    # 初始化模型名称
    model_name = "convnext"
    # 根据链接中的关键字更新模型名称
    if "tiny" in checkpoint_url:
        model_name += "-tiny"
    elif "small" in checkpoint_url:
        model_name += "-small"
    elif "base" in checkpoint_url:
        model_name += "-base"
    elif "xlarge" in checkpoint_url:
        model_name += "-xlarge"
    elif "large" in checkpoint_url:
        model_name += "-large"
    # 根据链接中的分辨率更新模型名称
    if "224" in checkpoint_url:
        model_name += "-224"
    elif "384" in checkpoint_url:
        model_name += "-384"
    # 根据链接中的训练集更新模型名称
    if "22k" in checkpoint_url and "1k" not in checkpoint_url:
        model_name += "-22k"
    if "22k" in checkpoint_url and "1k" in checkpoint_url:
        model_name += "-22k-1k"

    # 将模型推送到指定的模型中心仓库
    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
        organization="nielsr",
        commit_message="Add model",
    )
if __name__ == "__main__":
    # 如果脚本直接运行（而非被导入为模块），执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        help="URL of the original ConvNeXT checkpoint you'd like to convert."
    )
    # 添加一个命令行参数，用于指定 ConvNeXT 模型的原始检查点的 URL

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory."
    )
    # 添加一个命令行参数，用于指定输出 PyTorch 模型的目录路径，并且是必须提供的参数

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 对象中

    convert_convnext_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
    # 调用函数 convert_convnext_checkpoint，传递解析后的参数 args 中的 checkpoint_url 和 pytorch_dump_folder_path
```