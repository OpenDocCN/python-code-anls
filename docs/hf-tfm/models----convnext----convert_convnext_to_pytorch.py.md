# `.\models\convnext\convert_convnext_to_pytorch.py`

```
# 设定编码格式为 utf-8
# 版权信息声明
# 根据 Apache 许可证版本 2.0 进行许可
# 当您使用本文件时，除非符合许可证的规定，否则您不得使用本文件
# 您可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依据许可证分发的软件均为"按原样"分发，
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请参阅许可证获取有关权限和限制的具体语言

"""Convert ConvNext checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""

# 引入所需的库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.utils import logging

# 设定日志的详细程度为信息
logging.set_verbosity_info()
# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 根据给定的检查点 URL 获取 ConvNext 的配置
def get_convnext_config(checkpoint_url):
    # 创建 ConvNext 的配置对象
    config = ConvNextConfig()

    # 根据不同的检查点 URL 设置不同的深度和隐藏层大小
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

    # 根据检查点 URL 设置不同的类别数、文件名和期望的形状
    if "1k" in checkpoint_url:
        num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)
    else:
        num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        expected_shape = (1, 21841)

    # 设置标签文件的存储库 ID，并加载标签文件
    repo_id = "huggingface/label-files"
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    # 如果不是 1k 的检查点 URL，则根据指定的网址删除两个类别
    if "1k" not in checkpoint_url:
        del id2label[9205]
        del id2label[15027]
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    return config, expected_shape

# 重命名键名的函数
def rename_key(name):
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")
    if "downsample_layers.0.1" in name:
        name = name.replace("downsample_layers.0.1", "embeddings.norm")  # 稍后我们将更改为 layernorm
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")
    # 检查名字中是否包含特定字符串，如果包含 "stages" 但不包含 "downsampling_layer"，则进行特定的替换
    if "stages" in name and "downsampling_layer" not in name:
        # stages.0.0. for instance should be renamed to stages.0.layers.0.
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "gamma" in name:
        name = name.replace("gamma", "layer_scale_parameter")
    # 检查名字中是否包含特定字符串，然后替换成对应的新字符串
    if "head" in name:
        name = name.replace("head", "classifier")

    # 返回处理后的名字
    return name
# 导入必要的库
import requests
from PIL import Image
import torch
from transformers import ConvNextForImageClassification, ConvNextImageProcessor
from typing import Dict, Any

# 为了验证结果，我们将使用一张可爱的猫咪图片
def prepare_img() -> Image.Image:
    # 图片下载链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过URL打开图片
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_convnext_checkpoint(checkpoint_url: str, pytorch_dump_folder_path: str) -> Dict[str, Any]:
    """
    将模型的权重复制/粘贴/调整为我们的 ConvNext 结构。
    """

    # 根据链接定义相应 ConvNext 配置及期望的维度
    config, expected_shape = get_convnext_config(checkpoint_url)
    # 从链接下载并加载模型的原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    # 重命名模型中的键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 为除了分类器头部以外的所有键添加前缀
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnext." + key
        state_dict[key] = val

    # 加载 HuggingFace 模型
    model = ConvNextForImageClassification(config)
    model.load_state_dict(state_dict)
    # 设置模型为评估模式
    model.eval()

    # 使用 ConvNextImageProcessor 对一张由 ConvNext 图像处理器准备的图像进行输出检测
    size = 224 if "224" in checkpoint_url else 384
    image_processor = ConvNextImageProcessor(size=size)
    pixel_values = image_processor(images=prepare_img(), return_tensors="pt").pixel_values

    # 获取模型输出
    logits = model(pixel_values).logits

    # 注意：下面的模型输出是没有使用中心裁剪的结果
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
        expected_logits = torch.tensor
    # 若checkpoint_url为指定的链接，则设置预期logits
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
    # 若checkpoint_url不是任何指定链接，则引发错误
    else:
        raise ValueError(f"Unknown URL: {checkpoint_url}")

    # 判断logits的前3个值是否在预期值范围内
    assert torch.allclose(logits[0, :3], expected_logits, atol=1e-3)
    # 判断logits的形状是否与预期形状相同
    assert logits.shape == expected_shape

    # 若指定路径不存在，则创建该路径
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印信息，将模型保存到指定路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印信息，将图像处理器保存到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")

    # 打印信息，将模型推送至hub
    print("Pushing model to the hub...")
    # 根据checkpoint_url设置模型名称
    model_name = "convnext"
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
    if "224" in checkpoint_url:
        model_name += "-224"
    elif "384" in checkpoint_url:
        model_name += "-384"
    if "22k" in checkpoint_url and "1k" not in checkpoint_url:
        model_name += "-22k"
    if "22k" in checkpoint_url and "1k" in checkpoint_url:
        model_name += "-22k-1k"

    # 推送模型至hub
    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
        organization="nielsr",
        commit_message="Add model",
    )
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--checkpoint_url",  # 指定模型检查点的 URL
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",  # 默认使用的 ConvNeXT 模型检查点的 URL
        type=str,
        help="URL of the original ConvNeXT checkpoint you'd like to convert.",  # 参数说明：原始 ConvNeXT 检查点的 URL
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 指定 PyTorch 模型输出目录的路径
        default=None,  # 默认为 None
        type=str,
        required=True,  # 必需参数
        help="Path to the output PyTorch model directory.",  # 参数说明：输出 PyTorch 模型的目录路径
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 ConvNeXT 模型检查点转换为 PyTorch 模型
    convert_convnext_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```