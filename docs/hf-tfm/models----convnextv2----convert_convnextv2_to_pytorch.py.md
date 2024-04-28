# `.\models\convnextv2\convert_convnextv2_to_pytorch.py`

```
# coding=utf-8

# 引入所需的库和模块
import argparse     # 用于命令行参数解析
import json         # 用于处理 JSON 数据
import os           # 用于操作文件和目录

import requests     # 用于发送 HTTP 请求
import torch        # 用于深度学习框架
from huggingface_hub import hf_hub_download   # 用于从 Hugging Face Hub 下载模型
from PIL import Image   # 用于图像处理

from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification   # 从 transformers 库中引入所需模块和类
from transformers.image_utils import PILImageResampling   # 用于图像处理的相关函数
from transformers.utils import logging   # 用于日志记录

# 设置日志级别为 "info"
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 根据模型的 checkpoint URL 获取 ConvNeXTV2 配置对象
def get_convnextv2_config(checkpoint_url):
    # 创建指定模型的配置对象
    config = ConvNextV2Config()

    # 根据模型的 checkpoint URL 加载不同的深度和隐藏层大小
    if "atto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [40, 80, 160, 320]
    if "femto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [48, 96, 192, 384]
    if "pico" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [64, 128, 256, 512]
    if "nano" in checkpoint_url:
        depths = [2, 2, 8, 2]
        hidden_sizes = [80, 160, 320, 640]
    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if "huge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [352, 704, 1408, 2816]

    # 设置分类标签数量为 1000
    num_labels = 1000
    # 图像标签对应的文件名
    filename = "imagenet-1k-id2label.json"
    # 预期的输出形状
    expected_shape = (1, 1000)

    # 设置仓库的 ID，用于从 Hub 下载标签文件
    repo_id = "huggingface/label-files"
    # 设置分类标签数量
    config.num_labels = num_labels
    # 从 Hub 下载图像标签对应的文件，并加载为字典形式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 转换字典中键的类型为整数
    id2label = {int(k): v for k, v in id2label.items()}

    # 配置对象中设置图像标签对应的字典
    config.id2label = id2label
    # 配置对象中设置标签对应的图像标签
    config.label2id = {v: k for k, v in id2label.items()}
    # 配置���象中设置不同隐藏层的大小和深度
    config.hidden_sizes = hidden_sizes
    config.depths = depths

    # 返回配置对象和预期输出形状
    return config, expected_shape


# 重命名键名，用于模型的加载
def rename_key(name):
    # 将键名中的特定字符串替换为新的字符串
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")
    if "downsample_layers.0.1" in name:
        name = name.replace("downsample_layers.0.1", "embeddings.norm")   # 之后将重命名为 layernorm
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")

# 注意：此处缺少最后一行代码的注释
    # 检查名字中是否包含"downsample_layers.1.1"，如果是则替换为"stages.1.downsampling_layer.1"
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")
    # 检查名字中是否包含"downsample_layers.2.0"，如果是则替换为"stages.2.downsampling_layer.0"
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")
    # 检查名字中是否包含"downsample_layers.2.1"，如果是则替换为"stages.2.downsampling_layer.1"
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")
    # 检查名字中是否包含"downsample_layers.3.0"，如果是则替换为"stages.3.downsampling_layer.0"
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")
    # 检查名字中是否包含"downsample_layers.3.1"，如果是则替换为"stages.3.downsampling_layer.1"
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")
    # 如果名字中包含"stages"且不包含"downsampling_layer"，例如"stages.0.0."，则重命名为"stages.0.layers.0."
    if "stages" in name and "downsampling_layer" not in name:
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    # 将名字中的"gamma"替换为"weight"
    if "gamma" in name:
        name = name.replace("gamma", "weight")
    # 将名字中的"beta"替换为"bias"
    if "beta" in name:
        name = name.replace("beta", "bias")
    # 将名字中的"stages"替换为"encoder.stages"
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    # 将名字中的"norm"替换为"layernorm"
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    # 将名字中的"head"替换为"classifier"
    if "head" in name:
        name = name.replace("head", "classifier")

    return name
# 准备图像的函数
def prepare_img():
    # 图像链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 请求图片并以流的形式打开
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 将检查点 URL 转换为预处理器
def convert_preprocessor(checkpoint_url):
    if "224" in checkpoint_url:
        size = 224
        crop_pct = 224 / 256
    elif "384" in checkpoint_url:
        size = 384
        crop_pct = None
    else:
        size = 512
        crop_pct = None

    return ConvNextImageProcessor(
        size=size,
        crop_pct=crop_pct,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        resample=PILImageResampling.BICUBIC,
    )


# 转换 ConvNeXTV2 的检查点
@torch.no_grad()
def convert_convnextv2_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ConvNeXTV2 structure.
    """
    # 提示下载原始模型
    print("Downloading original model from checkpoint...")
    # 根据 URL 定义 ConvNeXTV2 配置
    config, expected_shape = get_convnextv2_config(checkpoint_url)
    # 从 URL 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    # 提示正在转换模型参数
    print("Converting model parameters...")
    # 重命名键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 对所有键添加前缀，除了分类器头部
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnextv2." + key
        state_dict[key] = val

    # 加载 HuggingFace 模型
    model = ConvNextV2ForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 通过 ConvNextImageProcessor 准备图像并检查输出
    preprocessor = convert_preprocessor(checkpoint_url)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")
    logits = model(**inputs).logits

    # 提示下面的 logits 是在无居中裁剪的情况下获得的
    # 根据不同的检查点 URL，定义预期的 logits
    if checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.3930, 0.1747, -0.5246, 0.4177, 0.4295])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1727, -0.5341, -0.7818, -0.4745, -0.6566])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.0333, 0.1563, -0.9137, 0.1054, 0.0381])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1744, -0.1555, -0.0713, 0.0950, -0.1431])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt":
        # 如果检查点 URL 是指定的值，则设置期望的logits值
        expected_logits = torch.tensor([0.9996, 0.1966, -0.4386, -0.3472, 0.6661])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt":
        # 如果检查点 URL 是指定的值，则设置期望的logits值
        expected_logits = torch.tensor([-0.2553, -0.6708, -0.1359, 0.2518, -0.2488])
    # ... (以下几个elif部分的代码逻辑相似，不再赘述)
    else:
        # 如果检查点 URL 不是任何已知的值，则抛出值错误
        raise ValueError(f"Unknown URL: {checkpoint_url}")

    # 断言实际logits的前5个值与期望的logits值在给定的容差范围内近似相等
    assert torch.allclose(logits[0, :5], expected_logits, atol=1e-3)
    # 检查 logits 的形状是否和预期形状相匹配，如果不匹配则抛出异常
    assert logits.shape == expected_shape
    # 输出模型输出是否与原始结果匹配的信息

    if save_model:
        # 如果需要保存模型
        print("Saving model to local...")
        # 创建用于保存模型的文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # 保存模型和预处理器到指定的路径
        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    # 设置模型名称为 "convnextv2"
    model_name = "convnextv2"
    # 若 checkpoint_url 中包含特定字符串，则在模型名称末尾加上相应的标记
    if "atto" in checkpoint_url:
        model_name += "-atto"
    if "femto" in checkpoint_url:
        model_name += "-femto"
    if "pico" in checkpoint_url:
        model_name += "-pico"
    if "nano" in checkpoint_url:
        model_name += "-nano"
    elif "tiny" in checkpoint_url:
        model_name += "-tiny"
    elif "base" in checkpoint_url:
        model_name += "-base"
    elif "large" in checkpoint_url:
        model_name += "-large"
    elif "huge" in checkpoint_url:
        model_name += "-huge"
    # 根据 checkpoint_url 中包含的信息，进一步修改模型名称
    if "22k" in checkpoint_url and "1k" not in checkpoint_url:
        model_name += "-22k"
    elif "22k" in checkpoint_url and "1k" in checkpoint_url:
        model_name += "-22k-1k"
    elif "1k" in checkpoint_url:
        model_name += "-1k"
    if "224" in checkpoint_url:
        model_name += "-224"
    elif "384" in checkpoint_url:
        model_name += "-384"
    elif "512" in checkpoint_url:
        model_name += "-512"

    if push_to_hub:
        # 如果需要将模型推送到 Hub
        print(f"Pushing {model_name} to the hub...")
        # 将模型和预处理器推送到 Hub 上指定的模型名称下
        model.push_to_hub(model_name)
        preprocessor.push_to_hub(model_name)
# 如果此代码是作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
        type=str,
        help="URL of the original ConvNeXTV2 checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 ConvNeXTV2 模型 checkpoint 转换为 PyTorch 模型
    convert_convnextv2_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
```