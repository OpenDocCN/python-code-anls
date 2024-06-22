# `.\models\dinov2\convert_dinov2_to_hf.py`

```
# 设置文件编码格式为 utf-8
# 版权声明
# 许可证信息
"""从原始仓库转换 DINOv2 检查点。

URL: https://github.com/facebookresearch/dinov2/tree/main
"""

# 导入需要的库
import argparse
import json
from pathlib import Path
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging

# 设置日志的详细程度为 info
logging.set_verbosity_info()
# 获取 logger 实例
logger = logging.get_logger(__name__)

# 根据模型名称和是否是图片分类器，获取 DINOv2 的配置信息
def get_dinov2_config(model_name, image_classifier=False):
    config = Dinov2Config(image_size=518, patch_size=14)

    # 根据模型名称设置不同的隐藏层大小和注意力头
    if "vits" in model_name:
        config.hidden_size = 384
        config.num_attention_heads = 6
    elif "vitb" in model_name:
        pass
    elif "vitl" in model_name:
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "vitg" in model_name:
        config.use_swiglu_ffn = True
        config.hidden_size = 1536
        config.num_hidden_layers = 40
        config.num_attention_heads = 24
    else:
        raise ValueError("Model not supported")

    # 如果是图片分类器，设置 repo_id 和 filename，并加载配置信息
    if image_classifier:
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        config.num_labels = 1000
        config.id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        config.id2label = {int(k): v for k, v in config.id2label.items()}

    return config

# 创建重命名键列表
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))
    # 遍历隐藏层的数量，进行重命名操作
    for i in range(config.num_hidden_layers):
        # layernorms
        # 重命名 layernorms 的权重和偏置参数对应的键值
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        # MLP
        # 根据配置选择使用哪种 MLP 结构，并根据不同选择对应重命名权重和偏置参数的键值
        if config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        # 重命名 layerscale 的参数对应的键值
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        # 重命名 attention projection layer 的参数对应的键值
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    # 重命名最后一层 layernorm 参数对应的键值
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    # 返回重命名后的键值列表
    return rename_keys
# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键关联
    dct[new] = val


# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict, config):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在timm中，这是一个单独的矩阵加上偏置）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 接下来，按顺序添加查询、键和值到状态字典
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 我们将在一张可爱的猫图片上验证我们的结果
def prepare_img():
    # 图片链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 打开图片
    image = Image.open(requests.get(url, stream=True).raw)
    # 返回图片对象
    return image


# 使用torch.no_grad()修饰的函数，表示在该函数中不需要计算梯度
@torch.no_grad()
def convert_dinov2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    复制/粘贴/调整模型的权重到我们的DINOv2结构。
    """

    # 定义默认的Dinov2配置
    image_classifier = "1layer" in model_name
    config = get_dinov2_config(model_name, image_classifier=image_classifier)

    # 从torch hub加载原始模型
    original_model = torch.hub.load("facebookresearch/dinov2", model_name.replace("_1layer", ""))
    original_model.eval()

    # 加载原始模型的state_dict，移除和重命名一些键
    state_dict = original_model.state_dict()
    # 创建重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        # 重命名键
        rename_key(state_dict, src, dest)
    # 读取查询、键和值
    read_in_q_k_v(state_dict, config)

    # 复制state_dict并处理一些键
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        # 更新键
        state_dict[key] = val

    # 加载HuggingFace模型
    # 如果有图像分类器
    if image_classifier:
        # 加载Dinov2ForImageClassification模型并设置为评估模式
        model = Dinov2ForImageClassification(config).eval()
        # 载入模型的状态字典
        model.dinov2.load_state_dict(state_dict)
        
        # 设置模型名称到分类器字典URL的映射关系
        model_name_to_classifier_dict_url = {
            "dinov2_vits14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth",
            "dinov2_vitb14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth",
            "dinov2_vitl14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth",
            "dinov2_vitg14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth",
        }
        
        # 获取模型名称对应的分类器字典URL
        url = model_name_to_classifier_dict_url[model_name]
        # 从URL加载分类器状态字典
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        # 设置模型的分类器权重和偏置
        model.classifier.weight = nn.Parameter(classifier_state_dict["weight"])
        model.classifier.bias = nn.Parameter(classifier_state_dict["bias"])
    else:
        # 加载Dinov2Model模型并设置为评估模式
        model = Dinov2Model(config).eval()
        # 载入模型的状态字典
        model.load_state_dict(state_dict)

    # 加载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 图像预处理
    transformations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # 调整大小为 256x256
            transforms.CenterCrop(224),  # 中心裁剪为 224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(  # 标准化
                mean=IMAGENET_DEFAULT_MEAN,  # 图像网默认均值
                std=IMAGENET_DEFAULT_STD,  # 图像网默认标准差
            ),
        ]
    )

    # 进行预处理后的像素值
    original_pixel_values = transformations(image).unsqueeze(0)  # 插入批处理维度

    # 使用BitImageProcessor预处理图像像素值
    processor = BitImageProcessor(
        size={"shortest_edge": 256},  # 最短边为 256
        resample=PILImageResampling.BICUBIC,  # 重采样算法为 BICUBIC
        image_mean=IMAGENET_DEFAULT_MEAN,  # 图像网默认均值
        image_std=IMAGENET_DEFAULT_STD,  # 图像网默认标准差
    )
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 断言原始像素值与处理后的像素值相等
    assert torch.allclose(original_pixel_values, pixel_values)

    # 禁止梯度计算
    with torch.no_grad():
        # 获取模型输出，并返回隐藏状态
        outputs = model(pixel_values, output_hidden_states=True)
        # 获取原始模型的输出
        original_outputs = original_model(pixel_values)

    # 断言数值
    if image_classifier:
        # 如果有图像分类器，则打印预测的类别
        print("Predicted class:")
        class_idx = outputs.logits.argmax(-1).item()
        print(model.config.id2label[class_idx])
    else:
        # 否则断言输出的最后隐藏状态和原始输出形状相同，并且值相似
        assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape
        assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=1e-3)
    # 打印结果
    print("Looks ok!")
    # 如果提供了 PyTorch 模型导出目录路径，执行以下操作
    if pytorch_dump_folder_path is not None:
        # 确保导出目录存在，如果不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印模型保存的信息
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存至指定的 PyTorch 目录
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印图像处理器保存的信息
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 将图像处理器保存至同一个指定的 PyTorch 目录
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果设置为推送到 hub（Hugging Face Model Hub）
    if push_to_hub:
        # 为不同的模型名称映射到 Hugging Face Model Hub 上的具体模型名称
        model_name_to_hf_name = {
            "dinov2_vits14": "dinov2-small",
            "dinov2_vitb14": "dinov2-base",
            "dinov2_vitl14": "dinov2-large",
            "dinov2_vitg14": "dinov2-giant",
            "dinov2_vits14_1layer": "dinov2-small-imagenet1k-1-layer",
            "dinov2_vitb14_1layer": "dinov2-base-imagenet1k-1-layer",
            "dinov2_vitl14_1layer": "dinov2-large-imagenet1k-1-layer",
            "dinov2_vitg14_1layer": "dinov2-giant-imagenet1k-1-layer",
        }

        # 根据提供的模型名称，获取映射后的 Hub 模型名称
        name = model_name_to_hf_name[model_name]
        # 将模型推送到 Hugging Face Model Hub，包括组织名和模型名
        model.push_to_hub(f"facebook/{name}")
        # 将图像处理器推送到 Hugging Face Model Hub，包括组织名和模型名
        processor.push_to_hub(f"facebook/{name}")
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必要参数
    parser.add_argument(
        "--model_name",
        default="dinov2_vitb14",
        type=str,
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
            "dinov2_vits14_1layer",
            "dinov2_vitb14_1layer",
            "dinov2_vitl14_1layer",
            "dinov2_vitg14_1layer",
        ],
        help="Name of the model you'd like to convert."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析传入的参数
    args = parser.parse_args()
    # 调用函数，传入解析后的参数
    convert_dinov2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```