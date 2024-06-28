# `.\models\dinov2\convert_dinov2_to_hf.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可信息，指定此代码的使用条件
# 此脚本用于从原始存储库转换 DINOv2 检查点

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据处理
from pathlib import Path  # 用于处理文件路径

import requests  # 用于发起 HTTP 请求
import torch  # PyTorch 深度学习库
import torch.nn as nn  # PyTorch 神经网络模块
from huggingface_hub import hf_hub_download  # 用于从 Hugging Face Hub 下载文件
from PIL import Image  # Python 图像库，用于图像处理
from torchvision import transforms  # PyTorch 的视觉处理工具集

# 导入 Transformers 库中相关的类和函数
from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging  # Transformers 库的日志工具

# 设置日志输出级别为 info
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def get_dinov2_config(model_name, image_classifier=False):
    # 创建一个 Dinov2Config 对象，指定图像大小和补丁大小
    config = Dinov2Config(image_size=518, patch_size=14)

    # 根据模型名调整配置参数
    if "vits" in model_name:
        config.hidden_size = 384
        config.num_attention_heads = 6
    elif "vitb" in model_name:
        pass  # 如果模型名包含 'vitb'，则保持默认设置
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
        raise ValueError("Model not supported")  # 抛出异常，指示不支持的模型

    # 如果需要为图像分类器设置配置参数
    if image_classifier:
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        # 从 Hugging Face Hub 下载并加载标签文件，将标签映射添加到配置中
        config.num_labels = 1000
        config.id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        config.id2label = {int(k): v for k, v in config.id2label.items()}

    return config


def create_rename_keys(config):
    rename_keys = []

    # 下面的注释是为了指定列表的格式
    # fmt: off

    # 将原始键名和目标键名添加到重命名键列表中，用于模型权重加载时的映射
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))
    # 对每个隐藏层进行循环迭代，生成需要重命名的键值对列表
    for i in range(config.num_hidden_layers):
        # layernorms
        # 添加权重和偏置的重命名键值对，映射到编码器层的第i层的第1个归一化层
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        
        # MLP
        # 根据配置决定使用不同的MLP结构进行重命名
        if config.use_swiglu_ffn:
            # 使用 SwiGLU 结构的前馈神经网络，添加相应的重命名键值对
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            # 使用普通的全连接层结构，添加相应的重命名键值对
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        
        # layerscale
        # 添加层尺度的重命名键值对，映射到编码器层的第i层的层尺度参数
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        
        # attention projection layer
        # 添加注意力投影层的重命名键值对，映射到编码器层的第i层的注意力输出层
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    # 添加最终的归一化层权重和偏置的重命名键值对，映射到最终的归一化层
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    # 返回所有的重命名键值对列表
    return rename_keys
# 从字典 dct 中移除键 old，并将其对应的值赋给变量 val，然后将键 new 添加到字典 dct 中，其值为 val
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 对每个编码器层的状态字典 state_dict 执行操作，将每一层的查询（query）、键（key）和值（value）分别读取并添加到 state_dict 中
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # 读取输入投影层（在 timm 中，这是一个单独的矩阵加偏置项）的权重和偏置
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        
        # 将查询（query）、键（key）、值（value）依次添加到状态字典 state_dict 中
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

# 准备图像，从指定 URL 获取图像并返回
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image

# 无需梯度的上下文管理器装饰器，用于 DINOv2 模型的权重转换操作
@torch.no_grad()
def convert_dinov2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DINOv2 structure.
    """

    # 根据模型名称和是否为 1layer 模型获取 DINOv2 的配置信息
    image_classifier = "1layer" in model_name
    config = get_dinov2_config(model_name, image_classifier=image_classifier)

    # 从 Torch Hub 加载原始模型
    original_model = torch.hub.load("facebookresearch/dinov2", model_name.replace("_1layer", ""))
    original_model.eval()

    # 加载原始模型的状态字典，移除和重命名一些键
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # 复制状态字典的键值对，并根据需要修改键名
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        state_dict[key] = val

    # 加载 HuggingFace 模型
    # 如果存在图像分类器，则使用Dinov2ForImageClassification模型，加载状态字典，并设为评估模式
    if image_classifier:
        model = Dinov2ForImageClassification(config).eval()
        model.dinov2.load_state_dict(state_dict)
        
        # 根据模型名称选择对应的分类器状态字典的 URL
        model_name_to_classifier_dict_url = {
            "dinov2_vits14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth",
            "dinov2_vitb14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth",
            "dinov2_vitl14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth",
            "dinov2_vitg14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth",
        }
        url = model_name_to_classifier_dict_url[model_name]
        
        # 使用 torch.hub 从 URL 加载分类器状态字典到本地，并在 CPU 上加载
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        
        # 将加载的分类器权重和偏置设为模型的参数
        model.classifier.weight = nn.Parameter(classifier_state_dict["weight"])
        model.classifier.bias = nn.Parameter(classifier_state_dict["bias"])
    else:
        # 否则使用Dinov2Model，并加载状态字典
        model = Dinov2Model(config).eval()
        model.load_state_dict(state_dict)

    # 加载图像数据
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 图像预处理步骤
    transformations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # 调整图像大小
            transforms.CenterCrop(224),  # 中心裁剪图像
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(  # 标准化图像数据
                mean=IMAGENET_DEFAULT_MEAN,  # 图像数据的均值
                std=IMAGENET_DEFAULT_STD,  # 图像数据的标准差
            ),
        ]
    )

    # 对原始像素值应用预处理，并增加批处理维度
    original_pixel_values = transformations(image).unsqueeze(0)

    # 使用 BitImageProcessor 处理图像，返回处理后的像素值
    processor = BitImageProcessor(
        size={"shortest_edge": 256},  # 最短边设置为256像素
        resample=PILImageResampling.BICUBIC,  # 使用双三次插值重采样
        image_mean=IMAGENET_DEFAULT_MEAN,  # 图像数据的均值
        image_std=IMAGENET_DEFAULT_STD,  # 图像数据的标准差
    )
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 断言处理后的像素值与原始像素值在数值上的近似性
    assert torch.allclose(original_pixel_values, pixel_values)

    # 关闭梯度计算，在推理阶段不更新模型参数
    with torch.no_grad():
        # 获取模型输出及隐藏状态
        outputs = model(pixel_values, output_hidden_states=True)
        original_outputs = original_model(pixel_values)

    # 断言检查
    if image_classifier:
        # 如果是图像分类任务，输出预测类别
        print("Predicted class:")
        class_idx = outputs.logits.argmax(-1).item()
        print(model.config.id2label[class_idx])
    else:
        # 否则，断言原始输出和当前输出的最后隐藏状态的一致性
        assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape
        assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=1e-3)
    print("Looks ok!")
    # 如果指定了 pytorch_dump_folder_path，则执行以下操作
    if pytorch_dump_folder_path is not None:
        # 创建目录，如果目录已存在则不报错
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印信息，显示正在保存模型到指定路径
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印信息，显示正在保存图像处理器到指定路径
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 将图像处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 根据模型名称映射到对应的 Hub 名称
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
        
        # 根据模型名称获取对应的 Hub 名称
        name = model_name_to_hf_name[model_name]
        # 将模型推送到 Hub，使用格式化的 Hub 路径
        model.push_to_hub(f"facebook/{name}")
        # 将图像处理器推送到 Hub，使用格式化的 Hub 路径
        processor.push_to_hub(f"facebook/{name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
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
        help="Name of the model you'd like to convert.",
    )
    # 添加一个参数选项，指定模型的名称，必须从预定义的选项中选择

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数选项，指定输出的PyTorch模型目录的路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个参数选项，指定是否将转换后的模型推送到Hugging Face hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_dinov2_checkpoint，传入解析后的参数
    convert_dinov2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```