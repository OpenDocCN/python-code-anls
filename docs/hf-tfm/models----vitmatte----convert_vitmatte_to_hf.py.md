# `.\models\vitmatte\convert_vitmatte_to_hf.py`

```py
# 加载 argparse 库，用于处理命令行参数
import argparse

# 加载 requests 库，用于发送 HTTP 请求
import requests

# 加载 PyTorch 库，用于深度学习模型操作
import torch

# 从 huggingface_hub 库中导入 hf_hub_download 函数，用于从 HF Hub 下载模型
from huggingface_hub import hf_hub_download

# 从 PIL 库中导入 Image 类，用于图像处理
from PIL import Image

# 从 transformers 库中导入 VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor 类
from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor


def get_config(model_name):
    # 根据模型名称确定隐藏层大小和注意力头数
    hidden_size = 384 if "small" in model_name else 768
    num_attention_heads = 6 if "small" in model_name else 12

    # 创建 VitDetConfig 实例，定义了图像检测器的配置
    backbone_config = VitDetConfig(
        num_channels=4,
        image_size=512,
        pretrain_image_size=224,
        patch_size=16,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=True,
        window_size=14,
        # 定义用于全局注意力的窗口块索引
        window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10],
        # 定义残差块索引
        residual_block_indices=[2, 5, 8, 11],
        out_features=["stage12"],
    )

    # 创建并返回 VitMatteConfig 实例，包含了 VitDetConfig 和隐藏层大小
    return VitMatteConfig(backbone_config=backbone_config, hidden_size=hidden_size)


# 创建需要重命名的键值对列表
def create_rename_keys(config):
    rename_keys = []

    # 格式化设置关闭以保留对应代码块的缩进
    # stem
    rename_keys.append(("backbone.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("backbone.patch_embed.proj.weight", "backbone.embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "backbone.embeddings.projection.bias"))

    return rename_keys


# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    # 获取配置信息
    config = get_config(model_name)

    # 加载原始状态字典
    model_name_to_filename = {
        "vitmatte-small-composition-1k": "ViTMatte_S_Com.pth",
        "vitmatte-base-composition-1k": "ViTMatte_B_Com.pth",
        "vitmatte-small-distinctions-646": "ViTMatte_S_DIS.pth",
        "vitmatte-base-distinctions-646": "ViTMatte_B_DIS.pth",
    }

    filename = model_name_to_filename[model_name]
    # 从 HF Hub 下载模型文件路径
    filepath = hf_hub_download(repo_id="nielsr/vitmatte-checkpoints", filename=filename, repo_type="model")
    # 使用 torch.load() 加载模型文件到 state_dict 中，使用 CPU 进行映射
    state_dict = torch.load(filepath, map_location="cpu")

    # 待续：重命名键


这段代码中，我们需要继续完成 `convert_vitmatte_checkpoint` 函数内的代码注释。
    # 遍历 state_dict 的拷贝中的所有键
    for key in state_dict.copy().keys():
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 如果键中包含 "backbone.blocks"，替换为 "backbone.encoder.layer"
        if "backbone.blocks" in key:
            key = key.replace("backbone.blocks", "backbone.encoder.layer")
        # 如果键中包含 "attn"，替换为 "attention"
        if "attn" in key:
            key = key.replace("attn", "attention")
        # 如果键中包含 "fusion_blks"，替换为 "fusion_blocks"
        if "fusion_blks" in key:
            key = key.replace("fusion_blks", "fusion_blocks")
        # 如果键中包含 "bn"，替换为 "batch_norm"
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        # 将更新后的键和原始值存回 state_dict
        state_dict[key] = val

    # 创建重命名后的键列表
    rename_keys = create_rename_keys(config)
    # 遍历重命名列表，逐一更新 state_dict 的键
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 创建模型处理器对象
    processor = VitMatteImageProcessor()
    # 创建 VitMatte 模型对象
    model = VitMatteForImageMatting(config)
    # 设置模型为评估模式
    model.eval()

    # 加载 state_dict 到模型
    model.load_state_dict(state_dict)

    # 从网络获取示例图像并转换为 RGB 格式
    url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_rgb.png?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # 从网络获取示例图像的 trimap
    url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_trimap.png?raw=true"
    trimap = Image.open(requests.get(url, stream=True).raw)

    # 使用 processor 处理图像和 trimap，返回像素值张量
    pixel_values = processor(images=image, trimaps=trimap.convert("L"), return_tensors="pt").pixel_values

    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型预测 alpha 通道值
        alphas = model(pixel_values).alphas

    # 根据模型名称选择期望的 alpha 值切片
    if model_name == "vitmatte-small-composition-1k":
        expected_slice = torch.tensor([[0.9977, 0.9987, 0.9990], [0.9980, 0.9998, 0.9998], [0.9983, 0.9998, 0.9998]])
    elif model_name == "vitmatte-base-composition-1k":
        expected_slice = torch.tensor([[0.9972, 0.9971, 0.9981], [0.9948, 0.9987, 0.9994], [0.9963, 0.9992, 0.9995]])
    elif model_name == "vitmatte-small-distinctions-646":
        expected_slice = torch.tensor([[0.9880, 0.9970, 0.9972], [0.9960, 0.9996, 0.9997], [0.9963, 0.9996, 0.9997]])
    elif model_name == "vitmatte-base-distinctions-646":
        expected_slice = torch.tensor([[0.9963, 0.9998, 0.9999], [0.9995, 1.0000, 1.0000], [0.9992, 0.9999, 1.0000]])

    # 断言模型预测的 alpha 值切片与期望的切片在指定的容差范围内相近
    assert torch.allclose(alphas[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 打印确认消息
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存文件夹路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型和处理器的消息
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果指定推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的消息
        print(f"Pushing model and processor for {model_name} to hub")
        # 推送模型到指定 Hub 仓库
        model.push_to_hub(f"hustvl/{model_name}")
        # 推送处理器到指定 Hub 仓库
        processor.push_to_hub(f"hustvl/{model_name}")
if __name__ == "__main__":
    # 如果这个脚本是直接运行的主程序，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必需的参数
    parser.add_argument(
        "--model_name",
        default="vitmatte-small-composition-1k",
        type=str,
        choices=[
            "vitmatte-small-composition-1k",
            "vitmatte-base-composition-1k",
            "vitmatte-small-distinctions-646",
            "vitmatte-base-distinctions-646",
        ],
        help="Name of the VitMatte model you'd like to convert."
    )
    # 添加一个参数选项，用于指定 VitMatte 模型的名称，有预设的几个选择

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数选项，用于指定输出 PyTorch 模型的目录路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个参数选项，表示是否将转换后的模型推送到 🤗 hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_vitmatte_checkpoint，传入命令行参数中指定的模型名称、输出目录路径和是否推送到 hub 的选项
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```