# `.\transformers\models\vitmatte\convert_vitmatte_to_hf.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，使用此文件需要遵守许可证规定
# 可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
# 用于从原始存储库转换 VitMatte 检查点
# URL: https://github.com/hustvl/ViTMatte

import argparse  # 导入解析命令行参数的模块
import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型
from PIL import Image  # 导入 Python Imaging Library 用于图像处理

from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor  # 导入 VitMatte 相关模块

# 获取配置信息
def get_config(model_name):
    hidden_size = 384 if "small" in model_name else 768
    num_attention_heads = 6 if "small" in model_name else 12

    # 设置 VitDetConfig 配置信息
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
        window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10],
        residual_block_indices=[2, 5, 8, 11],
        out_features=["stage12"],
    )

    return VitMatteConfig(backbone_config=backbone_config, hidden_size=hidden_size)

# 创建需要重命名的键值对列表
def create_rename_keys(config):
    rename_keys = []

    # stem
    rename_keys.append(("backbone.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("backbone.patch_embed.proj.weight", "backbone.embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "backbone.embeddings.projection.bias"))

    return rename_keys

# 重命名键值对
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 转换 VitMatte 检查点
def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_config(model_name)

    # 加载原始状态字典
    model_name_to_filename = {
        "vitmatte-small-composition-1k": "ViTMatte_S_Com.pth",
        "vitmatte-base-composition-1k": "ViTMatte_B_Com.pth",
        "vitmatte-small-distinctions-646": "ViTMatte_S_DIS.pth",
        "vitmatte-base-distinctions-646": "ViTMatte_B_DIS.pth",
    }

    filename = model_name_to_filename[model_name]
    filepath = hf_hub_download(repo_id="nielsr/vitmatte-checkpoints", filename=filename, repo_type="model")
    state_dict = torch.load(filepath, map_location="cpu")

    # 重命名键
    # 遍历 state_dict 的副本中的所有键
    for key in state_dict.copy().keys():
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 如果键中包含"backbone.blocks"，则替换为"backbone.encoder.layer"
        if "backbone.blocks" in key:
            key = key.replace("backbone.blocks", "backbone.encoder.layer")
        # 如果键中包含"attn"，则替换为"attention"
        if "attn" in key:
            key = key.replace("attn", "attention")
        # 如果键中包含"fusion_blks"，则替换为"fusion_blocks"
        if "fusion_blks" in key:
            key = key.replace("fusion_blks", "fusion_blocks")
        # 如果键中包含"bn"，则替换为"batch_norm"
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        # 将更新后的键值对重新加入 state_dict
        state_dict[key] = val

    # 生成重命名键的列表
    rename_keys = create_rename_keys(config)
    # 遍历重命名键列表，对 state_dict 进行键的重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 创建 VitMatteImageProcessor 实例
    processor = VitMatteImageProcessor()
    # 创建 VitMatteForImageMatting 模型实例
    model = VitMatteForImageMatting(config)
    # 设置模型为评估模式
    model.eval()

    # 加载 state_dict 到模型中
    model.load_state_dict(state_dict)

    # 从网络加载示例图片和 trimap
    url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_rgb.png?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_trimap.png?raw=true"
    trimap = Image.open(requests.get(url, stream=True).raw)

    # 使用 processor 处理图片和 trimap，返回像素值
    pixel_values = processor(images=image, trimaps=trimap.convert("L"), return_tensors="pt").pixel_values

    # 使用模型预测 alpha 值
    with torch.no_grad():
        alphas = model(pixel_values).alphas

    # 根据模型名称设置预期的 alpha 值切片
    if model_name == "vitmatte-small-composition-1k":
        expected_slice = torch.tensor([[0.9977, 0.9987, 0.9990], [0.9980, 0.9998, 0.9998], [0.9983, 0.9998, 0.9998]])
    elif model_name == "vitmatte-base-composition-1k":
        expected_slice = torch.tensor([[0.9972, 0.9971, 0.9981], [0.9948, 0.9987, 0.9994], [0.9963, 0.9992, 0.9995]])
    elif model_name == "vitmatte-small-distinctions-646":
        expected_slice = torch.tensor([[0.9880, 0.9970, 0.9972], [0.9960, 0.9996, 0.9997], [0.9963, 0.9996, 0.9997]])
    elif model_name == "vitmatte-base-distinctions-646":
        expected_slice = torch.tensor([[0.9963, 0.9998, 0.9999], [0.9995, 1.0000, 1.0000], [0.9992, 0.9999, 1.0000]])

    # 断言模型预测的 alpha 值与预期值接近
    assert torch.allclose(alphas[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 打印结果
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和 processor
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"hustvl/{model_name}")
        processor.push_to_hub(f"hustvl/{model_name}")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
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
        help="Name of the VitMatte model you'd like to convert.",
    )
    # 添加参数：输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数：是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 VitMatte 检查点转换为 PyTorch 模型
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```