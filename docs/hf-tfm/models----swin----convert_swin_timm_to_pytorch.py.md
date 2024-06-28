# `.\models\swin\convert_swin_timm_to_pytorch.py`

```py
# 导入必要的模块：命令行参数解析、JSON 数据处理、HTTP 请求、模型库导入、深度学习框架导入、图像处理库导入
import argparse
import json
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import AutoImageProcessor, SwinConfig, SwinForImageClassification

# 根据模型名称获取相应的 Swin Transformer 配置
def get_swin_config(swin_name):
    config = SwinConfig()

    # 解析模型名称
    name_split = swin_name.split("_")
    
    # 根据模型名称中的信息设置不同的参数
    model_size = name_split[1]
    img_size = int(name_split[4])
    window_size = int(name_split[3][-1])

    if model_size == "tiny":
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "small":
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "base":
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    else:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)

    # 根据模型名称中的信息设置不同的分类数和标签映射
    if "in22k" in swin_name:
        num_classes = 21841
    else:
        num_classes = 1000
        # 从 Hugging Face Hub 下载并加载 ImageNet 分类标签映射
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 设置 Swin Transformer 配置对象的各项参数
    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

    return config

# 根据模型中的参数名字进行重命名，以适应不同的模型加载需求
def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swin." + name

    return name

# 将原始模型状态字典进行转换以适应不同命名的加载
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的键列表的副本
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值，并赋给变量val
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "mask"，则跳过处理当前循环的剩余部分
        if "mask" in key:
            continue
        # 如果键名中包含 "qkv"
        elif "qkv" in key:
            # 将键名按 "." 分割成列表
            key_split = key.split(".")
            # 从键名中获取层号和块号，并转换为整数
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            # 获取模型中对应位置的注意力机制的维度大小
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键名中包含 "weight"
            if "weight" in key:
                # 更新原始状态字典中的权重相关键值对
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]  # 更新查询权重
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim:dim * 2, :]  # 更新键权重
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]  # 更新值权重
            else:
                # 更新原始状态字典中的偏置相关键值对
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]  # 更新查询偏置
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim:dim * 2]  # 更新键偏置
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]  # 更新值偏置
        else:
            # 对于不包含 "mask" 和 "qkv" 的键名，通过特定函数重命名后更新原始状态字典
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict
# 导入必要的库和模块
import argparse
import requests
from PIL import Image
import torch
import timm
from transformers import AutoImageProcessor
from swin_transformer import SwinForImageClassification
from utils import convert_state_dict, get_swin_config

# 定义函数用于将 timm 模型转换为 Swin 模型
def convert_swin_checkpoint(swin_name, pytorch_dump_folder_path):
    # 使用 timm 库创建指定预训练模型，并设置为评估模式
    timm_model = timm.create_model(swin_name, pretrained=True)
    timm_model.eval()

    # 获取指定 Swin 模型配置
    config = get_swin_config(swin_name)
    # 使用配置创建 SwinForImageClassification 模型对象，并设置为评估模式
    model = SwinForImageClassification(config)
    model.eval()

    # 转换 timm 模型的状态字典到 Swin 模型兼容格式
    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    # 加载新的状态字典到 Swin 模型
    model.load_state_dict(new_state_dict)

    # 指定测试用的图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 从预训练模型名称创建图像处理器对象
    image_processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(swin_name.replace("_", "-")))
    # 从 URL 获取图像的流数据并打开为图像对象
    image = Image.open(requests.get(url, stream=True).raw)
    # 使用图像处理器处理图像，并转换为 PyTorch 张量
    inputs = image_processor(images=image, return_tensors="pt")

    # 使用 timm 模型处理图像输入并获取输出
    timm_outs = timm_model(inputs["pixel_values"])
    # 使用 Swin 模型处理图像输入并获取输出 logits
    hf_outs = model(**inputs).logits

    # 断言两个模型输出在指定的绝对误差范围内相等
    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # 打印保存模型的信息，包括模型名称和保存路径
    print(f"Saving model {swin_name} to {pytorch_dump_folder_path}")
    # 将 Swin 模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 打印保存图像处理器的信息，包括保存路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到同一路径
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数：swin_name，指定要转换的 Swin timm 模型名称
    parser.add_argument(
        "--swin_name",
        default="swin_tiny_patch4_window7_224",
        type=str,
        help="Name of the Swin timm model you'd like to convert.",
    )
    # 添加必需参数：pytorch_dump_folder_path，指定 PyTorch 模型输出目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_swin_checkpoint 函数，传入解析后的参数值
    convert_swin_checkpoint(args.swin_name, args.pytorch_dump_folder_path)
```