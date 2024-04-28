# `.\transformers\models\swin\convert_swin_timm_to_pytorch.py`

```
# 导入必要的模块和库


import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import requests  # 用于发送 HTTP 请求
import timm  # 用于加载和训练神经网络模型
import torch  # 用于构建和训练神经网络
from huggingface_hub import hf_hub_download  # 用于从 Hugging Face Hub 下载模型和文件
from PIL import Image  # 用于处理图像
from transformers import AutoImageProcessor, SwinConfig, SwinForImageClassification  # 用于图像处理和分类


# 定义函数 `get_swin_config`


def get_swin_config(swin_name):
    # 创建一个 SwinConfig 对象
    config = SwinConfig()
    # 将 swin_name 按照 "_" 分割成多个部分
    name_split = swin_name.split("_")

    # 获取模型大小
    model_size = name_split[1]
    # 获取图像大小
    img_size = int(name_split[4])
    # 获取窗口大小
    window_size = int(name_split[3][-1])

    # 根据模型大小设置不同的嵌入维度、层数和注意力头数
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

    # 根据模型名称设置不同的类别数量和标签映射
    if "in22k" in swin_name:
        num_classes = 21841
    else:
        num_classes = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        # 从 Hugging Face Hub 下载标签映射文件，并加载为字典
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 更新 SwinConfig 的属性值
    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

    return config


# 定义函数 `rename_key`


def rename_key(name):
    # 将模型参数名中的 "patch_embed.proj" 替换为 "embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # 将模型参数名中的 "patch_embed.norm" 替换为 "embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 将模型参数名中的 "layers" 替换为 "encoder."
    if "layers" in name:
        name = "encoder." + name
    # 将模型参数名中的 "attn.proj" 替换为 "attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    # 将模型参数名中的 "attn" 替换为 "attention.self"
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    # 将模型参数名中的 "norm1" 替换为 "layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 将模型参数名中的 "norm2" 替换为 "layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 将模型参数名中的 "mlp.fc1" 替换为 "intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 将模型参数名中的 "mlp.fc2" 替换为 "output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    # 将模型参数名中的 "norm.weight" 替换为 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    # 将模型参数名中的 "norm.bias" 替换为 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"

    # 将模型参数名中的 "head" 替换为 "classifier"，其他部分保持不变
    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swin." + name

    return name


# 定义函数 `convert_state_dict`


def convert_state_dict(orig_state_dict, model):

（以上部分注释请勿输出）
    # 遍历原始状态字典的键的副本
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键中包含"mask"，则跳过当前循环，进入下一次循环
        if "mask" in key:
            continue
        # 如果键中包含"qkv"，则执行以下操作
        elif "qkv" in key:
            # 根据"."分割键，获取层数和块数
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            # 获取模型中当前自注意力层的维度
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键中包含"weight"，则执行以下操作
            if "weight" in key:
                # 更新原始状态字典，将查询权重、键权重和值权重分别映射到对应的位置
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"] = val[:dim, :]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"] = val[-dim:, :]
            # 如果键中不包含"weight"，则执行以下操作
            else:
                # 更新原始状态字典，将查询偏置、键偏置和值偏置分别映射到对应的位置
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[:dim]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[-dim:]
        # 如果键中既不包含"mask"也不包含"qkv"，则执行以下操作
        else:
            # 更新原始状态字典，将重命名后的键与对应值存入字典
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict
def convert_swin_checkpoint(swin_name, pytorch_dump_folder_path):
    # 使用 timm 库创建指定预训练模型
    timm_model = timm.create_model(swin_name, pretrained=True)
    # 将模型设置为评估模式
    timm_model.eval()

    # 获取指定 Swin 模型的配置
    config = get_swin_config(swin_name)
    # 创建 SwinForImageClassification 模型
    model = SwinForImageClassification(config)
    # 将模型设置为评估模式
    model.eval()

    # 转换 timm_model 的状态字典以适应 model
    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    model.load_state_dict(new_state_dict)

    # 定义图片 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 从预训练模型的名称中创建 AutoImageProcessor 实例
    image_processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(swin_name.replace("_", "-")))
    # 使用 requests 库获取图片并以原始流的形式打开
    image = Image.open(requests.get(url, stream=True).raw)
    # 处理图片并返回模型输入
    inputs = image_processor(images=image, return_tensors="pt")

    # 通过 timm_model 对输入进行推理
    timm_outs = timm_model(inputs["pixel_values"])
    # 通过 model 对输入进行推理
    hf_outs = model(**inputs).logits

    # 断言两种推理结果在给定阈值下近似相等
    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # 打印信息，保存模型到指定路径
    print(f"Saving model {swin_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    # 打印信息，保存图片处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

# 判断是否以主程序方式执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--swin_name",
        default="swin_tiny_patch4_window7_224",
        type=str,
        help="Name of the Swin timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析并获取参数
    args = parser.parse_args()
    # 调用 convert_swin_checkpoint 函数进行转换
    convert_swin_checkpoint(args.swin_name, args.pytorch_dump_folder_path)
```