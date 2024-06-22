# `.\transformers\models\swinv2\convert_swinv2_timm_to_pytorch.py`

```py
# 指定编码格式为 UTF-8
# 版权声明
# 根据Apache许可证2.0获得许可
# 可以从以下位置获取许可证的副本http://www.apache.org/licenses/LICENSE-2.0
# 未经授权，不得使用此文件或代码
# 除非适用法律或书面同意，否则不得以“现状”方式分发软件
# 无论是明示的还是默示的，包括但不限于适销性或特定用途的适用性，没有任何担保或条件
# 请参阅许可证以获取有关权限和限制的特定语言
# 转换来自timm库的Swinv2检查点

# 导入必要的库
import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, Swinv2Config, Swinv2ForImageClassification

# 获取Swinv2配置
def get_swinv2_config(swinv2_name):
    # 初始化Swinv2配置
    config = Swinv2Config()

    # 解析模型名称，获取模型尺寸、图像分辨率和窗口大小
    name_split = swinv2_name.split("_")
    model_size = name_split[1]
    if "to" in name_split[3]:
        img_size = int(name_split[3][-3:])
    else:
        img_size = int(name_split[3])
    if "to" in name_split[2]:
        window_size = int(name_split[2][-2:])
    else:
        window_size = int(name_split[2][6:])

    # 根据模型尺寸不同，设置不同的嵌入维度、层数和头数
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

    # 如果模型名称包含"to"，则设置预训练窗口大小
    if "to" in swinv2_name:
        config.pretrained_window_sizes = (12, 12, 12, 6)

    # 如果模型名称包含"22k"且不包含"to"，则设置类别数为21841
    # 否则，设置类别数为1000
    if ("22k" in swinv2_name) and ("to" not in swinv2_name):
        num_classes = 21841
        repo_id = "huggingface/label-files"
        filename = "imagenet-22k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        num_classes = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 设置Swinv2配置的图像大小、标签数、嵌入维度、层数、头数和窗口大小
    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

    return config

# 重新命名键
def rename_key(name):
    # 如果文件名中包含"patch_embed.proj"，则替换为"embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # 如果文件名中包含"patch_embed.norm"，则替换为"embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 如果文件名中包含"layers"，则在前面添加"encoder."
    if "layers" in name:
        name = "encoder." + name
    # 如果文件名中包含"attn.proj"，则替换为"attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    # 如果文件名中包含"attn"，则替换为"attention.self"
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    # 如果文件名中包含"norm1"，则替换为"layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果文件名中包含"norm2"，则替换为"layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果文件名中包含"mlp.fc1"，则替换为"intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果文件名中包含"mlp.fc2"，则替换为"output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果文件名中包含"q_bias"，则替换为"query.bias"
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    # 如果文件名中包含"k_bias"，则替换为"key.bias"
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    # 如果文件名中包含"v_bias"，则替换为"value.bias"
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    # 如果文件名中包含"cpb_mlp"，则替换为"continuous_position_bias_mlp"
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    # 如果文件名为"norm.weight"，则替换为"layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    # 如果文件名为"norm.bias"，则替换为"layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"

    # 如果文件名中包含"head"，则替换为"classifier"，否则在文件名前加上"swinv2."
    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swinv2." + name
    
    # 返回修改后的文件名
    return name
# 将原始状态字典转换为适合新模型的状态字典
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的键，获取键和对应值
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        # 如果键包含"mask"，则跳过该键
        if "mask" in key:
            continue
        # 如果键包含"qkv"，则进行下列操作
        elif "qkv" in key:
            # 拆分键并获取层号、块号、以及维度
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            dim = model.swinv2.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键中包含"weight"，则拆分值，并以新的键值的形式存储
            if "weight" in key:
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            # 否则将偏置值以新的键值的形式存储
            else:
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        # 如果键不包含"mask"或"qkv"，则重命名键并存储值
        else:
            orig_state_dict[rename_key(key)] = val

    # 返回转换后的状态字典
    return orig_state_dict


def convert_swinv2_checkpoint(swinv2_name, pytorch_dump_folder_path):
    # 加载预训练的timm模型，并设为评估模式
    timm_model = timm.create_model(swinv2_name, pretrained=True)
    timm_model.eval()

    # 获取Swin Transformer的配置并创建模型
    config = get_swinv2_config(swinv2_name)
    model = Swinv2ForImageClassification(config)
    model.eval()

    # 转换timm模型的状态字典，并加载到Swinv2模型中
    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    model.load_state_dict(new_state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 加载自动图像处理器，加载图像并获取处理后的输入
    image_processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(swinv2_name.replace("_", "-")))
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")

    # 将输入传入timm模型和Swinv2模型，获取输出并进行断言检查
    timm_outs = timm_model(inputs["pixel_values"])
    hf_outs = model(**inputs).logits

    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # 保存Swinv2模型和图像处理器
    print(f"Saving model {swinv2_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 推送模型至Hub
    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, swinv2_name),
        organization="nandwalritik",
        commit_message="Add model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    # 添加一个命令行参数，用来指定 Swinv2 模型的名称，默认为 "swinv2_tiny_patch4_window8_256"
    parser.add_argument(
        "--swinv2_name",
        default="swinv2_tiny_patch4_window8_256",
        type=str,
        help="Name of the Swinv2 timm model you'd like to convert.",
    )
    # 添加一个命令行参数，用来指定输出的 PyTorch 模型目录的路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将指定的 Swinv2 模型转换成 PyTorch 模型，并输出到指定目录
    convert_swinv2_checkpoint(args.swinv2_name, args.pytorch_dump_folder_path)
```