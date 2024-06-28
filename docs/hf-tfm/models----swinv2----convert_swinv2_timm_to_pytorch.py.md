# `.\models\swinv2\convert_swinv2_timm_to_pytorch.py`

```
# 设置编码格式为UTF-8

# 版权声明和许可信息，声明本代码版权归HuggingFace Inc.团队所有，并遵循Apache License 2.0许可
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Swinv2 checkpoints from the timm library."""

import argparse  # 导入命令行参数解析模块
import json  # 导入JSON处理模块
from pathlib import Path  # 导入路径操作模块

import requests  # 导入HTTP请求库
import timm  # 导入模型库timm
import torch  # 导入PyTorch深度学习库
from huggingface_hub import hf_hub_download  # 导入Hugging Face模型中心下载函数
from PIL import Image  # 导入PIL图像处理库

from transformers import AutoImageProcessor, Swinv2Config, Swinv2ForImageClassification  # 导入transformers库中相关模块


def get_swinv2_config(swinv2_name):
    config = Swinv2Config()  # 创建一个Swinv2Config配置对象
    name_split = swinv2_name.split("_")  # 使用下划线分割模型名称

    model_size = name_split[1]  # 提取模型尺寸信息
    if "to" in name_split[3]:
        img_size = int(name_split[3][-3:])  # 提取图像尺寸信息
    else:
        img_size = int(name_split[3])
    if "to" in name_split[2]:
        window_size = int(name_split[2][-2:])  # 提取窗口大小信息
    else:
        window_size = int(name_split[2][6:])

    # 根据模型尺寸选择对应的嵌入维度、深度和头数配置
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

    # 如果模型名称中包含'to'，设置预训练窗口大小配置
    if "to" in swinv2_name:
        config.pretrained_window_sizes = (12, 12, 12, 6)

    # 根据模型名称和数据集情况设置相应的类别数和标签映射
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

    # 设置配置对象的图像大小、类别数、嵌入维度、深度、头数和窗口大小
    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

    return config


def rename_key(name):
    # 如果文件名中包含 "patch_embed.proj"，替换为 "embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    
    # 如果文件名中包含 "patch_embed.norm"，替换为 "embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    
    # 如果文件名中包含 "layers"，在前面加上 "encoder."
    if "layers" in name:
        name = "encoder." + name
    
    # 如果文件名中包含 "attn.proj"，替换为 "attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    
    # 如果文件名中包含 "attn"，替换为 "attention.self"
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    
    # 如果文件名中包含 "norm1"，替换为 "layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    
    # 如果文件名中包含 "norm2"，替换为 "layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    
    # 如果文件名中包含 "mlp.fc1"，替换为 "intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    
    # 如果文件名中包含 "mlp.fc2"，替换为 "output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    
    # 如果文件名中包含 "q_bias"，替换为 "query.bias"
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    
    # 如果文件名中包含 "k_bias"，替换为 "key.bias"
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    
    # 如果文件名中包含 "v_bias"，替换为 "value.bias"
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    
    # 如果文件名中包含 "cpb_mlp"，替换为 "continuous_position_bias_mlp"
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    
    # 如果文件名为 "norm.weight"，替换为 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    
    # 如果文件名为 "norm.bias"，替换为 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"
    
    # 如果文件名中包含 "head"，替换为 "classifier"；否则在文件名前面加上 "swinv2."
    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swinv2." + name
    
    # 返回处理后的文件名
    return name
# 定义一个函数，用于转换模型的状态字典，以适配特定模型结构
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的键（复制的列表），逐一处理
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "mask"，则跳过当前循环
        if "mask" in key:
            continue
        # 如果键名中包含 "qkv"
        elif "qkv" in key:
            # 拆分键名为列表
            key_split = key.split(".")
            # 获取层号和块号
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            # 获取注意力机制的维度
            dim = model.swinv2.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键名中包含 "weight"
            if "weight" in key:
                # 更新状态字典，设置查询权重
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                # 更新状态字典，设置键权重
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                # 更新状态字典，设置值权重
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新状态字典，设置查询偏置
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                # 更新状态字典，设置键偏置
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                # 更新状态字典，设置值偏置
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        else:
            # 对于其余键，通过 rename_key 函数重命名后存储
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict


# 定义一个函数，用于将 timm 模型的状态字典转换为 swinv2 模型的状态字典
def convert_swinv2_checkpoint(swinv2_name, pytorch_dump_folder_path):
    # 使用 timm 库创建指定预训练模型的模型对象
    timm_model = timm.create_model(swinv2_name, pretrained=True)
    # 将模型设置为评估模式
    timm_model.eval()

    # 获取 swinv2 模型的配置
    config = get_swinv2_config(swinv2_name)
    # 创建 swinv2 模型对象
    model = Swinv2ForImageClassification(config)
    # 将 swinv2 模型设置为评估模式
    model.eval()

    # 转换 timm 模型的状态字典为适应 swinv2 模型的新状态字典
    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    # 加载新的状态字典到 swinv2 模型中
    model.load_state_dict(new_state_dict)

    # 定义要使用的示例图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 使用 AutoImageProcessor 从预训练模型加载图像处理器
    image_processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(swinv2_name.replace("_", "-")))
    # 打开图像并转换为 PIL 图像对象
    image = Image.open(requests.get(url, stream=True).raw)
    # 使用图像处理器将图像转换为模型输入的张量表示
    inputs = image_processor(images=image, return_tensors="pt")

    # 使用 timm 模型对输入图像进行推理
    timm_outs = timm_model(inputs["pixel_values"])
    # 使用 swinv2 模型对输入图像进行推理，获取分类 logits
    hf_outs = model(**inputs).logits

    # 断言两个模型输出的值在给定的误差范围内接近
    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # 打印保存模型的信息
    print(f"Saving model {swinv2_name} to {pytorch_dump_folder_path}")
    # 将 swinv2 模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 将模型推送到指定的 Hub 仓库
    model.push_to_hub(
        repo_path_or_name=Path(pytorch_dump_folder_path, swinv2_name),
        organization="nandwalritik",
        commit_message="Add model",
    )


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数说明
    # （这里省略了具体的参数添加，因为没有提供详细的代码示例）
    parser.add_argument(
        "--swinv2_name",  # 定义一个命令行参数，用于指定要转换的Swinv2模型的名称
        default="swinv2_tiny_patch4_window8_256",  # 默认参数值为"swinv2_tiny_patch4_window8_256"
        type=str,  # 参数类型为字符串
        help="Name of the Swinv2 timm model you'd like to convert.",  # 参数的帮助信息，解释了该参数的作用
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 定义另一个命令行参数，用于指定输出PyTorch模型的目录路径
        default=None,  # 默认值为None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 参数的帮助信息，解释了该参数的作用
    )

    args = parser.parse_args()  # 解析命令行参数，将参数存储在args对象中
    convert_swinv2_checkpoint(args.swinv2_name, args.pytorch_dump_folder_path)
    # 调用函数convert_swinv2_checkpoint，传入解析后的参数args中的swinv2_name和pytorch_dump_folder_path作为参数
```