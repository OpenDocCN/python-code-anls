# `.\models\upernet\convert_convnext_upernet_to_pytorch.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ConvNext + UperNet checkpoints from mmsegmentation."""

import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 数据的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from PIL import Image  # 导入处理图像的 PIL 库

from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation  # 导入模型配置和语义分割相关的类


def get_upernet_config(model_name):
    auxiliary_in_channels = 384  # 初始化辅助输入通道数为 384
    if "tiny" in model_name:
        depths = [3, 3, 9, 3]  # 如果模型名中包含 "tiny"，则设置深度列表
        hidden_sizes = [96, 192, 384, 768]  # 设置隐藏层大小列表
    if "small" in model_name:
        depths = [3, 3, 27, 3]  # 如果模型名中包含 "small"，则设置深度列表
        hidden_sizes = [96, 192, 384, 768]  # 设置隐藏层大小列表
    if "base" in model_name:
        depths = [3, 3, 27, 3]  # 如果模型名中包含 "base"，则设置深度列表
        hidden_sizes = [128, 256, 512, 1024]  # 设置隐藏层大小列表
        auxiliary_in_channels = 512  # 设置辅助输入通道数为 512
    if "large" in model_name:
        depths = [3, 3, 27, 3]  # 如果模型名中包含 "large"，则设置深度列表
        hidden_sizes = [192, 384, 768, 1536]  # 设置隐藏层大小列表
        auxiliary_in_channels = 768  # 设置辅助输入通道数为 768
    if "xlarge" in model_name:
        depths = [3, 3, 27, 3]  # 如果模型名中包含 "xlarge"，则设置深度列表
        hidden_sizes = [256, 512, 1024, 2048]  # 设置隐藏层大小列表
        auxiliary_in_channels = 1024  # 设置辅助输入通道数为 1024

    # 设置标签信息
    num_labels = 150  # 设置标签数量为 150
    repo_id = "huggingface/label-files"  # 仓库 ID
    filename = "ade20k-id2label.json"  # 文件名
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 从 Hub 下载并加载 ID 到标签的映射
    id2label = {int(k): v for k, v in id2label.items()}  # 转换为整数类型的字典
    label2id = {v: k for k, v in id2label.items()}  # 反向映射，从标签到 ID 的字典

    backbone_config = ConvNextConfig(
        depths=depths, hidden_sizes=hidden_sizes, out_features=["stage1", "stage2", "stage3", "stage4"]
    )  # 创建 ConvNext 模型的配置对象
    config = UperNetConfig(
        backbone_config=backbone_config,
        auxiliary_in_channels=auxiliary_in_channels,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )  # 创建 UperNet 模型的配置对象

    return config  # 返回配置对象


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []  # 初始化重命名键列表

    # fmt: off
    # stem
    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.embeddings.patch_embeddings.weight"))
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.embeddings.patch_embeddings.bias"))
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.embeddings.layernorm.weight"))
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.embeddings.layernorm.bias"))
    # stages
    # 遍历 backbone_config.depths 列表的长度，这里 i 是索引
    for i in range(len(config.backbone_config.depths)):
        # 遍历 config.backbone_config.depths[i] 次数，这里 j 是索引
        for j in range(config.backbone_config.depths[i]):
            # 将原始键值对映射到新的键值对，修改 gamma 参数的命名
            rename_keys.append((f"backbone.stages.{i}.{j}.gamma", f"backbone.encoder.stages.{i}.layers.{j}.layer_scale_parameter"))
            # 修改深度卷积的权重命名
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.weight", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.weight"))
            # 修改深度卷积的偏置命名
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.bias", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.bias"))
            # 修改归一化层权重命名
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.weight", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.weight"))
            # 修改归一化层偏置命名
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.bias", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.bias"))
            # 修改第一个点卷积层的权重命名
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.weight"))
            # 修改第一个点卷积层的偏置命名
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.bias"))
            # 修改第二个点卷积层的权重命名
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.weight"))
            # 修改第二个点卷积层的偏置命名
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.bias"))
        
        # 如果 i 大于 0，则需要处理下采样层的命名映射
        if i > 0:
            # 修改下采样层第一个卷积层的权重命名
            rename_keys.append((f"backbone.downsample_layers.{i}.0.weight", f"backbone.encoder.stages.{i}.downsampling_layer.0.weight"))
            # 修改下采样层第一个卷积层的偏置命名
            rename_keys.append((f"backbone.downsample_layers.{i}.0.bias", f"backbone.encoder.stages.{i}.downsampling_layer.0.bias"))
            # 修改下采样层第二个归一化层的权重命名
            rename_keys.append((f"backbone.downsample_layers.{i}.1.weight", f"backbone.encoder.stages.{i}.downsampling_layer.1.weight"))
            # 修改下采样层第二个归一化层的偏置命名
            rename_keys.append((f"backbone.downsample_layers.{i}.1.bias", f"backbone.encoder.stages.{i}.downsampling_layer.1.bias"))

        # 修改归一化层权重命名
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        # 修改归一化层偏置命名
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    # decode head 部分的命名映射
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )

    # 返回处理后的所有重命名映射列表
    return rename_keys
# 定义函数，用于将字典 dct 中的键 old 更名为 new，保持其对应的值不变
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将该值与新键 new 组成新的键值对，添加到字典 dct 中
    dct[new] = val


# 定义函数，用于从指定的 URL 下载指定模型的预训练检查点，并加载其状态字典
def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    # 模型名到预训练检查点 URL 的映射字典
    model_name_to_url = {
        "upernet-convnext-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth",
        "upernet-convnext-small": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208-1b1e394f.pth",
        "upernet-convnext-base": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth",
        "upernet-convnext-large": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth",
        "upernet-convnext-xlarge": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth",
    }
    
    # 根据给定的模型名获取对应的预训练检查点 URL
    checkpoint_url = model_name_to_url[model_name]
    # 使用 torch.hub 下载指定 URL 的模型状态字典，并存储在变量 state_dict 中
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    # 获取指定模型名的配置信息
    config = get_upernet_config(model_name)
    # 根据配置信息创建 UperNetForSemanticSegmentation 模型实例
    model = UperNetForSemanticSegmentation(config)
    # 设置模型为评估模式
    model.eval()

    # 将状态字典中所有键包含 "bn" 的项更名为包含 "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # 使用预定义函数 create_rename_keys(config) 创建需要重命名的键对列表 rename_keys
    rename_keys = create_rename_keys(config)
    # 遍历 rename_keys 列表，对状态字典中的键进行重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 使用更新后的状态字典加载模型参数
    model.load_state_dict(state_dict)

    # 从指定 URL 获取测试图像，并转换为 RGB 格式
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 创建 SegformerImageProcessor 实例处理图像
    processor = SegformerImageProcessor()
    # 将图像转换为 PyTorch 张量格式
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 关闭梯度计算，在模型推理时不计算梯度
    with torch.no_grad():
        # 使用模型进行图像的语义分割推理
        outputs = model(pixel_values)

    # 根据模型名选择对应的预期输出结果片段 expected_slice
    if model_name == "upernet-convnext-tiny":
        expected_slice = torch.tensor(
            [[-8.8110, -8.8110, -8.6521], [-8.8110, -8.8110, -8.6521], [-8.7746, -8.7746, -8.6130]]
        )
    elif model_name == "upernet-convnext-small":
        expected_slice = torch.tensor(
            [[-8.8236, -8.8236, -8.6771], [-8.8236, -8.8236, -8.6771], [-8.7638, -8.7638, -8.6240]]
        )
    elif model_name == "upernet-convnext-base":
        expected_slice = torch.tensor(
            [[-8.8558, -8.8558, -8.6905], [-8.8558, -8.8558, -8.6905], [-8.7669, -8.7669, -8.6021]]
        )
    # 如果模型名称为 "upernet-convnext-large"，设定期望的输出张量切片
    elif model_name == "upernet-convnext-large":
        expected_slice = torch.tensor(
            [[-8.6660, -8.6660, -8.6210], [-8.6660, -8.6660, -8.6210], [-8.6310, -8.6310, -8.5964]]
        )
    # 如果模型名称为 "upernet-convnext-xlarge"，设定期望的输出张量切片
    elif model_name == "upernet-convnext-xlarge":
        expected_slice = torch.tensor(
            [[-8.4980, -8.4980, -8.3977], [-8.4980, -8.4980, -8.3977], [-8.4379, -8.4379, -8.3412]]
        )
    # 打印模型输出的 logits 的部分内容，用于调试和验证
    print("Logits:", outputs.logits[0, 0, :3, :3])
    # 断言模型输出的 logits 的部分内容与预期的输出张量切片在给定的误差范围内相似
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 如果通过断言，则打印消息表示结果看起来正常
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型的消息，包括模型名称和保存路径
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印保存处理器的消息，包括保存路径
        print(f"Saving processor to {pytorch_dump_folder_path}")
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的消息，包括模型名称
        print(f"Pushing model and processor for {model_name} to hub")
        # 将模型推送到 Hub，命名为 "openmmlab/{model_name}"
        model.push_to_hub(f"openmmlab/{model_name}")
        # 将处理器推送到 Hub，命名为 "openmmlab/{model_name}"
        processor.push_to_hub(f"openmmlab/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而非被导入到其他脚本中），则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default="upernet-convnext-tiny",
        type=str,
        choices=[f"upernet-convnext-{size}" for size in ["tiny", "small", "base", "large", "xlarge"]],
        help="Name of the ConvNext UperNet model you'd like to convert."
    )
    # 添加模型名称参数，可以选择的值包括指定格式的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数，指定输出 PyTorch 模型的目录路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加参数，指定是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数，并存储在 args 对象中

    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_upernet_checkpoint，传递解析后的参数进行模型转换操作
```