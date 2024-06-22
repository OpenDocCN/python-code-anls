# `.\transformers\models\upernet\convert_convnext_upernet_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权许可声明
# 通过提供的链接获取 Apache 2.0 许可证的副本
# 在适用法律要求或书面同意的情况下，按"原样"分发软件，不提供任何形式的担保或条件，无论是明示的还是隐含的，详见许可证
# 用于区分不同语言的具体性能和限制
"""从 mmsegmentation 转换 ConvNext + UperNet 检查点。

import argparse  # 导入处理命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型
from PIL import Image  # 导入 Python Imaging Library 用于处理图片

from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation  # 导入转换模型需要的类和接口


def get_upernet_config(model_name):  # 获取 UperNet 配置信息
    auxiliary_in_channels = 384  # 定义辅助输入通道
    if "tiny" in model_name:  # 如果模型名中包含 "tiny"
        depths = [3, 3, 9, 3]  # 设置深度
        hidden_sizes = [96, 192, 384, 768]  # 设置隐藏层大小
    if "small" in model_name:  # 如果模型名中包含 "small"
        depths = [3, 3, 27, 3]  # 设置深度
        hidden_sizes = [96, 192, 384, 768]  # 设置隐藏层大小    
    if "base" in model_name:  # 如果模型名中包含 "base"
        depths = [3, 3, 27, 3]  # 设置深度
        hidden_sizes = [128, 256, 512, 1024]  # 设置隐藏层大小
        auxiliary_in_channels = 512  # 更新辅助输入通道为 512
    if "large" in model_name:  # 如果模型名中包含 "large"
        depths = [3, 3, 27, 3]  # 设置深度
        hidden_sizes = [192, 384, 768, 1536]  # 设置隐藏层大小
        auxiliary_in_channels = 768  # 更新辅助输入通道为 768
    if "xlarge" in model_name:  # 如果模型名中包含 "xlarge"
        depths = [3, 3, 27, 3]  # 设置深度
        hidden_sizes = [256, 512, 1024, 2048]  # 设置隐藏层大小
        auxiliary_in_channels = 1024  # 更新辅助输入通道为 1024

    # 设置标签信息
    num_labels = 150  # 定义标签数量
    repo_id = "huggingface/label-files"  # 设置仓库 ID
    filename = "ade20k-id2label.json"  # 设置文件名
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 从 Hugging Face Hub 下载标签文件，并加载为 JSON
    id2label = {int(k): v for k, v in id2label.items()}  # 转换标签字典的键为整数
    label2id = {v: k for k, v in id2label.items()}  # 反转标签字典的键值对

    backbone_config = ConvNextConfig(  # 定义 ConvNext 骨干网络配置
        depths=depths,  # 设置深度
        hidden_sizes=hidden_sizes,  # 设置隐藏层大小
        out_features=["stage1", "stage2", "stage3", "stage4"]  # 设置输出特征
    )
    config = UperNetConfig(  # 定义 UperNet 配置
        backbone_config=backbone_config,  # 设置 ConvNext 骨干网络配置
        auxiliary_in_channels=auxiliary_in_channels,  # 设置辅助输入通道
        num_labels=num_labels,  # 设置标签数量
        id2label=id2label,  # 设置 ID 到标签的映射
        label2id=label2id,  # 设置标签到 ID 的映射
    )

    return config  # 返回 UperNet 配置

# 这里列出了所有需要重命名的键（左侧是原始名称，右侧是我们的名称）
def create_rename_keys(config):  # 创建重命名键
    rename_keys = []  # 初始化重命名键列表

    # fmt: off
    # stem
    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.embeddings.patch_embeddings.weight"))  # 添加一个键值对到重命名键列表
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.embeddings.patch_embeddings.bias"))  # 添加一个键值对到重命名键列表
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.embeddings.layernorm.weight"))  # 添加一个键值对到重命名键列表
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.embeddings.layernorm.bias"))  # 添加一个键值对到重命名键列表
    # stages
    # 遍历 backbone_config.depths 中的每个元素，表示网络的深度
    for i in range(len(config.backbone_config.depths)):
        # 遍历当前深度下的层数
        for j in range(config.backbone_config.depths[i]):
            # 重命名 backbone 中的参数到 encoder 中对应的参数
            rename_keys.append((f"backbone.stages.{i}.{j}.gamma", f"backbone.encoder.stages.{i}.layers.{j}.layer_scale_parameter"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.weight", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.bias", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.weight", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.bias", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.bias"))
        # 如果当前深度大于 0
        if i > 0:
            # 重命名 backbone 中的下采样层参数到 encoder 中对应的参数
            rename_keys.append((f"backbone.downsample_layers.{i}.0.weight", f"backbone.encoder.stages.{i}.downsampling_layer.0.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.0.bias", f"backbone.encoder.stages.{i}.downsampling_layer.0.bias"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.weight", f"backbone.encoder.stages.{i}.downsampling_layer.1.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.bias", f"backbone.encoder.stages.{i}.downsampling_layer.1.bias"))

        # 重命名 backbone 中的归一化层参数到 encoder 中对应的参数
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    # decode head 和 auxiliary head 参数重命名
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # 返回重命名后的参数列表
    return rename_keys
# 定义一个函数，用于重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将该值使用新键重新添加到字典中
    dct[new] = val


# 定义一个函数，用于将 upernet 模型的检查点转换为 PyTorch 模型
def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    # 定义模型名称到下载链接的映射关系
    model_name_to_url = {
        "upernet-convnext-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth",
        "upernet-convnext-small": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208-1b1e394f.pth",
        "upernet-convnext-base": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth",
        "upernet-convnext-large": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth",
        "upernet-convnext-xlarge": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth",
    }
    # 获取指定模型名称对应的检查点下载链接
    checkpoint_url = model_name_to_url[model_name]
    # 从指定 URL 加载模型的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    # 获取指定模型名称的配置信息
    config = get_upernet_config(model_name)
    # 创建 UperNetForSemanticSegmentation 模型
    model = UperNetForSemanticSegmentation(config)
    # 设置为评估模式
    model.eval()

    # 将状态字典中键中包含 "bn" 的部分替换为 "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # 重命名状态字典中的键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 加载状态字典到模型中
    model.load_state_dict(state_dict)

    # 用图像验证模型的效果
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 创建 SegformerImageProcessor 对象
    processor = SegformerImageProcessor()
    # 将图像处理为模型所需的张量格式
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 用无梯度计算的方式获取模型的输出结果
    with torch.no_grad():
        outputs = model(pixel_values)

    # 如果模型名称为 "upernet-convnext-tiny"，提供预期输出的一部分
    if model_name == "upernet-convnext-tiny":
        expected_slice = torch.tensor(
            [[-8.8110, -8.8110, -8.6521], [-8.8110, -8.8110, -8.6521], [-8.7746, -8.7746, -8.6130]]
        )
    # 如果模型名称为 "upernet-convnext-small"，提供预期输出的一部分
    elif model_name == "upernet-convnext-small":
        expected_slice = torch.tensor(
            [[-8.8236, -8.8236, -8.6771], [-8.8236, -8.8236, -8.6771], [-8.7638, -8.7638, -8.6240]]
        )
    # 如果模型名称为 "upernet-convnext-base"，提供预期输出的一部分
    elif model_name == "upernet-convnext-base":
        expected_slice = torch.tensor(
            [[-8.8558, -8.8558, -8.6905], [-8.8558, -8.8558, -8.6905], [-8.7669, -8.7669, -8.6021]]
        )
    # 如果模型名称为 "upernet-convnext-large"，定义期望的输出 logits 矩阵
    elif model_name == "upernet-convnext-large":
        expected_slice = torch.tensor(
            [[-8.6660, -8.6660, -8.6210], [-8.6660, -8.6660, -8.6210], [-8.6310, -8.6310, -8.5964]]
        )
    # 如果模型名称为 "upernet-convnext-xlarge"，定义期望的输出 logits 矩阵    
    elif model_name == "upernet-convnext-xlarge":
        expected_slice = torch.tensor(
            [[-8.4980, -8.4980, -8.3977], [-8.4980, -8.4980, -8.3977], [-8.4379, -8.4379, -8.3412]]
        )
    # 打印输出的前 3x3 logits 矩阵
    print("Logits:", outputs.logits[0, 0, :3, :3])
    # 检查当前输出的 logits 矩阵是否与期望的矩阵在指定精度内一致
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 如果检查通过，打印 "Looks ok!"
    print("Looks ok!")
    
    # 如果指定了模型保存路径，保存模型和处理器到该路径
    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)
    
    # 如果指定了推送到 Hub，将模型和处理器推送到 Hub
    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"openmmlab/{model_name}")
        processor.push_to_hub(f"openmmlab/{model_name}")
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_name",  # 模型名称参数
        default="upernet-convnext-tiny",  # 默认值为 "upernet-convnext-tiny"
        type=str,  # 参数类型为字符串
        choices=[f"upernet-convnext-{size}" for size in ["tiny", "small", "base", "large", "xlarge"]],  # 可选值为给定列表的各种组合
        help="Name of the ConvNext UperNet model you'd like to convert.",  # 帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch 模型输出文件夹路径参数
        default=None,  # 默认值为 None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 帮助信息
    )
    parser.add_argument(
        "--push_to_hub",  # 推送到 🤗 hub 参数
        action="store_true",  # 设置为真
        help="Whether or not to push the converted model to the 🤗 hub."  # 帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将模型文件转换为 PyTorch 模型
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```