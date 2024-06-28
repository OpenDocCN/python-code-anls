# `.\models\mobilevit\convert_mlcvnets_to_pytorch.py`

```
# 设置脚本的编码格式为UTF-8
# 版权声明，使用 Apache License, Version 2.0 许可协议
# 详细许可信息可以在 http://www.apache.org/licenses/LICENSE-2.0 找到
# 本脚本用于从 ml-cvnets 库中转换 MobileViT 模型检查点

# 引入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
from pathlib import Path  # 提供处理文件和目录路径的类和函数

import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 下载模型文件的辅助函数
from PIL import Image  # Python Imaging Library，处理图像的库

# 从 transformers 库中导入相关模块和函数
from transformers import (
    MobileViTConfig,  # MobileViT 模型配置类
    MobileViTForImageClassification,  # MobileViT 图像分类模型
    MobileViTForSemanticSegmentation,  # MobileViT 语义分割模型
    MobileViTImageProcessor,  # MobileViT 图像处理器
)
from transformers.utils import logging  # transformers 模块的日志记录工具

# 设置日志记录器的详细程度为 INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def get_mobilevit_config(mobilevit_name):
    # 创建一个空的 MobileViTConfig 配置对象
    config = MobileViTConfig()

    # 根据模型名称设置不同的隐藏层大小和颈部隐藏层大小
    if "mobilevit_s" in mobilevit_name:
        config.hidden_sizes = [144, 192, 240]
        config.neck_hidden_sizes = [16, 32, 64, 96, 128, 160, 640]
    elif "mobilevit_xs" in mobilevit_name:
        config.hidden_sizes = [96, 120, 144]
        config.neck_hidden_sizes = [16, 32, 48, 64, 80, 96, 384]
    elif "mobilevit_xxs" in mobilevit_name:
        config.hidden_sizes = [64, 80, 96]
        config.neck_hidden_sizes = [16, 16, 24, 48, 64, 80, 320]
        config.hidden_dropout_prob = 0.05
        config.expand_ratio = 2.0

    # 根据模型名称设置不同的图片大小、输出步长和标签数
    if mobilevit_name.startswith("deeplabv3_"):
        config.image_size = 512
        config.output_stride = 16
        config.num_labels = 21
        filename = "pascal-voc-id2label.json"
    else:
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"

    # 从 Hugging Face Hub 下载标签映射文件，并加载为 JSON 格式
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name, base_model=False):
    # 根据模型结构重命名模型参数名称中的关键部分
    for i in range(1, 6):
        if f"layer_{i}." in name:
            name = name.replace(f"layer_{i}.", f"encoder.layer.{i - 1}.")

    if "conv_1." in name:
        name = name.replace("conv_1.", "conv_stem.")
    if ".block." in name:
        name = name.replace(".block.", ".")

    if "exp_1x1" in name:
        name = name.replace("exp_1x1", "expand_1x1")
    if "red_1x1" in name:
        name = name.replace("red_1x1", "reduce_1x1")
    if ".local_rep.conv_3x3." in name:
        name = name.replace(".local_rep.conv_3x3.", ".conv_kxk.")
    if ".local_rep.conv_1x1." in name:
        name = name.replace(".local_rep.conv_1x1.", ".conv_1x1.")
    # 如果文件名中包含".norm."，替换为".normalization."
    if ".norm." in name:
        name = name.replace(".norm.", ".normalization.")
    
    # 如果文件名中包含".conv."，替换为".convolution."
    if ".conv." in name:
        name = name.replace(".conv.", ".convolution.")
    
    # 如果文件名中包含".conv_proj."，替换为".conv_projection."
    if ".conv_proj." in name:
        name = name.replace(".conv_proj.", ".conv_projection.")
    
    # 替换文件名中格式为".i.j."的部分为".i.layer.j."
    for i in range(0, 2):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.layer.{j}.")
    
    # 替换文件名中格式为".i.j."的部分为".i."，并根据特定条件进一步修改
    for i in range(2, 6):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.")
                # 如果文件名中包含特定关键词，进行进一步替换
                if "expand_1x1" in name:
                    name = name.replace("expand_1x1", "downsampling_layer.expand_1x1")
                if "conv_3x3" in name:
                    name = name.replace("conv_3x3", "downsampling_layer.conv_3x3")
                if "reduce_1x1" in name:
                    name = name.replace("reduce_1x1", "downsampling_layer.reduce_1x1")
    
    # 替换文件名中格式为".global_rep.i.weight"的部分为".layernorm.weight"
    for i in range(2, 5):
        if f".global_rep.{i}.weight" in name:
            name = name.replace(f".global_rep.{i}.weight", ".layernorm.weight")
        if f".global_rep.{i}.bias" in name:
            name = name.replace(f".global_rep.{i}.bias", ".layernorm.bias")
    
    # 如果文件名中包含".global_rep."，替换为".transformer."
    if ".global_rep." in name:
        name = name.replace(".global_rep.", ".transformer.")
    
    # 如果文件名中包含".pre_norm_mha.0."，替换为".layernorm_before."
    if ".pre_norm_mha.0." in name:
        name = name.replace(".pre_norm_mha.0.", ".layernorm_before.")
    
    # 如果文件名中包含".pre_norm_mha.1.out_proj."，替换为".attention.output.dense."
    if ".pre_norm_mha.1.out_proj." in name:
        name = name.replace(".pre_norm_mha.1.out_proj.", ".attention.output.dense.")
    
    # 如果文件名中包含".pre_norm_ffn.0."，替换为".layernorm_after."
    if ".pre_norm_ffn.0." in name:
        name = name.replace(".pre_norm_ffn.0.", ".layernorm_after.")
    
    # 如果文件名中包含".pre_norm_ffn.1."，替换为".intermediate.dense."
    if ".pre_norm_ffn.1." in name:
        name = name.replace(".pre_norm_ffn.1.", ".intermediate.dense.")
    
    # 如果文件名中包含".pre_norm_ffn.4."，替换为".output.dense."
    if ".pre_norm_ffn.4." in name:
        name = name.replace(".pre_norm_ffn.4.", ".output.dense.")
    
    # 如果文件名中包含".transformer."，替换为".transformer.layer."
    if ".transformer." in name:
        name = name.replace(".transformer.", ".transformer.layer.")
    
    # 如果文件名中包含".aspp_layer."，替换为"."
    if ".aspp_layer." in name:
        name = name.replace(".aspp_layer.", ".")
    
    # 如果文件名中包含".aspp_pool."，替换为"."
    if ".aspp_pool." in name:
        name = name.replace(".aspp_pool.", ".")
    
    # 如果文件名中包含"seg_head."，替换为"segmentation_head."
    if "seg_head." in name:
        name = name.replace("seg_head.", "segmentation_head.")
    
    # 如果文件名中包含"segmentation_head.classifier.classifier."，替换为"segmentation_head.classifier."
    if "segmentation_head.classifier.classifier." in name:
        name = name.replace("segmentation_head.classifier.classifier.", "segmentation_head.classifier.")
    
    # 如果文件名中包含"classifier.fc."，替换为"classifier."
    if "classifier.fc." in name:
        name = name.replace("classifier.fc.", "classifier.")
    # 否则，如果base_model为假且文件名中不包含"segmentation_head."，在文件名前加上"mobilevit."
    elif (not base_model) and ("segmentation_head." not in name):
        name = "mobilevit." + name
    
    # 返回修改后的文件名
    return name
# 定义函数，将原始状态字典转换为适合移动ViT模型的状态字典
def convert_state_dict(orig_state_dict, model, base_model=False):
    # 如果是基础模型，则模型前缀为空字符串
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevit."

    # 遍历原始状态字典的复制键列表
    for key in orig_state_dict.copy().keys():
        # 弹出键值对，并用变量val接收值
        val = orig_state_dict.pop(key)

        # 如果键以"encoder."开头，则去除这个前缀
        if key[:8] == "encoder.":
            key = key[8:]

        # 如果键中包含"qkv"，则处理注意力权重和偏置
        if "qkv" in key:
            # 分割键名，并解析出层编号和变压器编号
            key_split = key.split(".")
            layer_num = int(key_split[0][6:]) - 1
            transformer_num = int(key_split[3])

            # 获取指定层的注意力头尺寸
            layer = model.get_submodule(f"{model_prefix}encoder.layer.{layer_num}")
            dim = layer.transformer.layer[transformer_num].attention.attention.all_head_size

            # 构造权重或偏置的前缀路径
            prefix = (
                f"{model_prefix}encoder.layer.{layer_num}.transformer.layer.{transformer_num}.attention.attention."
            )

            # 根据键名中是否包含"weight"，更新相应的权重或偏置值
            if "weight" in key:
                orig_state_dict[prefix + "query.weight"] = val[:dim, :]
                orig_state_dict[prefix + "key.weight"] = val[dim : dim * 2, :]
                orig_state_dict[prefix + "value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[prefix + "query.bias"] = val[:dim]
                orig_state_dict[prefix + "key.bias"] = val[dim : dim * 2]
                orig_state_dict[prefix + "value.bias"] = val[-dim:]
        else:
            # 对于其他键名，使用自定义函数rename_key重命名键后放回原始状态字典
            orig_state_dict[rename_key(key, base_model)] = val

    # 返回转换后的原始状态字典
    return orig_state_dict


# 使用torch.no_grad()修饰，定义函数，将原始权重加载到MobileViT结构中
@torch.no_grad()
def convert_movilevit_checkpoint(mobilevit_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileViT structure.
    """
    # 获取MobileViT配置
    config = get_mobilevit_config(mobilevit_name)

    # 加载原始状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 根据模型名称选择合适的MobileViT模型
    if mobilevit_name.startswith("deeplabv3_"):
        model = MobileViTForSemanticSegmentation(config).eval()
    else:
        model = MobileViTForImageClassification(config).eval()

    # 转换原始状态字典，并加载到模型中
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 使用MobileViTImageProcessor准备图像
    image_processor = MobileViTImageProcessor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    
    # 对准备好的图像进行模型推理
    outputs = model(**encoding)
    logits = outputs.logits
    # 检查 mobilevit_name 是否以 "deeplabv3_" 开头
    if mobilevit_name.startswith("deeplabv3_"):
        # 断言 logits 的形状应为 (1, 21, 32, 32)
        assert logits.shape == (1, 21, 32, 32)

        # 根据不同的 mobilevit_name 设置期望的 logits
        if mobilevit_name == "deeplabv3_mobilevit_s":
            expected_logits = torch.tensor(
                [
                    [[6.2065, 6.1292, 6.2070], [6.1079, 6.1254, 6.1747], [6.0042, 6.1071, 6.1034]],
                    [[-6.9253, -6.8653, -7.0398], [-7.3218, -7.3983, -7.3670], [-7.1961, -7.2482, -7.1569]],
                    [[-4.4723, -4.4348, -4.3769], [-5.3629, -5.4632, -5.4598], [-5.1587, -5.3402, -5.5059]],
                ]
            )
        elif mobilevit_name == "deeplabv3_mobilevit_xs":
            expected_logits = torch.tensor(
                [
                    [[5.4449, 5.5733, 5.6314], [5.1815, 5.3930, 5.5963], [5.1656, 5.4333, 5.4853]],
                    [[-9.4423, -9.7766, -9.6714], [-9.1581, -9.5720, -9.5519], [-9.1006, -9.6458, -9.5703]],
                    [[-7.7721, -7.3716, -7.1583], [-8.4599, -8.0624, -7.7944], [-8.4172, -7.8366, -7.5025]],
                ]
            )
        elif mobilevit_name == "deeplabv3_mobilevit_xxs":
            expected_logits = torch.tensor(
                [
                    [[6.9811, 6.9743, 7.3123], [7.1777, 7.1931, 7.3938], [7.5633, 7.8050, 7.8901]],
                    [[-10.5536, -10.2332, -10.2924], [-10.2336, -9.8624, -9.5964], [-10.8840, -10.8158, -10.6659]],
                    [[-3.4938, -3.0631, -2.8620], [-3.4205, -2.8135, -2.6875], [-3.4179, -2.7945, -2.8750]],
                ]
            )
        else:
            # 如果 mobilevit_name 不属于已知类型，则抛出 ValueError 异常
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        # 断言 logits 的部分数据与期望的 logits 非常接近，使用指定的容差
        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        # 如果 mobilevit_name 不以 "deeplabv3_" 开头，则断言 logits 的形状应为 (1, 1000)
        assert logits.shape == (1, 1000)

        # 根据不同的 mobilevit_name 设置期望的 logits
        if mobilevit_name == "mobilevit_s":
            expected_logits = torch.tensor([-0.9866, 0.2392, -1.1241])
        elif mobilevit_name == "mobilevit_xs":
            expected_logits = torch.tensor([-2.4761, -0.9399, -1.9587])
        elif mobilevit_name == "mobilevit_xxs":
            expected_logits = torch.tensor([-1.9364, -1.2327, -0.4653])
        else:
            # 如果 mobilevit_name 不属于已知类型，则抛出 ValueError 异常
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        # 断言 logits 的部分数据与期望的 logits 非常接近，使用指定的容差
        assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 创建一个目录，如果已存在则忽略错误
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印信息，保存模型到指定路径
    print(f"Saving model {mobilevit_name} to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印信息，保存图像处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
    # 如果需要推送到模型中心
    if push_to_hub:
        # 定义模型名称映射关系
        model_mapping = {
            "mobilevit_s": "mobilevit-small",
            "mobilevit_xs": "mobilevit-x-small",
            "mobilevit_xxs": "mobilevit-xx-small",
            "deeplabv3_mobilevit_s": "deeplabv3-mobilevit-small",
            "deeplabv3_mobilevit_xs": "deeplabv3-mobilevit-x-small",
            "deeplabv3_mobilevit_xxs": "deeplabv3-mobilevit-xx-small",
        }

        # 打印推送到模型中心的消息
        print("Pushing to the hub...")

        # 根据当前 mobilevit_name 获取对应的模型名称
        model_name = model_mapping[mobilevit_name]

        # 调用 image_processor 对象的 push_to_hub 方法，将模型推送到模型中心（组织为 "apple"）
        image_processor.push_to_hub(model_name, organization="apple")

        # 调用 model 对象的 push_to_hub 方法，将模型推送到模型中心（组织为 "apple"）
        model.push_to_hub(model_name, organization="apple")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需的参数
    parser.add_argument(
        "--mobilevit_name",
        default="mobilevit_s",
        type=str,
        help=(
            "Name of the MobileViT model you'd like to convert. Should be one of 'mobilevit_s', 'mobilevit_xs',"
            " 'mobilevit_xxs', 'deeplabv3_mobilevit_s', 'deeplabv3_mobilevit_xs', 'deeplabv3_mobilevit_xxs'."
        ),
    )
    # 添加命令行参数 `--mobilevit_name`，默认为 `"mobilevit_s"`，类型为字符串，用于指定要转换的 MobileViT 模型名称

    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    # 添加命令行参数 `--checkpoint_path`，必需参数，类型为字符串，用于指定原始状态字典文件（.pt 文件）的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加命令行参数 `--pytorch_dump_folder_path`，必需参数，类型为字符串，用于指定输出 PyTorch 模型的目录路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加命令行参数 `--push_to_hub`，如果存在则设置为 True，用于指定是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数并存储在 `args` 变量中

    convert_movilevit_checkpoint(
        args.mobilevit_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
    # 调用 `convert_movilevit_checkpoint` 函数，传递解析后的参数以执行模型转换操作
```