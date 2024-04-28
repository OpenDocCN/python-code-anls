# `.\transformers\models\mobilevit\convert_mlcvnets_to_pytorch.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 引入必要的库
import argparse  # 解析命令行参数
import json  # 处理json格式数据
from pathlib import Path  # 处理文件路径

import requests  # 发送http请求
import torch  # 机器学习框架
from huggingface_hub import hf_hub_download  # 从huggingface hub下载模型
from PIL import Image  # Python图像处理库

# 引入transformers库中的相关模块
from transformers import (
    MobileViTConfig,  # 移动视觉Transformer的配置文件
    MobileViTForImageClassification,  # 用于图像分类的移动视觉Transformer
    MobileViTForSemanticSegmentation,  # 用于语义分割的移动视觉Transformer
    MobileViTImageProcessor,  # 移动视觉Transformer的图像处理器
)
from transformers.utils import logging  # 日志记录

# 设置日志输出级别为info
logging.set_verbosity_info()
# 获取logger对象
logger = logging.get_logger(__name__)


# 定义一个函数，用于获取MobileViT配置
def get_mobilevit_config(mobilevit_name):
    # 创建MobileViTConfig对象
    config = MobileViTConfig()

    # 根据模型名称设置不同的隐藏层大小和neck隐藏层大小
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

    # 如果模型名称以"deeplabv3_"开头
    if mobilevit_name.startswith("deeplabv3_"):
        # 设置图片大小、输出步幅和标签数
        config.image_size = 512
        config.output_stride = 16
        config.num_labels = 21
        filename = "pascal-voc-id2label.json"
    else:
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"

    # 从huggingface hub下载模型的标签文件
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# 定义一个函数，重命名键名
def rename_key(name, base_model=False):
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
    # 如果文件名中包含 ".norm."，则替换为 ".normalization."
    if ".norm." in name:
        name = name.replace(".norm.", ".normalization.")
    # 如果文件名中包含 ".conv."，则替换为 ".convolution."
    if ".conv." in name:
        name = name.replace(".conv.", ".convolution.")
    # 如果文件名中包含 ".conv_proj."，则替换为 ".conv_projection."
    if ".conv_proj." in name:
        name = name.replace(".conv_proj.", ".conv_projection.")

    # 遍历 0 到 1 的范围
    for i in range(0, 2):
        # 遍历 0 到 3 的范围
        for j in range(0, 4):
            # 如果文件名中包含 ".i.j."，则替换为 ".i.layer.j."
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.layer.{j}.")

    # 遍历 2 到 5 的范围
    for i in range(2, 6):
        # 遍历 0 到 3 的范围
        for j in range(0, 4):
            # 如果文件名中包含 ".i.j."，则替换为 ".i."
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.")
                # 如果文件名中包含 "expand_1x1"，则替换为 "downsampling_layer.expand_1x1"
                if "expand_1x1" in name:
                    name = name.replace("expand_1x1", "downsampling_layer.expand_1x1")
                # 如果文件名中包含 "conv_3x3"，则替换为 "downsampling_layer.conv_3x3"
                if "conv_3x3" in name:
                    name = name.replace("conv_3x3", "downsampling_layer.conv_3x3")
                # 如果文件名中包含 "reduce_1x1"，则替换为 "downsampling_layer.reduce_1x1"
                if "reduce_1x1" in name:
                    name = name.replace("reduce_1x1", "downsampling_layer.reduce_1x1")

    # 遍历 2 到 4 的范围
    for i in range(2, 5):
        # 如果文件名中包含 ".global_rep.i.weight"，则替换为 ".layernorm.weight"
        if f".global_rep.{i}.weight" in name:
            name = name.replace(f".global_rep.{i}.weight", ".layernorm.weight")
        # 如果文件名中包含 ".global_rep.i.bias"，则替换为 ".layernorm.bias"
        if f".global_rep.{i}.bias" in name:
            name = name.replace(f".global_rep.{i}.bias", ".layernorm.bias")

    # 如果文件名中包含 ".global_rep."，则替换为 ".transformer."
    if ".global_rep." in name:
        name = name.replace(".global_rep.", ".transformer.")
    # 如果文件名中包含 ".pre_norm_mha.0."，则替换为 ".layernorm_before."
    if ".pre_norm_mha.0." in name:
        name = name.replace(".pre_norm_mha.0.", ".layernorm_before.")
    # 如果文件名中包含 ".pre_norm_mha.1.out_proj."，则替换为 ".attention.output.dense."
    if ".pre_norm_mha.1.out_proj." in name:
        name = name.replace(".pre_norm_mha.1.out_proj.", ".attention.output.dense.")
    # 如果文件名中包含 ".pre_norm_ffn.0."，则替换为 ".layernorm_after."
    if ".pre_norm_ffn.0." in name:
        name = name.replace(".pre_norm_ffn.0.", ".layernorm_after.")
    # 如果文件名中包含 ".pre_norm_ffn.1."，则替换为 ".intermediate.dense."
    if ".pre_norm_ffn.1." in name:
        name = name.replace(".pre_norm_ffn.1.", ".intermediate.dense.")
    # 如果文件名中包含 ".pre_norm_ffn.4."，则替换为 ".output.dense."
    if ".pre_norm_ffn.4." in name:
        name = name.replace(".pre_norm_ffn.4.", ".output.dense.")
    # 如果文件名中包含 ".transformer."，则替换为 ".transformer.layer."
    if ".transformer." in name:
        name = name.replace(".transformer.", ".transformer.layer.")

    # 如果文件名中包含 ".aspp_layer."，则替换为 "."
    if ".aspp_layer." in name:
        name = name.replace(".aspp_layer.", ".")
    # 如果文件名中包含 ".aspp_pool."，则替换为 "."
    if ".aspp_pool." in name:
        name = name.replace(".aspp_pool.", ".")
    # 如果文件名中包含 "seg_head."，则替换为 "segmentation_head."
    if "seg_head." in name:
        name = name.replace("seg_head.", "segmentation_head.")
    # 如果文件名中包含 "segmentation_head.classifier.classifier."，则替换为 "segmentation_head.classifier."
    if "segmentation_head.classifier.classifier." in name:
        name = name.replace("segmentation_head.classifier.classifier.", "segmentation_head.classifier.")

    # 如果文件名中包含 "classifier.fc."，则替换为 "classifier."
    if "classifier.fc." in name:
        name = name.replace("classifier.fc.", "classifier.")
    # 如果不是基础模型且文件名中不包含 "segmentation_head."，则在文件名前添加 "mobilevit."
    elif (not base_model) and ("segmentation_head." not in name):
        name = "mobilevit." + name

    # 返回处理后的文件名
    return name
# 将原始状态字典转换为模型的状态字典
def convert_state_dict(orig_state_dict, model, base_model=False):
    # 如果是基础模型，则前缀为空，否则前缀为"mobilevit."
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevit."

    # 遍历原始状态字典的键
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        # 如果键以"encoder."开头，则去掉前缀
        if key[:8] == "encoder.":
            key = key[8:]

        # 如果键包含"qkv"，则将值分解为query、key和value的权重和偏置
        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[0][6:]) - 1
            transformer_num = int(key_split[3])
            layer = model.get_submodule(f"{model_prefix}encoder.layer.{layer_num}")
            dim = layer.transformer.layer[transformer_num].attention.attention.all_head_size
            prefix = (
                f"{model_prefix}encoder.layer.{layer_num}.transformer.layer.{transformer_num}.attention.attention."
            )
            if "weight" in key:
                orig_state_dict[prefix + "query.weight"] = val[:dim, :]
                orig_state_dict[prefix + "key.weight"] = val[dim : dim * 2, :]
                orig_state_dict[prefix + "value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[prefix + "query.bias"] = val[:dim]
                orig_state_dict[prefix + "key.bias"] = val[dim : dim * 2]
                orig_state_dict[prefix + "value.bias"] = val[-dim:]
        else:
            # 如果键不包含"qkv"，则直接将键重命名后存入原始状态字典
            orig_state_dict[rename_key(key, base_model)] = val

    return orig_state_dict


# 准备一张可爱猫咪图像
def prepare_img():
    # 从 URL 下载图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 转换 MobileViT 检查点
@torch.no_grad()
def convert_movilevit_checkpoint(mobilevit_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    将模型权重复制/粘贴/微调到我们的 MobileViT 结构中。
    """
    # 获取 MobileViT 配置
    config = get_mobilevit_config(mobilevit_name)

    # 加载原始状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 加载 🤗 模型
    if mobilevit_name.startswith("deeplabv3_"):
        model = MobileViTForSemanticSegmentation(config).eval()
    else:
        model = MobileViTForImageClassification(config).eval()

    # 将原始状态字典转换为模型状态字典
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 在准备好的图像上检查输出
    image_processor = MobileViTImageProcessor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    # 检查模型名称是否以"deeplabv3_"开头
    if mobilevit_name.startswith("deeplabv3_"):
        # 断言模型输出的形状是否符合预期
        assert logits.shape == (1, 21, 32, 32)

        # 根据不同的模型名称，设置期望的输出值
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
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        # 断言模型输出的部分是否与预期值接近
        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        # 断言模型输出的形状是否符合预期
        assert logits.shape == (1, 1000)

        # 根据不同的模型名称，设置期望的输出值
        if mobilevit_name == "mobilevit_s":
            expected_logits = torch.tensor([-0.9866, 0.2392, -1.1241])
        elif mobilevit_name == "mobilevit_xs":
            expected_logits = torch.tensor([-2.4761, -0.9399, -1.9587])
        elif mobilevit_name == "mobilevit_xxs":
            expected_logits = torch.tensor([-1.9364, -1.2327, -0.4653])
        else:
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        # 断言模型输出的部分是否与预期值接近
        assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 确保模型输出文件夹存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息
    print(f"Saving model {mobilevit_name} to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
    # 如果需要推送到hub
    if push_to_hub:
        # 设置模型映射关系
        model_mapping = {
            "mobilevit_s": "mobilevit-small",
            "mobilevit_xs": "mobilevit-x-small",
            "mobilevit_xxs": "mobilevit-xx-small",
            "deeplabv3_mobilevit_s": "deeplabv3-mobilevit-small",
            "deeplabv3_mobilevit_xs": "deeplabv3-mobilevit-x-small",
            "deeplabv3_mobilevit_xxs": "deeplabv3-mobilevit-xx-small",
        }
    
        # 打印提示信息
        print("Pushing to the hub...")
        # 根据mobilevit_name在模型映射关系中获取模型名称
        model_name = model_mapping[mobilevit_name]
        # 将图像处理器推送到hub
        image_processor.push_to_hub(model_name, organization="apple")
        # 将模型推送到hub
        model.push_to_hub(model_name, organization="apple")
# 如果当前脚本被作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加必要参数
    parser.add_argument(
        "--mobilevit_name",
        default="mobilevit_s",
        type=str,
        help=(
            "Name of the MobileViT model you'd like to convert. Should be one of 'mobilevit_s', 'mobilevit_xs',"
            " 'mobilevit_xxs', 'deeplabv3_mobilevit_s', 'deeplabv3_mobilevit_xs', 'deeplabv3_mobilevit_xxs'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将移动视觉（MobileViT）检查点转换为 PyTorch 模型
    convert_movilevit_checkpoint(
        args.mobilevit_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```