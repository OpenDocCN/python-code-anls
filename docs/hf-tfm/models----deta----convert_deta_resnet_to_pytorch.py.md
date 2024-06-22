# `.\models\deta\convert_deta_resnet_to_pytorch.py`

```
# coding=utf-8
# 版权声明，使用 Apache License 2.0 协议
# 导入必要的模块和库
import argparse   # 用于解析命令行参数
import json   # 用于处理 JSON 数据
from pathlib import Path   # 用于处理文件路径

import requests   # 用于发送 HTTP 请求，下载文件
import torch   # PyTorch 深度学习库
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url   # 用于从 Hugging Face 模型库下载和缓存模型
from PIL import Image   # Python 图像处理库

from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor   # 导入 DETA 相关类和函数
from transformers.utils import logging   # 导入日志模块

# 设置日志级别
logging.set_verbosity_info()
# 实例化 logger 对象
logger = logging.get_logger(__name__)

# 获取 DETA 的配置
def get_deta_config():
    # 初始化 DETA 配置对象
    config = DetaConfig(
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # 设置标签
    config.num_labels = 91
    # 设置模型所使用的标签对应的 JSON 文件路径
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    # 从 Hugging Face 模型库中缓存该文件
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    # 将标签字典中的键转换为整数类型
    id2label = {int(k): v for k, v in id2label.items()}
    # 设置 DETA 配置对象中的 id2label 和 label2id 属性
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config

# 这里列出了所有要重命名的键（原始名称在左边，我们的名称在右边）
def create_rename_keys(config):
    rename_keys = []

    # stem
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "model.backbone.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "model.backbone.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "model.backbone.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "model.backbone.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "model.backbone.model.embedder.embedder.normalization.running_var"))
    # stages
    # transformer encoder


该部分代码主要实现了以下功能：
- 导入所需的模块和库
- 设置日志级别和获取日志记录器
- 定义获取 DETA 配置的函数
- 定义重命名键的函数

其中，`get_deta_config` 函数用于读取和设置标签。它从 Hugging Face 模型库中下载缓存了一个 JSON 文件，将其中的标签信息加载到 DETA 配置对象中，并将标签字典的键的数据类型转换为整数类型。

`create_rename_keys` 函数用于创建一个列表，其中包含需要重命名的键。在这里，列出了要更改的一些模型权重的键名。
    # 对于配置中编码器层数的每一层，将键重命名并添加到重命名列表中
    for i in range(config.encoder_layers):
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        # 将权重的键从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        # 将偏置的键从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.weight", f"model.encoder.layers.{i}.self_attn.attention_weights.weight"))
        # 将注意力权重的键从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.bias", f"model.encoder.layers.{i}.self_attn.attention_weights.bias"))
        # 将注意力权重的偏置从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.weight", f"model.encoder.layers.{i}.self_attn.value_proj.weight"))
        # 将值投影的权重从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.bias", f"model.encoder.layers.{i}.self_attn.value_proj.bias"))
        # 将值投影的偏置从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.weight", f"model.encoder.layers.{i}.self_attn.output_proj.weight"))
        # 将输出投影的权重从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.bias", f"model.encoder.layers.{i}.self_attn.output_proj.bias"))
        # 将输出投影的偏置从transformer的编码器层i转换为对应的模型编码器层i
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.weight", f"model.encoder.layers.{i}.self_attn_layer_norm.weight"))
        # 将norm1的权重从transformer的编码器层i转换为对应的模型编码器层i的self_attn_layer_norm的权重
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"model.encoder.layers.{i}.self_attn_layer_norm.bias"))
        # 将norm1的偏置从transformer的编码器层i转换为对应的模型编码器层i的self_attn_layer_norm的偏置
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"model.encoder.layers.{i}.fc1.weight"))
        # 将linear1的权重从transformer的编码器层i转换为对应的模型编码器层i的fc1的权重
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"model.encoder.layers.{i}.fc1.bias"))
        # 将linear1的偏置从transformer的编码器层i转换为对应的模型编码器层i的fc1的偏置
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"model.encoder.layers.{i}.fc2.weight"))
        # 将linear2的权重从transformer的编码器层i转换为对应的模型编码器层i的fc2的权重
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"model.encoder.layers.{i}.fc2.bias"))
        # 将linear2的偏置从transformer的编码器层i转换为对应的模型编码器层i的fc2的偏置
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"model.encoder.layers.{i}.final_layer_norm.weight"))
        # 将norm2的权重从transformer的编码器层i转换为对应的模型编码器层i的final_layer_norm的权重
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"model.encoder.layers.{i}.final_layer_norm.bias"))
        # 将norm2的偏置从transformer的编码器层i转换为对应的模型编码器层i的final_layer_norm的偏置

    # transformer解码器
    # 循环遍历decoder_layers次，逐个处理层
    for i in range(config.decoder_layers):
        # 将表示转换参数的键值对添加到rename_keys列表中，用于重命名模型参数
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.weight", f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.bias", f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.weight", f"model.decoder.layers.{i}.encoder_attn.value_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.bias", f"model.decoder.layers.{i}.encoder_attn.value_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.weight", f"model.decoder.layers.{i}.encoder_attn.output_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.bias", f"model.decoder.layers.{i}.encoder_attn.output_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"model.decoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"model.decoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias"))
    
    # 返回重命名后的键值对列表
    return rename_keys
# 定义一个函数，用于重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键关联起来
    dct[new] = val


# 定义一个函数，用于读取模型的权重并将其导入到指定结构中
def read_in_decoder_q_k_v(state_dict, config):
    # 获取 Transformer 解码器的隐藏层大小
    hidden_size = config.d_model
    # 遍历 Transformer 解码器的每一层
    for i in range(config.decoder_layers):
        # 读取自注意力层的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将权重和偏置分配到查询、键和值的投影层中
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]


# 定义一个函数，用于准备图像数据
# 我们将在一组可爱的猫的图像上验证我们的结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 获取图片的原始流数据，并使用 PIL 打开图片
    im = Image.open(requests.get(url, stream=True).raw)

    return im


# 使用 torch.no_grad() 上下文管理器，确保在模型推理时不进行梯度计算
@torch.no_grad()
# 定义一个函数，用于将 Delta 模型的权重转换为 DETA 结构
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETA structure.
    """

    # 加载配置信息
    config = get_deta_config()

    # 加载原始的状态字典
    if model_name == "deta-resnet-50":
        filename = "adet_checkpoint0011.pth"
    elif model_name == "deta-resnet-50-24-epochs":
        filename = "adet_2x_checkpoint0023.pth"
    else:
        raise ValueError(f"Model name {model_name} not supported")
    # 下载模型检查点文件并加载状态字典
    checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename=filename)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取自注意力层的查询、键和值
    read_in_decoder_q_k_v(state_dict, config)

    # 修正一些前缀
    for key in state_dict.copy().keys():
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        if "input_proj" in key:
            val = state_dict.pop(key)
            state_dict["model." + key] = val
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer", "model")] = val
    # 最后，创建 HuggingFace 模型并加载状态字典
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 如果 GPU 可用，则将模型移到 GPU 上，否则移到 CPU 上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 加载图像处理器
    processor = DetaImageProcessor(format="coco_detection")
    
    # 检查图像处理器是否正常工作
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))
    
    # 检查输出的逻辑值（logits）
    if model_name == "deta-resnet-50":
        expected_logits = torch.tensor(
            [[-7.3978, -2.5406, -4.1668], [-8.2684, -3.9933, -3.8096], [-7.0515, -3.7973, -5.8516]]
        )
        expected_boxes = torch.tensor([[0.5043, 0.4973, 0.9998], [0.2542, 0.5489, 0.4748], [0.5490, 0.2765, 0.0570]])
    elif model_name == "deta-resnet-50-24-epochs":
        expected_logits = torch.tensor(
            [[-7.1688, -2.4857, -4.8669], [-7.8630, -3.8154, -4.2674], [-7.2730, -4.1865, -5.5323]]
        )
        expected_boxes = torch.tensor([[0.5021, 0.4971, 0.9994], [0.2546, 0.5486, 0.4731], [0.1686, 0.1986, 0.2142]])
    
    # 检查模型的输出逻辑值和边界框是否与预期值相符
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    print("Everything ok!")
    
    # 如果指定了 pytorch_dump_folder_path，则保存模型和处理器
    if pytorch_dump_folder_path:
        logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    
    # 如果指定了 push_to_hub，则将模型和处理器推送到 Hub
    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(f"jozhang97/{model_name}")
        processor.push_to_hub(f"jozhang97/{model_name}")
# 如果该脚本作为独立运行的程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个参数，用于指定模型的名称
    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-resnet-50",
        choices=["deta-resnet-50", "deta-resnet-50-24-epochs"],
        help="Name of the model you'd like to convert.",
    )
    # 添加一个参数，用于指定导出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    # 添加一个参数，用于指定是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入参数模型名称、PyTorch 模型输出路径、是否推送到 hub
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```