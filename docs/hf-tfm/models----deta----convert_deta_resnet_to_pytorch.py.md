# `.\models\deta\convert_deta_resnet_to_pytorch.py`

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
"""Convert DETA checkpoints from the original repository.

URL: https://github.com/jozhang97/DETA/tree/master"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image

from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor
from transformers.utils import logging

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义获取DETA配置信息的函数
def get_deta_config():
    # 创建DETA配置对象，设置各种参数
    config = DetaConfig(
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # 设置标签信息
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    # 从Hugging Face Hub下载并加载COCO检测标签映射文件
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    # 转换为整数类型的键值对字典
    id2label = {int(k): v for k, v in id2label.items()}
    # 设置DETA配置对象的id到标签的映射和标签到id的映射
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config

# 定义创建重命名键列表的函数
# 这里列出所有需要重命名的键（左侧是原始名称，右侧是我们的名称）
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

    # fmt: on
    # 遍历编码器层数量次数，进行以下操作
    for i in range(config.encoder_layers):
        # 添加重命名键值对，将transformer.encoder.layers中的权重和偏置重命名为model.encoder.layers中对应的权重和偏置
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.weight", f"model.encoder.layers.{i}.self_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.bias", f"model.encoder.layers.{i}.self_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.weight", f"model.encoder.layers.{i}.self_attn.value_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.bias", f"model.encoder.layers.{i}.self_attn.value_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.weight", f"model.encoder.layers.{i}.self_attn.output_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.bias", f"model.encoder.layers.{i}.self_attn.output_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.weight", f"model.encoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"model.encoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"model.encoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"model.encoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"model.encoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"model.encoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"model.encoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"model.encoder.layers.{i}.final_layer_norm.bias"))

    # transformer decoder
    # 遍历从配置中获取的解码器层数量次数
    for i in range(config.decoder_layers):
        # 重命名键，将transformer.decoder.layers中的权重和偏置名映射到model.decoder.layers中的对应位置
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

    # 格式化代码，结束长行格式化
    # fmt: on

    # 返回重命名后的键列表
    return rename_keys
# 重命名字典中的键名，将旧键名对应的值移除，并将其值存储到新键名下
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 读取解码器的查询、键和值的权重和偏置，将它们添加到状态字典中
def read_in_decoder_q_k_v(state_dict, config):
    # 获取隐藏层大小
    hidden_size = config.d_model
    # 遍历解码器层
    for i in range(config.decoder_layers):
        # 获取自注意力层输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询、键和值的投影权重和偏置添加到状态字典中
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size:2*hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size:2*hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]

# 准备图片数据，从指定的 URL 中获取图片并返回
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

# 使用 torch.no_grad() 装饰器，将模型权重转换并复制到 DETA 结构中
@torch.no_grad()
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    复制/粘贴/调整模型权重到我们的 DETA 结构中。
    """

    # 加载配置信息
    config = get_deta_config()

    # 加载原始状态字典
    if model_name == "deta-resnet-50":
        filename = "adet_checkpoint0011.pth"
    elif model_name == "deta-resnet-50-24-epochs":
        filename = "adet_2x_checkpoint0023.pth"
    else:
        raise ValueError(f"Model name {model_name} not supported")
    # 从指定的 HF Hub 下载检查点文件
    checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename=filename)
    # 使用 torch 加载模型状态字典，设定在 CPU 上处理
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取解码器的查询、键和值的权重和偏置，将它们添加到状态字典中
    read_in_decoder_q_k_v(state_dict, config)

    # 修正部分前缀
    for key in state_dict.copy().keys():
        # 如果键中包含 "transformer.decoder.class_embed" 或 "transformer.decoder.bbox_embed"
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            # 替换键名前缀为 "model.decoder"
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        # 如果键中包含 "input_proj"
        if "input_proj" in key:
            val = state_dict.pop(key)
            # 替换键名前缀为 "model."
            state_dict["model." + key] = val
        # 如果键中包含 "level_embed"、"pos_trans"、"pix_trans" 或 "enc_output"
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            # 替换键名前缀为 "model"
            state_dict[key.replace("transformer", "model")] = val
    # 创建一个用于物体检测的 Deta 模型，并加载状态字典
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 检测是否可以使用 CUDA，如果可以则将模型移动到 CUDA 设备上，否则使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 加载图像处理器，使用 COCO 检测格式
    processor = DetaImageProcessor(format="coco_detection")

    # 准备图像并进行编码
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))

    # 验证模型输出的 logits 是否符合预期
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

    # 使用 assert 语句检查模型输出的 logits 和预期值的接近程度，设置容差为 1e-4
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    print("Everything ok!")

    # 如果指定了 PyTorch 模型保存路径，则保存模型和处理器
    if pytorch_dump_folder_path:
        logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型和处理器推送到 Hub
    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(f"jozhang97/{model_name}")
        processor.push_to_hub(f"jozhang97/{model_name}")
# 如果这个脚本被直接运行，执行以下操作
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个参数选项，用于指定模型名称
    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-resnet-50",
        choices=["deta-resnet-50", "deta-resnet-50-24-epochs"],
        help="Name of the model you'd like to convert.",
    )

    # 添加一个参数选项，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )

    # 添加一个布尔类型的参数选项，用于指定是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_deta_checkpoint，传入解析后的参数
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```