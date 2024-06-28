# `.\models\yolos\convert_yolos_to_pytorch.py`

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
"""Convert YOLOS checkpoints from the original repository. URL: https://github.com/hustvl/YOLOS"""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式数据
from pathlib import Path  # 用于处理文件和目录路径操作

import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 用于从 HF Hub 下载资源
from PIL import Image  # Python Imaging Library，用于图像处理

from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor  # YOLOS 模型相关类
from transformers.utils import logging  # Transformers 日志工具模块


logging.set_verbosity_info()  # 设置日志输出级别为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_yolos_config(yolos_name: str) -> YolosConfig:
    config = YolosConfig()

    # 根据模型名称设置 YOLOS 配置参数
    if "yolos_ti" in yolos_name:
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
        config.image_size = [800, 1333]
        config.use_mid_position_embeddings = False
    elif yolos_name == "yolos_s_dWr":
        config.hidden_size = 330
        config.num_hidden_layers = 14
        config.num_attention_heads = 6
        config.intermediate_size = 1320
    elif "yolos_s" in yolos_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif "yolos_b" in yolos_name:
        config.image_size = [800, 1344]

    config.num_labels = 91  # 设置标签数为 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    # 从 HF Hub 下载 COCO 检测标签映射文件，并加载为字典形式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 将键转换为整数类型
    config.id2label = id2label  # 设置 id 到标签的映射字典
    config.label2id = {v: k for k, v in id2label.items()}  # 设置标签到 id 的映射字典

    return config


# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict: dict, config: YolosConfig, base_model: bool = False):
    # 循环遍历隐藏层的数量，通常用于处理神经网络的层数
    for i in range(config.num_hidden_layers):
        # 弹出输入投影层的权重和偏置，这些在Timm中表示为单矩阵和偏置
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 将查询(query)、键(keys)和值(values)依次添加到状态字典中
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]
# 定义一个函数，用于重命名模型状态字典的键
def rename_key(name: str) -> str:
    if "backbone" in name:
        name = name.replace("backbone", "vit")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "det_token" in name:
        name = name.replace("det_token", "embeddings.detection_tokens")
    if "mid_pos_embed" in name:
        name = name.replace("mid_pos_embed", "encoder.mid_position_embeddings")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
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
    if "class_embed" in name:
        name = name.replace("class_embed", "class_labels_classifier")
    if "bbox_embed" in name:
        name = name.replace("bbox_embed", "bbox_predictor")
    if "vit.norm" in name:
        name = name.replace("vit.norm", "vit.layernorm")

    return name


# 定义一个函数，用于将原始的模型状态字典转换为新的模型状态字典格式
def convert_state_dict(orig_state_dict: dict, model: YolosForObjectDetection) -> dict:
    # 遍历原始状态字典的键（复制一个副本进行遍历）
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)  # 弹出当前键对应的值

        # 如果键包含 "qkv"
        if "qkv" in key:
            key_split = key.split(".")  # 使用点号分割键名
            layer_num = int(key_split[2])  # 解析层号
            dim = model.vit.encoder.layer[layer_num].attention.attention.all_head_size  # 获取维度信息

            # 根据键名中是否包含 "weight" 来决定如何处理值
            if "weight" in key:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[dim:dim * 2, :]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[dim:dim * 2]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]

        else:
            # 对键名进行重命名处理并更新状态字典
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# 定义一个函数，用于从 URL 加载并返回一张图像
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)  # 使用 requests 获取图像流并打开为 Image 对象
    return im


# 使用 torch.no_grad() 装饰器，确保在此函数调用期间不会计算梯度
@torch.no_grad()
# 定义一个函数，用于将指定模型的权重转换到 YOLOS 结构中
def convert_yolos_checkpoint(
    yolos_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    复制/粘贴/调整模型的权重到我们的 YOLOS 结构中。
    """
    # 获取 YOLOS 配置信息
    config = get_yolos_config(yolos_name)

    # 加载原始模型的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 加载 YOLOS 目标检测模型
    model = YolosForObjectDetection(config)
    model.eval()

    # 将原始模型的状态字典转换为适应 YOLOS 结构的新状态字典
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 使用 YolosImageProcessor 准备图像，然后将其编码
    size = 800 if yolos_name != "yolos_ti" else 512
    image_processor = YolosImageProcessor(format="coco_detection", size=size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")

    # 在处理后的图像上运行模型，获取预测的 logits 和边界框
    outputs = model(**encoding)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes

    # 根据 yolos_name 设置预期的 logits 和边界框值
    expected_slice_logits, expected_slice_boxes = None, None
    if yolos_name == "yolos_ti":
        expected_slice_logits = torch.tensor(
            [[-39.5022, -11.9820, -17.6888], [-29.9574, -9.9769, -17.7691], [-42.3281, -20.7200, -30.6294]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4021, 0.0836, 0.7979], [0.0184, 0.2609, 0.0364], [0.1781, 0.2004, 0.2095]]
        )
    elif yolos_name == "yolos_s_200_pre":
        expected_slice_logits = torch.tensor(
            [[-24.0248, -10.3024, -14.8290], [-42.0392, -16.8200, -27.4334], [-27.2743, -11.8154, -18.7148]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]]
        )
    elif yolos_name == "yolos_s_300_pre":
        expected_slice_logits = torch.tensor(
            [[-36.2220, -14.4385, -23.5457], [-35.6970, -14.7583, -21.3935], [-31.5939, -13.6042, -16.8049]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7614, 0.2316, 0.4728], [0.7168, 0.4495, 0.3855], [0.4996, 0.1466, 0.9996]]
        )
    elif yolos_name == "yolos_s_dWr":
        expected_slice_logits = torch.tensor(
            [[-42.8668, -24.1049, -41.1690], [-34.7456, -14.1274, -24.9194], [-33.7898, -12.1946, -25.6495]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5587, 0.2773, 0.0605], [0.5004, 0.3014, 0.9994], [0.4999, 0.1548, 0.9994]]
        )
    elif yolos_name == "yolos_base":
        expected_slice_logits = torch.tensor(
            [[-40.6064, -24.3084, -32.6447], [-55.1990, -30.7719, -35.5877], [-51.4311, -33.3507, -35.6462]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5555, 0.2794, 0.0655], [0.9049, 0.2664, 0.1894], [0.9183, 0.1984, 0.1635]]
        )
    else:
        # 如果 yolos_name 不在预期的列表中，则抛出 ValueError
        raise ValueError(f"Unknown yolos_name: {yolos_name}")

    # 断言确保预测的 logits 和边界框与预期值接近
    assert torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)
    # 创建指定路径的文件夹，如果文件夹不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息，包括模型名称和保存路径
    print(f"Saving model {yolos_name} to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息，包括保存路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 定义模型名称到Hub名称的映射字典
        model_mapping = {
            "yolos_ti": "yolos-tiny",
            "yolos_s_200_pre": "yolos-small",
            "yolos_s_300_pre": "yolos-small-300",
            "yolos_s_dWr": "yolos-small-dwr",
            "yolos_base": "yolos-base",
        }

        # 打印提示信息，说明正在将模型推送到Hub
        print("Pushing to the hub...")
        # 根据模型名称从映射字典中获取对应的Hub名称
        model_name = model_mapping[yolos_name]
        # 将图像处理器推送到Hub，指定组织名称为"hustvl"
        image_processor.push_to_hub(model_name, organization="hustvl")
        # 将模型推送到Hub，指定组织名称为"hustvl"
        model.push_to_hub(model_name, organization="hustvl")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--yolos_name",
        default="yolos_s_200_pre",
        type=str,
        help=(
            "Name of the YOLOS model you'd like to convert. Should be one of 'yolos_ti', 'yolos_s_200_pre',"
            " 'yolos_s_300_pre', 'yolos_s_dWr', 'yolos_base'."
        ),
    )
    # 添加一个参数选项，用于指定要转换的 YOLOS 模型的名称

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original state dict (.pth file)."
    )
    # 添加一个参数选项，用于指定原始状态字典文件（.pth 文件）的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数选项，用于指定输出 PyTorch 模型目录的路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个参数选项，用于指定是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_yolos_checkpoint(args.yolos_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_yolos_checkpoint，传递解析后的参数作为函数的参数
```