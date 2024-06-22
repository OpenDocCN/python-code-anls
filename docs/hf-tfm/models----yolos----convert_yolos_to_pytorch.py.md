# `.\transformers\models\yolos\convert_yolos_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权信息
#
# 根据 Apache 许可证，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非有适用法律要求或书面同意，否则依“原样”提供软件
# 没有任何种类的明示或暗示的担保或条件
# 请查看许可证以了解特定语言规定的权限和限制
"""从原始仓库转换 YOLOS 的检查点。URL: https://github.com/hustvl/YOLOS"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor
from transformers.utils import logging

# 设置日志记录等级到 'info'
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 获取 YolosConfig 配置
def get_yolos_config(yolos_name: str) -> YolosConfig:
    config = YolosConfig()

    # 根据 yolos_name 设置架构大小
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

    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    # 从 HF Hub 下载标签文件
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config

# 将每个编码器层的矩阵分成查询、键和值
def read_in_q_k_v(state_dict: dict, config: YolosConfig, base_model: bool = False):
    # 遍历隐藏层的数量
    for i in range(config.num_hidden_layers):
        # 从状态字典中弹出输入投影层的权重和偏置项（在timm中，这是一个单独的矩阵加偏置项）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 接下来，将查询、键和值（按顺序）添加到状态字典中
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
# 定义函数，将给定的键名进行重命名并返回
def rename_key(name: str) -> str:
    # 如果键名中包含"backbone"，则替换为"vit"
    if "backbone" in name:
        name = name.replace("backbone", "vit")
    # 如果键名中包含"cls_token"，则替换为"embeddings.cls_token"
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    # ... 其他类似的替换规则 ...


# 定义函数，将原始的状态字典转换为新的状态字典
def convert_state_dict(orig_state_dict: dict, model: YolosForObjectDetection) -> dict:
    # 遍历原始状态字典的拷贝的键名列表
    for key in orig_state_dict.copy().keys():
        # 弹出当前键名对应的值
        val = orig_state_dict.pop(key)
        # 如果键名中包含"qkv"
        if "qkv" in key:
            # 对键名进行拆分，获取层数和维度信息
            key_split = key.split(".")
            layer_num = int(key_split[2])
            dim = model.vit.encoder.layer[layer_num].attention.attention.all_head_size
            # 如果键名中包含"weight"
            if "weight" in key:
                # 将值赋给新的键名
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                # ... 其他类似的赋值操作 ...
            else:
                # 将值赋给新的键名
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                # ... 其他类似的赋值操作 ...
        else:
            # 调用重命名函数，并将新的键值对加入到状态字典中
            orig_state_dict[rename_key(key)] = val
    # 返回新的状态字典
    return orig_state_dict


# 准备图像数据，并返回对应的张量
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 标记接下来的代码部分不需要进行 Torch 的梯度计算
@torch.no_grad()
def convert_yolos_checkpoint(
    yolos_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    """
    # 根据给定的 YOLOS 名称获取配置信息
    config = get_yolos_config(yolos_name)

    # 加载原始的 state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 加载 🤗 模型
    model = YolosForObjectDetection(config)
    model.eval()
    # 将原始 state_dict 转换为适合 YOLOS 结构的新 state_dict
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 在由 YolosImageProcessor 准备的图像上检查输出
    size = 800 if yolos_name != "yolos_ti" else 512
    image_processor = YolosImageProcessor(format="coco_detection", size=size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes

    expected_slice_logits, expected_slice_boxes = None, None
    if yolos_name == "yolos_ti":
        # 针对 yolos_ti，预期的输出切片 logits 和 boxes
        expected_slice_logits = torch.tensor(
            [[-39.5022, -11.9820, -17.6888], [-29.9574, -9.9769, -17.7691], [-42.3281, -20.7200, -30.6294]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4021, 0.0836, 0.7979], [0.0184, 0.2609, 0.0364], [0.1781, 0.2004, 0.2095]]
        )
    elif yolos_name == "yolos_s_200_pre":
        # 针对 yolos_s_200_pre，预期的输出切片 logits 和 boxes
        expected_slice_logits = torch.tensor(
            [[-24.0248, -10.3024, -14.8290], [-42.0392, -16.8200, -27.4334], [-27.2743, -11.8154, -18.7148]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]]
        )
    elif yolos_name == "yolos_s_300_pre":
        # 针对 yolos_s_300_pre，预期的输出切片 logits 和 boxes
        expected_slice_logits = torch.tensor(
            [[-36.2220, -14.4385, -23.5457], [-35.6970, -14.7583, -21.3935], [-31.5939, -13.6042, -16.8049]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7614, 0.2316, 0.4728], [0.7168, 0.4495, 0.3855], [0.4996, 0.1466, 0.9996]]
        )
    elif yolos_name == "yolos_s_dWr":
        # 针对 yolos_s_dWr，预期的输出切片 logits 和 boxes
        expected_slice_logits = torch.tensor(
            [[-42.8668, -24.1049, -41.1690], [-34.7456, -14.1274, -24.9194], [-33.7898, -12.1946, -25.6495]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5587, 0.2773, 0.0605], [0.5004, 0.3014, 0.9994], [0.4999, 0.1548, 0.9994]]
        )
    elif yolos_name == "yolos_base":
        # 针对 yolos_base，预期的输出切片 logits 和 boxes
        expected_slice_logits = torch.tensor(
            [[-40.6064, -24.3084, -32.6447], [-55.1990, -30.7719, -35.5877], [-51.4311, -33.3507, -35.6462]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5555, 0.2794, 0.0655], [0.9049, 0.2664, 0.1894], [0.9183, 0.1984, 0.1635]]
        )
    else:
        # 如果给定的 yolos_name 不在已知列表中，抛出 ValueError
        raise ValueError(f"Unknown yolos_name: {yolos_name}")

    # 使用 assert 检查模型输出是否与预期输出接近
    assert torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)
    # 创建文件夹，如果文件夹不存在则创建，存在则忽略
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息
    print(f"Saving model {yolos_name} to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果要推送到 hub
    if push_to_hub:
        # 定义模型名称映射关系
        model_mapping = {
            "yolos_ti": "yolos-tiny",
            "yolos_s_200_pre": "yolos-small",
            "yolos_s_300_pre": "yolos-small-300",
            "yolos_s_dWr": "yolos-small-dwr",
            "yolos_base": "yolos-base",
        }

        # 打印推送到 hub 的信息
        print("Pushing to the hub...")
        # 获取模型对应的名称
        model_name = model_mapping[yolos_name]
        # 将图像处理器推送到 hub
        image_processor.push_to_hub(model_name, organization="hustvl")
        # 将模型推送到 hub
        model.push_to_hub(model_name, organization="hustvl")
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--yolos_name",  # YOLOS 模型的名称
        default="yolos_s_200_pre",  # 默认值为 'yolos_s_200_pre'
        type=str,  # 参数类型为字符串
        help=(  # 参数的帮助文本
            "Name of the YOLOS model you'd like to convert. Should be one of 'yolos_ti', 'yolos_s_200_pre',"
            " 'yolos_s_300_pre', 'yolos_s_dWr', 'yolos_base'."
        ),
    )
    # 添加参数：原始状态字典的路径
    parser.add_argument(
        "--checkpoint_path",  # 原始状态字典的路径
        default=None,  # 默认值为 None
        type=str,  # 参数类型为字符串
        help="Path to the original state dict (.pth file).",  # 参数的帮助文本
    )
    # 添加参数：输出 PyTorch 模型的目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 输出 PyTorch 模型的目录路径
        default=None,  # 默认值为 None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory.",  # 参数的帮助文本
    )
    # 添加参数：是否将转换后的模型推送到🤗 hub
    parser.add_argument(
        "--push_to_hub",  # 是否将转换后的模型推送到🤗 hub
        action="store_true",  # 设置为 True 表示执行该操作
        help="Whether or not to push the converted model to the 🤗 hub.",  # 参数的帮助文本
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，执行 YOLOS 模型的状态字典转换
    convert_yolos_checkpoint(args.yolos_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
```