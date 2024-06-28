# `.\models\bros\convert_bros_to_pytorch.py`

```py
# 设置脚本的编码格式为 UTF-8
# 版权声明，指明版权归属于 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”提供本软件，不提供任何形式的明示或暗示保证或条件。
# 请参阅许可证获取特定语言的权限和限制。
"""将 Bros 检查点转换为 HuggingFace 模型格式"""

import argparse  # 导入命令行参数解析模块

import bros  # 原始仓库
import torch  # 导入 PyTorch 模块

from transformers import BrosConfig, BrosModel, BrosProcessor  # 导入转换所需的模块和类
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志记录的详细级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_configs(model_name):
    """获取指定模型的配置信息"""
    bros_config = BrosConfig.from_pretrained(model_name)
    return bros_config


def remove_ignore_keys_(state_dict):
    """移除指定的忽略键（如果存在）"""
    ignore_keys = [
        "embeddings.bbox_sinusoid_emb.inv_freq",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    """根据约定重命名给定的键"""
    if name == "embeddings.bbox_projection.weight":
        name = "bbox_embeddings.bbox_projection.weight"

    if name == "embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq"

    if name == "embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq"

    return name


def convert_state_dict(orig_state_dict, model):
    """将原始模型状态字典转换为适用于 HuggingFace 模型的格式"""
    # 重命名键
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    # 移除忽略的键
    remove_ignore_keys_(orig_state_dict)

    return orig_state_dict


def convert_bros_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """将 Bros 模型检查点转换为 HuggingFace 模型格式"""
    # 加载原始的 Bros 模型
    original_model = bros.BrosModel.from_pretrained(model_name).eval()

    # 加载 HuggingFace 模型
    bros_config = get_configs(model_name)
    model = BrosModel.from_pretrained(model_name, config=bros_config)
    model.eval()

    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 验证结果

    # 原始的 BROS 模型需要每个边界框 4 个点（8 个浮点数），准备形状为 [batch_size, seq_len, 8] 的边界框
    # 创建一个包含边界框信息的张量，用于定义对象的位置和大小
    bbox = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.4396, 0.6720, 0.4659, 0.6720, 0.4659, 0.6850, 0.4396, 0.6850],
                [0.4698, 0.6720, 0.4843, 0.6720, 0.4843, 0.6850, 0.4698, 0.6850],
                [0.4698, 0.6720, 0.4843, 0.6720, 0.4843, 0.6850, 0.4698, 0.6850],
                [0.2047, 0.6870, 0.2730, 0.6870, 0.2730, 0.7000, 0.2047, 0.7000],
                [0.2047, 0.6870, 0.2730, 0.6870, 0.2730, 0.7000, 0.2047, 0.7000],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            ]
        ]
    )

    # 从预训练模型加载 BrosProcessor 对象，用于处理文本输入
    processor = BrosProcessor.from_pretrained(model_name)

    # 使用 processor 对象处理输入文本，将边界框信息添加到编码结果中
    encoding = processor("His name is Rocco.", return_tensors="pt")
    encoding["bbox"] = bbox

    # 使用原始模型生成输入编码的最后隐藏状态
    original_hidden_states = original_model(**encoding).last_hidden_state
    # pixel_values = processor(image, return_tensors="pt").pixel_values

    # 使用微调后的模型生成输入编码的最后隐藏状态
    last_hidden_states = model(**encoding).last_hidden_state

    # 断言原始模型和微调后模型的最后隐藏状态在一定误差范围内相等
    assert torch.allclose(original_hidden_states, last_hidden_states, atol=1e-4)

    # 如果指定了 PyTorch 模型保存路径，则保存微调后的模型和 processor 对象
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型和 processor 推送到 Hub 上，则执行推送操作
    if push_to_hub:
        model.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")
        processor.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="jinho8345/bros-base-uncased",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    # 添加参数：输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数：是否推送转换后的模型和处理器到 🤗 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_bros_checkpoint，传入解析后的参数
    convert_bros_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```