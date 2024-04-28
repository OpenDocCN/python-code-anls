# `.\transformers\models\bros\convert_bros_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息
"""Convert Bros checkpoints."""

# 导入模块
import argparse

# 导入原始的 Bros 模块
import bros  # original repo
# 导入 PyTorch 库
import torch

# 导入 transformers 库中的 BrosConfig、BrosModel 和 BrosProcessor 类
from transformers import BrosConfig, BrosModel, BrosProcessor
# 从 transformers 库中导入 logging 模块
from transformers.utils import logging

# 设置日志级别为 INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 根据模型名称获取 BrosConfig 对象
def get_configs(model_name):
    # 从预训练模型加载 BrosConfig 对象
    bros_config = BrosConfig.from_pretrained(model_name)
    return bros_config


# 删除忽略的键
def remove_ignore_keys_(state_dict):
    # 要删除的键列表
    ignore_keys = [
        "embeddings.bbox_sinusoid_emb.inv_freq",
    ]
    # 遍历要删除的键列表
    for k in ignore_keys:
        # 如果键存在，则删除对应的键值对
        state_dict.pop(k, None)


# 重命名键名
def rename_key(name):
    # 重命名 "embeddings.bbox_projection.weight" 键名为 "bbox_embeddings.bbox_projection.weight"
    if name == "embeddings.bbox_projection.weight":
        name = "bbox_embeddings.bbox_projection.weight"

    # 重命名 "embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq" 键名为 "bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq"
    if name == "embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq"

    # 重命名 "embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq" 键名为 "bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq"
    if name == "embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq"

    return name


# 转换状态字典
def convert_state_dict(orig_state_dict, model):
    # 重命名键名
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    # 删除忽略的键
    remove_ignore_keys_(orig_state_dict)

    return orig_state_dict


# 转换 Bros 检查点
def convert_bros_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # 加载原始模型
    original_model = bros.BrosModel.from_pretrained(model_name).eval()

    # 加载 HuggingFace 模型
    bros_config = get_configs(model_name)
    model = BrosModel.from_pretrained(model_name, config=bros_config)
    model.eval()

    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 转换状态字典
    new_state_dict = convert_state_dict(state_dict, model)
    # 加载转换后的状态字典到 HuggingFace 模型
    model.load_state_dict(new_state_dict)

    # 验证结果

    # 原始的 BROS 模型需要每个边界框四个点（8个浮点值），准备形状为 [batch_size, seq_len, 8] 的边界框
```  
    # 创建包含边界框数据的张量
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

    # 从预训练模型名称创建 BrosProcessor 对象
    processor = BrosProcessor.from_pretrained(model_name)

    # 对输入文本进行编码，并将边界框数据添加到编码结果中
    encoding = processor("His name is Rocco.", return_tensors="pt")
    encoding["bbox"] = bbox

    # 获取原始模型的隐藏状态
    original_hidden_states = original_model(**encoding).last_hidden_state
    # pixel_values = processor(image, return_tensors="pt").pixel_values

    # 获取当前模型的隐藏状态
    last_hidden_states = model(**encoding).last_hidden_state

    # 检查原始模型和当前模型的隐藏状态是否非常接近
    assert torch.allclose(original_hidden_states, last_hidden_states, atol=1e-4)

    # 如果指定了 PyTorch 模型保存路径，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub，则将模型和处理器推送到指定 Hub 仓库
    if push_to_hub:
        model.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")
        processor.push_to_hub("jinho8345/" + model_name.split("/")[-1], commit_message="Update model")
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()

    # 必选参数
    # 添加模型名称参数
    parser.add_argument(
        "--model_name",
        default="jinho8345/bros-base-uncased",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    # 添加 PyTorch 模型输出目录参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加是否推送至 🤗 hub 的参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数转换 Bros 检查点
    convert_bros_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```