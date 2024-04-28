# `.\transformers\models\bert\convert_bert_original_tf2_checkpoint_to_pytorch.py`

```
# 引入 argparse 模块，用于解析命令行参数
import argparse
# 引入 os 模块，提供与操作系统交互的功能
import os
# 引入 re 模块，用于正则表达式操作
import re

# 引入 TensorFlow 模块
import tensorflow as tf
# 引入 PyTorch 模块
import torch

# 从 transformers 模块中引入 BertConfig 和 BertModel 类
from transformers import BertConfig, BertModel
# 从 transformers.utils 模块中引入 logging 函数
from transformers.utils import logging

# 设置日志输出级别为 INFO
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)


# 加载 TensorFlow 2.x 权重到 Bert 模型中的函数
def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    # 获取 TensorFlow checkpoint 文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 输出日志信息，显示正在从 TensorFlow checkpoint 中加载权重
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 加载 TensorFlow 模型的权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    layer_depth = []
    # 遍历 TensorFlow 模型的变量名和形状
    for full_name, shape in init_vars:
        # 将变量名根据 "/" 分割为列表
        name = full_name.split("/")
        # 如果变量名指向不是模型层或非模型参数，则跳过
        if full_name == "_CHECKPOINTABLE_OBJECT_GRAPH" or name[0] in ["global_step", "save_counter"]:
            logger.info(f"Skipping non-model layer {full_name}")
            continue
        # 如果变量名指向优化器相关参数，则跳过
        if "optimizer" in full_name:
            logger.info(f"Skipping optimization layer {full_name}")
            continue
        # 如果变量名的第一级为 "model"，则忽略该层
        if name[0] == "model":
            # 忽略初始的 "model" 层
            name = name[1:]
        # 计算变量名的深度（层数）
        depth = 0
        for _name in name:
            if _name.startswith("layer_with_weights"):
                depth += 1
            else:
                break
        # 记录变量名的深度
        layer_depth.append(depth)
        # 加载 TensorFlow 变量的数据
        array = tf.train.load_variable(tf_path, full_name)
        # 记录变量名和数据
        names.append("/".join(name))
        arrays.append(array)
    # 输出日志信息，显示共读取了多少层变量
    logger.info(f"Read a total of {len(arrays):,} layers")

    # 检查权重加载是否正常
    # 检查层深度集合的长度，确保所有层的深度相同，若不同则引发 ValueError 异常，给出不同深度的层信息
    if len(set(layer_depth)) != 1:
        raise ValueError(f"Found layer names with different depths (layer depth {list(set(layer_depth))})")
    # 将层深度转换为列表，并保留唯一的深度值
    layer_depth = list(set(layer_depth))[0]
    # 如果层深度不为 1，则说明模型包含了嵌入/编码器层以外的层，此脚本不处理 MLM/NSP 头部
    if layer_depth != 1:
        raise ValueError(
            "The model contains more than just the embedding/encoder layers. This script does not handle MLM/NSP"
            " heads."
        )

    # 转换权重
    logger.info("Converting weights...")  # 记录日志，提示正在转换权重
    # 返回模型
    return model
# 将 TensorFlow 2.x 的检查点文件转换为 PyTorch 模型的函数
def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    # 实例化模型
    logger.info(f"Loading model based on config from {config_path}...")
    # 从配置文件加载配置信息
    config = BertConfig.from_json_file(config_path)
    # 创建 BertModel 模型对象
    model = BertModel(config)

    # 从检查点文件加载权重
    logger.info(f"Loading weights from checkpoint {tf_checkpoint_path}...")
    # 加载 TensorFlow 2.x 检查点文件中的权重到 PyTorch 模型中
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # 保存 PyTorch 模型
    logger.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    # 将 PyTorch 模型的状态字典保存到指定路径
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入命令行参数中的路径信息
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
```