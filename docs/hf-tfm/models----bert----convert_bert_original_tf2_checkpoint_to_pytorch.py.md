# `.\models\bert\convert_bert_original_tf2_checkpoint_to_pytorch.py`

```py
    # 版权声明和许可信息
    """
    Copyright 2020 The HuggingFace Team. All rights reserved.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    
    # 引入所需库和模块
    import argparse  # 解析命令行参数的库
    import os  # 操作系统相关功能的库
    import re  # 正则表达式的库
    
    import tensorflow as tf  # TensorFlow 深度学习框架
    import torch  # PyTorch 深度学习框架
    
    from transformers import BertConfig, BertModel  # Hugging Face 提供的 Bert 相关类
    from transformers.utils import logging  # Hugging Face 提供的日志功能
    
    logging.set_verbosity_info()  # 设置日志记录级别为信息
    logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
    

def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，显示转换的 TensorFlow 检查点路径
    
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)  # 列出 TensorFlow 模型中的所有变量名和形状
    names = []  # 存储变量名
    arrays = []  # 存储加载的变量数组
    layer_depth = []  # 存储每个变量名的层级深度
    
    # 遍历每个变量名和形状
    for full_name, shape in init_vars:
        # logger.info(f"Loading TF weight {name} with shape {shape}")
        name = full_name.split("/")  # 按斜杠分割变量名，获取各级名称
        
        # 如果是特定的非模型层或优化层，则跳过加载
        if full_name == "_CHECKPOINTABLE_OBJECT_GRAPH" or name[0] in ["global_step", "save_counter"]:
            logger.info(f"Skipping non-model layer {full_name}")
            continue
        if "optimizer" in full_name:
            logger.info(f"Skipping optimization layer {full_name}")
            continue
        if name[0] == "model":
            # 忽略初始的 'model' 层级
            name = name[1:]
        
        # 计算变量名的层级深度
        depth = 0
        for _name in name:
            if _name.startswith("layer_with_weights"):
                depth += 1
            else:
                break
        layer_depth.append(depth)
        
        # 加载变量数据
        array = tf.train.load_variable(tf_path, full_name)
        names.append("/".join(name))  # 将分割后的名称重新连接为字符串形式
        arrays.append(array)  # 将加载的变量数组添加到列表中
    
    logger.info(f"Read a total of {len(arrays):,} layers")  # 记录日志，显示总共加载了多少层变量

    # 进行完整性检查
    # 检查层深度列表中是否存在不同的深度值，如果存在则抛出数值错误异常
    if len(set(layer_depth)) != 1:
        raise ValueError(f"Found layer names with different depths (layer depth {list(set(layer_depth))})")
    
    # 将层深度列表转换为集合去重，然后转换回列表，并获取唯一的深度值
    layer_depth = list(set(layer_depth))[0]
    
    # 检查模型的层深度是否为1，如果不是则抛出数值错误异常，说明模型包含了除了嵌入/编码器层之外的其他层
    if layer_depth != 1:
        raise ValueError(
            "The model contains more than just the embedding/encoder layers. This script does not handle MLM/NSP"
            " heads."
        )

    # 输出日志信息，表明开始转换权重
    logger.info("Converting weights...")
    
    # 返回已转换的模型对象
    return model
# 将 TensorFlow 2.x 的检查点文件转换为 PyTorch 模型的函数
def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    # 打印日志信息，加载基于指定配置文件的模型
    logger.info(f"Loading model based on config from {config_path}...")
    # 从 JSON 文件中加载配置信息
    config = BertConfig.from_json_file(config_path)
    # 根据配置创建 BertModel 实例
    model = BertModel(config)

    # 打印日志信息，加载 TensorFlow 2.x 检查点的权重
    logger.info(f"Loading weights from checkpoint {tf_checkpoint_path}...")
    # 调用函数加载 TensorFlow 2.x 检查点中的权重到 PyTorch 模型中
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # 打印日志信息，保存 PyTorch 模型
    logger.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    # 使用 PyTorch 的函数保存模型的状态字典到指定路径
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定 TensorFlow 2.x 检查点路径
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    # 添加命令行参数，指定 BERT 模型的配置文件路径
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    # 添加命令行参数，指定输出的 PyTorch 模型路径（包括文件名）
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入解析得到的参数
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
```