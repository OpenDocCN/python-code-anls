# `.\models\deprecated\transfo_xl\convert_transfo_xl_original_tf_checkpoint_to_pytorch.py`

```py
# 这是一个用于将 Transformer XL 检查点和数据集转换为 PyTorch 格式的脚本
# 导入必要的库和模块
import argparse
import os
import pickle
import sys

import torch

from transformers import TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.models.deprecated.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.deprecated.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging

# 设置日志级别为信息
logging.set_verbosity_info()

# 为了能够加载 Python 2 数据集的 pickle 文件，做一些兼容性处理
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules["data_utils"] = data_utils
sys.modules["vocabulary"] = data_utils

# 定义一个函数，将 TensorFlow 检查点转换为 PyTorch 格式
def convert_transfo_xl_checkpoint_to_pytorch(
    tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file
):
    # 如果提供了数据集文件
    if transfo_xl_dataset_file:
        # 载入预处理好的数据集
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        # 将词汇表和数据集缓存保存为字典格式
        pytorch_vocab_dump_path = pytorch_dump_folder_path + "/" + VOCAB_FILES_NAMES["pretrained_vocab_file"]
        print(f"Save vocabulary to {pytorch_vocab_dump_path}")
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop("vocab", None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + "/" + CORPUS_NAME
        print(f"Save dataset to {pytorch_dataset_dump_path}")
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)


这段代码是用于将 Transformer XL 检查点和数据集转换为 PyTorch 格式的脚本。主要包含以下功能:

1. 导入必要的库和模块。
2. 设置日志级别为信息。
3. 为了兼容 Python 2 数据集的 pickle 文件，做了一些兼容性处理。
4. 定义 `convert_transfo_xl_checkpoint_to_pytorch` 函数,用于将 TensorFlow 检查点转换为 PyTorch 格式。
   - 如果提供了数据集文件,会载入预处理好的数据集。
   - 将词汇表和数据集缓存保存为字典格式,并分别保存到指定文件路径。

整个脚本的目的是方便将 Transformer XL 模型从 TensorFlow 迁移到 PyTorch 使用。
    # 如果给定了 TensorFlow 检查点路径
    if tf_checkpoint_path:
        # 转换预训练的 TensorFlow 模型
        config_path = os.path.abspath(transfo_xl_config_file)
        # 获取 TensorFlow 检查点的绝对路径
        tf_path = os.path.abspath(tf_checkpoint_path)
    
        # 打印转换过程的信息
        print(f"Converting Transformer XL checkpoint from {tf_path} with config at {config_path}.")
        
        # 初始化 PyTorch 模型
        if transfo_xl_config_file == "":
            # 如果没有指定 TransfoXL 的配置文件，使用默认配置
            config = TransfoXLConfig()
        else:
            # 如果指定了 TransfoXL 的配置文件，从文件中加载配置
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        # 打印正在构建的 PyTorch 模型的配置信息
        print(f"Building PyTorch model from configuration: {config}")
        # 创建 TransfoXL 模型
        model = TransfoXLLMHeadModel(config)
    
        # 加载 TensorFlow 模型权重到 PyTorch 模型
        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
        
        # 保存 PyTorch 模型的权重
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
        # 保存 PyTorch 模型的配置文件
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
        print(f"Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}")
        # 将 PyTorch 模型的权重保存到文件
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print(f"Save configuration file to {os.path.abspath(pytorch_config_dump_path)}")
        # 将 PyTorch 模型的配置保存到 JSON 文件
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())
# 如果当前脚本是主程序入口（而不是被导入），则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个必需的参数，用于指定 PyTorch 模型或数据集/词汇表保存的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    # 添加一个可选参数，用于指定 TensorFlow 检查点路径
    parser.add_argument(
        "--tf_checkpoint_path",
        default="",
        type=str,
        help="An optional path to a TensorFlow checkpoint path to be converted.",
    )
    # 添加一个可选参数，用于指定 Transformer-XL 配置文件路径
    parser.add_argument(
        "--transfo_xl_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加一个可选参数，用于指定 Transformer-XL 数据集文件路径
    parser.add_argument(
        "--transfo_xl_dataset_file",
        default="",
        type=str,
        help="An optional dataset file to be converted in a vocabulary.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_transfo_xl_checkpoint_to_pytorch 函数，传入解析得到的参数
    convert_transfo_xl_checkpoint_to_pytorch(
        args.tf_checkpoint_path,
        args.transfo_xl_config_file,
        args.pytorch_dump_folder_path,
        args.transfo_xl_dataset_file,
    )
```