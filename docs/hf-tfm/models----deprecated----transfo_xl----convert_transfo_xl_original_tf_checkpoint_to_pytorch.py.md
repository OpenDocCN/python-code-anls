# `.\models\deprecated\transfo_xl\convert_transfo_xl_original_tf_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，使用 Apache 许可证 2.0 版本
# 详细信息可参考 http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面协议要求，本软件按"原样"分发，不附带任何明示或暗示的保证或条件
"""转换 Transformer XL 检查点和数据集。"""


import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块
import pickle  # 导入 pickle 序列化模块
import sys  # 导入系统相关的功能模块

import torch  # 导入 PyTorch 深度学习库

# 导入 Transformer XL 相关的配置、模型和权重加载工具
from transformers import TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.models.deprecated.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.deprecated.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logging.set_verbosity_info()  # 设置日志级别为 info

# 解决在加载 Python 2 数据集 pickle 文件时的问题
# 参考：https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/2121918#2121918
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules["data_utils"] = data_utils  # 修改模块路径，兼容 Python 2
sys.modules["vocabulary"] = data_utils  # 修改模块路径，兼容 Python 2


def convert_transfo_xl_checkpoint_to_pytorch(
    tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file
):
    if transfo_xl_dataset_file:
        # 转换预处理的语料库（参见原始 TensorFlow 仓库）
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")  # 加载 pickle 文件，编码为 Latin-1

        # 将词汇表和数据集缓存保存为字典（长期来看比 pickle 更好）
        pytorch_vocab_dump_path = pytorch_dump_folder_path + "/" + VOCAB_FILES_NAMES["pretrained_vocab_file"]
        print(f"Save vocabulary to {pytorch_vocab_dump_path}")
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)  # 保存词汇表字典到指定路径

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop("vocab", None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + "/" + CORPUS_NAME
        print(f"Save dataset to {pytorch_dataset_dump_path}")
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)  # 保存数据集字典（去除词汇表）到指定路径
    # 如果给定了 TensorFlow 的检查点路径
    if tf_checkpoint_path:
        # 将预训练的 TensorFlow 模型转换
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)

        # 打印转换过程中使用的配置和检查点路径信息
        print(f"Converting Transformer XL checkpoint from {tf_path} with config at {config_path}.")

        # 初始化 PyTorch 模型
        if transfo_xl_config_file == "":
            # 如果未提供配置文件路径，则使用默认配置
            config = TransfoXLConfig()
        else:
            # 从给定的 JSON 配置文件中加载配置
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        # 打印正在构建的 PyTorch 模型配置信息
        print(f"Building PyTorch model from configuration: {config}")
        
        # 使用配置初始化 TransformerXL 模型
        model = TransfoXLLMHeadModel(config)

        # 载入 TensorFlow 的权重到 PyTorch 模型中
        model = load_tf_weights_in_transfo_xl(model, config, tf_path)

        # 保存 PyTorch 模型的权重文件路径
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
        # 保存 PyTorch 模型的配置文件路径
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)

        # 打印保存 PyTorch 模型权重的路径
        print(f"Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}")
        # 将 PyTorch 模型的状态字典保存到指定路径
        torch.save(model.state_dict(), pytorch_weights_dump_path)

        # 打印保存 PyTorch 模型配置文件的路径
        print(f"Save configuration file to {os.path.abspath(pytorch_config_dump_path)}")
        # 将模型配置以 JSON 格式写入指定路径的文件中
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())
# 如果这个脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：用于指定 PyTorch 模型或数据集/词汇表的存储路径，是必需的参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    
    # 添加命令行参数：用于指定可选的 TensorFlow checkpoint 转换路径
    parser.add_argument(
        "--tf_checkpoint_path",
        default="",
        type=str,
        help="An optional path to a TensorFlow checkpoint path to be converted.",
    )
    
    # 添加命令行参数：用于指定可选的 TransfoXL 配置文件路径
    parser.add_argument(
        "--transfo_xl_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    
    # 添加命令行参数：用于指定可选的 TransfoXL 数据集文件路径，将其转换成词汇表
    parser.add_argument(
        "--transfo_xl_dataset_file",
        default="",
        type=str,
        help="An optional dataset file to be converted in a vocabulary.",
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_transfo_xl_checkpoint_to_pytorch，将 TensorFlow checkpoint 转换为 PyTorch 格式
    convert_transfo_xl_checkpoint_to_pytorch(
        args.tf_checkpoint_path,
        args.transfo_xl_config_file,
        args.pytorch_dump_folder_path,
        args.transfo_xl_dataset_file,
    )
```