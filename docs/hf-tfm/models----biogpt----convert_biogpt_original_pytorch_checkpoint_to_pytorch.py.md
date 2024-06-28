# `.\models\biogpt\convert_biogpt_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 声明编码格式为 UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，版权归 HuggingFace Inc. 团队所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 授权许可信息，使用 Apache License, Version 2.0 许可证。

# you may not use this file except in compliance with the License.
# 在遵守许可证的前提下，您不得使用本文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按"原样"分发本软件，
# 无论是明示的还是暗示的，包括但不限于对适销性和特定用途的适用性的暗示保证或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证了解权限和限制。

# 导入必要的库和模块
import argparse  # 参数解析模块
import json  # JSON 格式处理模块
import os  # 操作系统相关功能模块
import re  # 正则表达式模块
import shutil  # 文件操作模块

# 导入 PyTorch 库
import torch

# 导入 transformers 库中的配置和模型
from transformers import BioGptConfig, BioGptForCausalLM
# 从 biogpt 模型的 tokenization_biogpt 模块导入词汇文件相关常量
from transformers.models.biogpt.tokenization_biogpt import VOCAB_FILES_NAMES
# 从 transformers 库的 tokenization_utils_base 模块导入 TOKENIZER_CONFIG_FILE 常量
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
# 从 transformers 库的 utils 模块导入 WEIGHTS_NAME 和 logging 函数
from transformers.utils import WEIGHTS_NAME, logging

# 设置 logging 的警告级别为警告以上
logging.set_verbosity_warning()

# 设置 JSON 输出时的缩进量
json_indent = 2

# modified from https://github.com/facebookresearch/fairseq/blob/dd74992d0d143155998e9ed4076826bcea80fb06/fairseq/data/dictionary.py#L18
# 从 Fairseq 项目的代码中修改而来，用于创建字典映射符号到整数的类
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",  # 句子开始符号，默认为 "<s>"
        pad="<pad>",  # 填充符号，默认为 "<pad>"
        eos="</s>",  # 句子结束符号，默认为 "</s>"
        unk="<unk>",  # 未知符号，默认为 "<unk>"
        extra_special_symbols=None,  # 额外的特殊符号列表，默认为 None
    ):
        # 初始化各个符号
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        # 符号列表、符号计数、符号索引字典的初始化
        self.symbols = []
        self.count = []
        self.indices = {}
        # 添加开始、填充、结束、未知符号到符号列表中，并获取它们的索引
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        # 如果存在额外的特殊符号，则将其逐个添加到符号列表中
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        # 特殊符号的数量
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        # 检查当前字典对象与另一个字典对象是否相等
        return self.indices == other.indices

    def __getitem__(self, idx):
        # 获取指定索引处的符号，若索引超出范围则返回未知符号
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        # 返回字典中符号的数量
        return len(self.symbols)

    def __contains__(self, sym):
        # 检查指定符号是否存在于字典的索引中
        return sym in self.indices

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        # 从文本文件加载字典，文件格式为每行一个符号及其计数
        d = cls()
        d.add_from_file(f)
        return d
    # 将一个单词添加到字典中，如果单词已存在且不允许覆盖，则增加其出现次数
    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        # 如果单词已存在且不允许覆盖，则增加该单词的计数
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            # 否则将单词添加到字典中，并设置其初始计数
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    # 加载元数据（暂未实现具体逻辑）
    def _load_meta(self, lines):
        return 0

    # 从文件中加载预先存在的字典，将其符号添加到当前实例中
    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
        # 如果输入参数是字符串，则尝试打开文件进行递归调用
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please rebuild the dataset".format(f))
            return

        # 读取文件的所有行
        lines = f.readlines()
        # 获取元数据的起始行数（未实现具体逻辑）
        indices_start_line = self._load_meta(lines)

        # 遍历文件中的每一行（从起始行开始）
        for line in lines[indices_start_line:]:
            try:
                # 拆分行，获取单词和计数
                line, field = line.rstrip().rsplit(" ", 1)
                # 检查是否有 "#fairseq:overwrite" 标记，决定是否覆盖已存在的单词
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                # 如果字典中已存在该单词且不允许覆盖，则抛出异常
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                # 将单词添加到字典中
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")
def convert_biogpt_checkpoint_to_pytorch(biogpt_checkpoint_path, pytorch_dump_folder_path):
    # 检查源文件路径是否存在，若不存在则抛出数值错误异常
    if not os.path.exists(biogpt_checkpoint_path):
        raise ValueError(f"path {biogpt_checkpoint_path} does not exist!")
    
    # 创建目标文件夹路径，如果已存在则不创建
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    # 打印将结果写入的目标路径信息
    print(f"Writing results to {pytorch_dump_folder_path}")

    # 处理各种类型的模型

    # 拼接检查点文件路径
    checkpoint_file = os.path.join(biogpt_checkpoint_path, "checkpoint.pt")
    # 如果检查点文件不存在，则抛出数值错误异常
    if not os.path.isfile(checkpoint_file):
        raise ValueError(f"path to the file {checkpoint_file} does not exist!")
    # 加载检查点文件
    chkpt = torch.load(checkpoint_file, map_location="cpu")

    # 获取模型配置信息
    args = chkpt["cfg"]["model"]

    # 加载词典文件
    dict_file = os.path.join(biogpt_checkpoint_path, "dict.txt")
    # 如果词典文件不存在，则抛出数值错误异常
    if not os.path.isfile(dict_file):
        raise ValueError(f"path to the file {dict_file} does not exist!")
    # 加载并重写词典键值
    src_dict = Dictionary.load(dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    # 拼接目标词汇文件路径，并打印生成信息
    src_vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["vocab_file"])
    print(f"Generating {src_vocab_file} of {src_vocab_size} records")
    # 将重写后的词典写入目标词汇文件
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # 拼接并复制 BPE 代码文件到目标文件路径
    bpecodes_file = os.path.join(biogpt_checkpoint_path, "bpecodes")
    # 如果 BPE 代码文件不存在，则抛出数值错误异常
    if not os.path.isfile(bpecodes_file):
        raise ValueError(f"path to the file {bpecodes_file} does not exist!")
    # 拼接并复制 BPE 代码文件到目标文件路径
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    shutil.copyfile(bpecodes_file, merges_file)

    # 拼接并创建 Biogpt 模型配置文件路径
    biogpt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")
    # 定义模型配置字典，包含模型的各种配置参数
    model_conf = {
        "activation_dropout": args["activation_dropout"],  # 激活函数的dropout率
        "architectures": ["BioGptForCausalLM"],  # 模型架构，这里使用了单个架构
        "attention_probs_dropout_prob": args["attention_dropout"],  # 注意力机制中的dropout率
        "bos_token_id": 0,  # 起始标记的token id
        "eos_token_id": 2,  # 结束标记的token id
        "hidden_act": args["activation_fn"],  # 隐藏层激活函数类型
        "hidden_dropout_prob": args["dropout"],  # 隐藏层的dropout率
        "hidden_size": args["decoder_embed_dim"],  # 隐藏层的维度大小
        "initializer_range": 0.02,  # 初始化范围
        "intermediate_size": args["decoder_ffn_embed_dim"],  # 中间层的维度大小
        "layer_norm_eps": 1e-12,  # Layer Normalization的epsilon参数
        "layerdrop": args["decoder_layerdrop"],  # 层级dropout率
        "max_position_embeddings": args["max_target_positions"],  # 最大位置嵌入长度
        "model_type": "biogpt",  # 模型类型
        "num_attention_heads": args["decoder_attention_heads"],  # 注意力头的数量
        "num_hidden_layers": args["decoder_layers"],  # 隐藏层的数量
        "pad_token_id": 1,  # 填充标记的token id
        "scale_embedding": not args["no_scale_embedding"],  # 是否缩放嵌入
        "tie_word_embeddings": args["share_decoder_input_output_embed"],  # 是否共享解码器输入输出的嵌入
        "vocab_size": src_vocab_size,  # 词汇表大小
    }
    
    # 打印消息，指示正在生成biogpt模型的配置文件
    print(f"Generating {biogpt_model_config_file}")
    # 将模型配置写入JSON文件
    with open(biogpt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))
    
    # tokenizer配置
    biogpt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)
    
    tokenizer_conf = {
        "bos_token": "<s>",  # 起始标记
        "eos_token": "</s>",  # 结束标记
        "model_max_length": 1024,  # 模型的最大长度
        "pad_token": "<pad>",  # 填充标记
        "special_tokens_map_file": None,  # 特殊标记映射文件
        "tokenizer_class": "BioGptTokenizer",  # tokenizer类名
        "unk_token": "<unk>",  # 未知标记
    }
    
    # 打印消息，指示正在生成biogpt tokenizer的配置文件
    print(f"Generating {biogpt_tokenizer_config_file}")
    # 将tokenizer配置写入JSON文件
    with open(biogpt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))
    
    # 模型状态字典
    model_state_dict = chkpt["model"]
    
    # 移除不需要的键
    ignore_keys = [
        "decoder.version",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)
    
    # 获取所有层的名称
    layer_names = list(model_state_dict.keys())
    for layer_name in layer_names:
        if layer_name.endswith("output_projection.weight"):
            # 将decoder结尾的层名称替换为biogpt
            model_state_dict[layer_name.replace("decoder.", "")] = model_state_dict.pop(layer_name)
        else:
            # 将decoder替换为biogpt
            model_state_dict[layer_name.replace("decoder", "biogpt")] = model_state_dict.pop(layer_name)
    
    # 从预训练文件夹加载配置
    config = BioGptConfig.from_pretrained(pytorch_dump_folder_path)
    # 创建新的BioGptForCausalLM模型
    model_new = BioGptForCausalLM(config)
    
    # 检查模型加载是否成功
    model_new.load_state_dict(model_state_dict)
    
    # 保存模型权重
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)
    
    # 打印完成消息
    print("Conversion is done!")
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--biogpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts,"
            " bpecodes, etc."
        ),
    )
    # 添加一个必需的命令行参数，用于指定 Biogpt 的检查点文件路径

    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        required=True, 
        help="Path to the output PyTorch model."
    )
    # 添加另一个必需的命令行参数，用于指定 PyTorch 模型的输出文件夹路径

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 对象中

    convert_biogpt_checkpoint_to_pytorch(args.biogpt_checkpoint_path, args.pytorch_dump_folder_path)
    # 调用函数 convert_biogpt_checkpoint_to_pytorch，传递 Biogpt 检查点路径和 PyTorch 模型输出路径作为参数
```