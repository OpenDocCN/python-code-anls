# `.\transformers\models\biogpt\convert_biogpt_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置编码格式为 UTF-8
# 声明著作权和许可证信息
# 引入必要的模块
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据的处理
import os  # 用于操作系统相关的功能
import re  # 用于正则表达式匹配
import shutil  # 用于高级文件操作

import torch  # PyTorch 深度学习框架

# 从 transformers 库中引入必要的类和函数
from transformers import BioGptConfig, BioGptForCausalLM
from transformers.models.biogpt.tokenization_biogpt import VOCAB_FILES_NAMES  # 从 biogpt 模型中引入词汇表文件名
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE  # 从基类中引入分词器配置文件名
from transformers.utils import WEIGHTS_NAME, logging  # 引入权重文件名和日志模块

# 设置日志级别为警告
logging.set_verbosity_warning()

# 设置 JSON 输出缩进量
json_indent = 2


# 从 fairseq 库中修改而来，定义一个符号到整数的映射类
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # 使用关键字参数
        bos="<s>",  # 开始符号，默认为 "<s>"
        pad="<pad>",  # 填充符号，默认为 "<pad>"
        eos="</s>",  # 结束符号，默认为 "</s>"
        unk="<unk>",  # 未知符号，默认为 "<unk>"
        extra_special_symbols=None,  # 额外的特殊符号，默认为 None
    ):
        # 初始化特殊符号及其索引
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []  # 符号列表
        self.count = []  # 符号计数列表
        self.indices = {}  # 符号到索引的映射
        # 添加特殊符号到词汇表中，并获取其索引
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        # 若有额外的特殊符号，则依次添加到词汇表中
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        # 计算特殊符号的数量
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        # 比较两个字典是否相等
        return self.indices == other.indices

    def __getitem__(self, idx):
        # 根据索引获取符号
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        # 返回词汇表中符号的数量
        return len(self.symbols)

    def __contains__(self, sym):
        # 检查符号是否存在于词汇表中
        return sym in self.indices

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```py
        """
        # 从文本文件中加载字典
        d = cls()  # 创建字典对象
        d.add_from_file(f)  # 从文件中添加符号
        return d
    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        # 如果词已存在于索引中且不允许覆盖，则更新计数
        if word in self.indices and not overwrite:
            idx = self.indices[word]  # 获取词的索引
            self.count[idx] = self.count[idx] + n  # 更新词频计数
            return idx  # 返回词的索引
        else:
            idx = len(self.symbols)  # 获取新词的索引
            self.indices[word] = idx  # 将词和索引添加到索引字典中
            self.symbols.append(word)  # 将词添加到词表中
            self.count.append(n)  # 添加词频计数
            return idx  # 返回新词的索引

    def _load_meta(self, lines):
        """
        Placeholder function, not used in this implementation.
        """
        return 0  # 返回索引开始的行号，但在此实现中未使用

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
        # 如果输入为字符串，则尝试打开文件
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)  # 递归调用，打开文件对象进行处理
            except FileNotFoundError as fnfe:
                raise fnfe  # 文件未找到错误
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please rebuild the dataset".format(f))  # 文件编码错误
            return  # 处理完成，返回

        lines = f.readlines()  # 读取文件的所有行
        indices_start_line = self._load_meta(lines)  # 载入元数据，获取索引开始的行号

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)  # 从行中分离词和计数
                if field == "#fairseq:overwrite":  # 检查是否需要覆盖词条
                    overwrite = True
                    line, field = line.rsplit(" ", 1)  # 去除覆盖标记
                else:
                    overwrite = False
                count = int(field)  # 转换计数为整数
                word = line  # 获取词
                # 检查是否出现重复词条且不允许覆盖
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )  # 抛出重复词条错误
                self.add_symbol(word, n=count, overwrite=overwrite)  # 添加词到词典中
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")  # 抛出格式错误
def rewrite_dict_keys(d):
    # 移除单词分隔符，添加单词结束符（</w>），如果单词没有被分隔，
    # 例如：d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    keep_keys = "<s> <pad> </s> <unk>".split()
    # 恢复特殊标记
    for k in keep_keys:
        # 删除以 "</w>" 结尾的键
        del d2[f"{k}</w>"]
        # 恢复原始键值对
        d2[k] = d[k]  # 恢复
    return d2


def convert_biogpt_checkpoint_to_pytorch(biogpt_checkpoint_path, pytorch_dump_folder_path):
    # 准备工作
    if not os.path.exists(biogpt_checkpoint_path):
        raise ValueError(f"path {biogpt_checkpoint_path} does not exist!")
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    # 处理各种类型的模型

    checkpoint_file = os.path.join(biogpt_checkpoint_path, "checkpoint.pt")
    if not os.path.isfile(checkpoint_file):
        raise ValueError(f"path to the file {checkpoint_file} does not exist!")
    chkpt = torch.load(checkpoint_file, map_location="cpu")

    args = chkpt["cfg"]["model"]

    # 词典
    dict_file = os.path.join(biogpt_checkpoint_path, "dict.txt")
    if not os.path.isfile(dict_file):
        raise ValueError(f"path to the file {dict_file} does not exist!")
    src_dict = Dictionary.load(dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["vocab_file"])
    print(f"Generating {src_vocab_file} of {src_vocab_size} records")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # 合并文件（bpecodes）
    bpecodes_file = os.path.join(biogpt_checkpoint_path, "bpecodes")
    if not os.path.isfile(bpecodes_file):
        raise ValueError(f"path to the file {bpecodes_file} does not exist!")

    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    shutil.copyfile(bpecodes_file, merges_file)

    # 模型配置
    biogpt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")
    # 定义模型配置字典，包含各种模型参数
    model_conf = {
        "activation_dropout": args["activation_dropout"],
        "architectures": ["BioGptForCausalLM"],
        "attention_probs_dropout_prob": args["attention_dropout"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": args["activation_fn"],
        "hidden_dropout_prob": args["dropout"],
        "hidden_size": args["decoder_embed_dim"],
        "initializer_range": 0.02,
        "intermediate_size": args["decoder_ffn_embed_dim"],
        "layer_norm_eps": 1e-12,
        "layerdrop": args["decoder_layerdrop"],
        "max_position_embeddings": args["max_target_positions"],
        "model_type": "biogpt",
        "num_attention_heads": args["decoder_attention_heads"],
        "num_hidden_layers": args["decoder_layers"],
        "pad_token_id": 1,
        "scale_embedding": not args["no_scale_embedding"],
        "tie_word_embeddings": args["share_decoder_input_output_embed"],
        "vocab_size": src_vocab_size,
    }

    # 打印提示信息
    print(f"Generating {biogpt_model_config_file}")
    # 将模型配置字典写入文件
    with open(biogpt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))

    # tokenizer配置
    biogpt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)

    tokenizer_conf = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1024,
        "pad_token": "<pad>",
        "special_tokens_map_file": None,
        "tokenizer_class": "BioGptTokenizer",
        "unk_token": "<unk>",
    }

    # 打印提示信息
    print(f"Generating {biogpt_tokenizer_config_file}")
    # 将tokenizer配置字典写入文件
    with open(biogpt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))

    # 模型
    model_state_dict = chkpt["model"]

    # 移除不需要的键
    ignore_keys = [
        "decoder.version",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    layer_names = list(model_state_dict.keys())
    for layer_name in layer_names:
        # 重命名特定键
        if layer_name.endswith("output_projection.weight"):
            model_state_dict[layer_name.replace("decoder.", "")] = model_state_dict.pop(layer_name)
        else:
            model_state_dict[layer_name.replace("decoder", "biogpt")] = model_state_dict.pop(layer_name)

    # 从预训练模型路径加载配置
    config = BioGptConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = BioGptForCausalLM(config)

    # 检查模型加载是否正常
    model_new.load_state_dict(model_state_dict)

    # 保存模型权重
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    # 打印完成信息
    print("Conversion is done!")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
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
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 Biogpt 检查点转换为 PyTorch 模型
    convert_biogpt_checkpoint_to_pytorch(args.biogpt_checkpoint_path, args.pytorch_dump_folder_path)
```