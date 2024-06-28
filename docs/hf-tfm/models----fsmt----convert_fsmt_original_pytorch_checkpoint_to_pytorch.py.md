# `.\models\fsmt\convert_fsmt_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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

# Note: if you intend to run this script make sure you look under scripts/fsmt/
# to locate the appropriate script to do the work correctly. There is a set of scripts to:
# - download and prepare data and run the conversion script
# - perform eval to get the best hparam into the config
# - generate model_cards - useful if you have multiple models from the same paper

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理JSON格式数据
import os  # 用于操作系统相关的功能
import re  # 用于正则表达式操作
from collections import OrderedDict  # 导入OrderedDict，用于有序字典
from os.path import basename, dirname  # 导入basename和dirname函数，用于处理文件路径

import fairseq  # 导入fairseq库
import torch  # 导入PyTorch库
from fairseq import hub_utils  # 导入fairseq的hub_utils模块
from fairseq.data.dictionary import Dictionary  # 导入fairseq的Dictionary类

from transformers import FSMTConfig, FSMTForConditionalGeneration  # 导入transformers库中的FSMTConfig和FSMTForConditionalGeneration类
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES  # 导入transformers库中FSMT的tokenization_fsmt模块中的VOCAB_FILES_NAMES变量
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE  # 导入transformers库中的TOKENIZER_CONFIG_FILE变量
from transformers.utils import WEIGHTS_NAME, logging  # 导入transformers库中的WEIGHTS_NAME和logging模块

logging.set_verbosity_warning()  # 设置日志输出级别为警告级别

json_indent = 2  # 设置JSON格式化时的缩进空格数为2

# 基于在wmt19测试数据上对一系列`num_beams`、`length_penalty`和`early_stopping`值的搜索结果，选择最佳的默认值
best_score_hparams = {
    # fairseq模型配置:
    "wmt19-ru-en": {"length_penalty": 1.1},
    "wmt19-en-ru": {"length_penalty": 1.15},
    "wmt19-en-de": {"length_penalty": 1.0},
    "wmt19-de-en": {"length_penalty": 1.1},
    # allenai模型配置:
    "wmt16-en-de-dist-12-1": {"length_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"length_penalty": 0.6},
    "wmt16-en-de-12-1": {"length_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"length_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"length_penalty": 0.6},
}

# 将不同模型重新映射到它们的组织名称
org_names = {}
for m in ["wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"]:
    org_names[m] = "facebook"
for m in [
    "wmt16-en-de-dist-12-1",
    "wmt16-en-de-dist-6-1",
    "wmt16-en-de-12-1",
    "wmt19-de-en-6-6-base",
    "wmt19-de-en-6-6-big",
]:
    org_names[m] = "allenai"


def rewrite_dict_keys(d):
    # TODO: Implement function to rewrite dictionary keys
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up,
    # 创建一个新的字典 d2，将输入字典 d 中的特定键进行处理：
    # - 如果键以 "@@" 结尾，则去除 "@@" 后作为新键，保留原值 v；
    # - 否则，在键末尾添加 "</w>" 字符串作为新键，并保留原值 v。
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    
    # 定义要保留的特殊键列表
    keep_keys = "<s> <pad> </s> <unk>".split()
    
    # 遍历要保留的特殊键，并在 d2 中做相应的操作：
    # - 删除 d2 中以 "<key></w>" 形式结尾的键；
    # - 将原始键的值复制回 d2 中对应的键位置，以恢复原始值。
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k]  # 恢复原始值
    
    # 返回处理后的字典 d2
    return d2
def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):
    # 检查给定路径的文件是否存在
    assert os.path.exists(fsmt_checkpoint_path)
    # 创建目标文件夹路径，如果不存在则创建
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    # 打印提示信息，指示结果将写入的目标文件夹路径
    print(f"Writing results to {pytorch_dump_folder_path}")

    # 处理不同类型的模型

    # 获取检查点文件名和文件夹路径
    checkpoint_file = basename(fsmt_checkpoint_path)
    fsmt_folder_path = dirname(fsmt_checkpoint_path)

    # 使用 fairseq.model_parallel.models.transformer.ModelParallelTransformerModel 类
    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    # 获取可用的模型列表
    models = cls.hub_models()
    kwargs = {"bpe": "fastbpe", "tokenizer": "moses"}
    data_name_or_path = "."
    # 注意：由于模型转储旧，fairseq 已经升级了其模型，因此在保存的权重上进行了重写和分割，
    # 因此不能直接在模型文件上使用 torch.load()。
    # 参见 fairseq_model.py 中的 upgrade_state_dict(state_dict)。
    print(f"using checkpoint {checkpoint_file}")
    # 使用 hub_utils.from_pretrained 加载模型检查点
    chkpt = hub_utils.from_pretrained(
        fsmt_folder_path, checkpoint_file, data_name_or_path, archive_map=models, **kwargs
    )

    # 获取模型参数
    args = vars(chkpt["args"]["model"])

    # 获取源语言和目标语言
    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]

    # 获取数据根路径和模型目录名
    data_root = dirname(pytorch_dump_folder_path)
    model_dir = basename(pytorch_dump_folder_path)

    # 字典文件
    src_dict_file = os.path.join(fsmt_folder_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_folder_path, f"dict.{tgt_lang}.txt")

    # 加载源语言和目标语言的字典
    src_dict = Dictionary.load(src_dict_file)
    # 重写字典键值
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-src.json")
    # 打印提示信息，生成源语言词汇表文件
    print(f"Generating {src_vocab_file} of {src_vocab_size} of {src_lang} records")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # 检测是否需要执行小写转换，根据源语言词汇表中是否存在大写字母来判断
    do_lower_case = True
    for k in src_vocab.keys():
        if not k.islower():
            do_lower_case = False
            break

    # 加载目标语言的字典
    tgt_dict = Dictionary.load(tgt_dict_file)
    # 重写字典键值
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    tgt_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-tgt.json")
    # 打印提示信息，生成目标语言词汇表文件
    print(f"Generating {tgt_vocab_file} of {tgt_vocab_size} of {tgt_lang} records")
    with open(tgt_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))

    # merges_file (bpecodes)
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    # 遍历文件名列表，找到存在的合并文件
    for fn in ["bpecodes", "code"]:  # older fairseq called the merges file "code"
        fsmt_merges_file = os.path.join(fsmt_folder_path, fn)
        if os.path.exists(fsmt_merges_file):
            break
    # 从指定文件中读取内容，使用 UTF-8 编码打开文件
    with open(fsmt_merges_file, encoding="utf-8") as fin:
        merges = fin.read()
    # 使用正则表达式去除字符串末尾的数字（频率信息）
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    # 打印生成信息，输出文件名变量的值
    print(f"Generating {merges_file}")
    # 使用 UTF-8 编码打开文件，并将处理后的字符串写入文件
    with open(merges_file, "w", encoding="utf-8") as fout:
        fout.write(merges)

    # model config
    # 构建模型配置文件路径
    fsmt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    # 校验 BPE/tokenizer 配置，当前强制使用 moses+fastbpe -
    # 如果未来模型使用其他类型的 tokenizer，需要扩展支持
    assert args["bpe"] == "fastbpe", f"need to extend tokenizer to support bpe={args['bpe']}"
    assert args["tokenizer"] == "moses", f"need to extend tokenizer to support bpe={args['tokenizer']}"

    # 配置模型参数
    model_conf = {
        "architectures": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "activation_dropout": args["activation_dropout"],
        "activation_function": "relu",
        "attention_dropout": args["attention_dropout"],
        "d_model": args["decoder_embed_dim"],
        "dropout": args["dropout"],
        "init_std": 0.02,
        "max_position_embeddings": args["max_source_positions"],
        "num_hidden_layers": args["encoder_layers"],
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "langs": [src_lang, tgt_lang],
        "encoder_attention_heads": args["encoder_attention_heads"],
        "encoder_ffn_dim": args["encoder_ffn_embed_dim"],
        "encoder_layerdrop": args["encoder_layerdrop"],
        "encoder_layers": args["encoder_layers"],
        "decoder_attention_heads": args["decoder_attention_heads"],
        "decoder_ffn_dim": args["decoder_ffn_embed_dim"],
        "decoder_layerdrop": args["decoder_layerdrop"],
        "decoder_layers": args["decoder_layers"],
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "is_encoder_decoder": True,
        "scale_embedding": not args["no_scale_embedding"],
        "tie_word_embeddings": args["share_all_embeddings"],
    }

    # 设置模型配置的默认超参数
    model_conf["num_beams"] = 5
    model_conf["early_stopping"] = False
    # 如果最佳分数的超参数中包含长度惩罚项，则使用该值；否则设置为默认值 1.0
    if model_dir in best_score_hparams and "length_penalty" in best_score_hparams[model_dir]:
        model_conf["length_penalty"] = best_score_hparams[model_dir]["length_penalty"]
    else:
        model_conf["length_penalty"] = 1.0

    # 打印生成信息，输出模型配置文件名变量的值
    print(f"Generating {fsmt_model_config_file}")
    # 使用 UTF-8 编码打开文件，并将模型配置信息写入文件
    with open(fsmt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))

    # tokenizer config
    # 构建 tokenizer 配置文件路径
    fsmt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)

    # 配置 tokenizer 参数
    tokenizer_conf = {
        "langs": [src_lang, tgt_lang],
        "model_max_length": 1024,
        "do_lower_case": do_lower_case,
    }

    # 打印生成信息，输出 tokenizer 配置文件名变量的值
    print(f"Generating {fsmt_tokenizer_config_file}")
    # 打开文件 `fsmt_tokenizer_config_file` 以写入模式，编码为 UTF-8
    with open(fsmt_tokenizer_config_file, "w", encoding="utf-8") as f:
        # 将 `tokenizer_conf` 对象转换为 JSON 格式并写入文件
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))

    # 从 `chkpt` 字典中获取第一个模型，并获取其状态字典
    model = chkpt["models"][0]
    model_state_dict = model.state_dict()

    # 将模型状态字典中的键名加上前缀 'model.'
    model_state_dict = OrderedDict(("model." + k, v) for k, v in model_state_dict.items())

    # 移除不需要的键名
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        "model.encoder_embed_tokens.weight",
        "model.decoder_embed_tokens.weight",
        "model.encoder.embed_positions._float_tensor",
        "model.decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        # 从模型状态字典中移除对应的键名
        model_state_dict.pop(k, None)

    # 从指定路径 `pytorch_dump_folder_path` 加载 FSMT 模型配置
    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    # 基于加载的配置创建一个新的 FSMT 模型
    model_new = FSMTForConditionalGeneration(config)

    # 非严格模式加载模型状态字典到 `model_new`
    model_new.load_state_dict(model_state_dict, strict=False)

    # 设置用于保存权重的路径
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    # 打印保存路径信息
    print(f"Generating {pytorch_weights_dump_path}")
    # 使用 Torch 保存模型状态字典到指定路径
    torch.save(model_state_dict, pytorch_weights_dump_path)

    # 打印转换完成信息
    print("Conversion is done!")
    # 打印下一步上传文件到 S3的指引
    print("\nLast step is to upload the files to s3")
    # 打印进入 `data_root` 目录的指引
    print(f"cd {data_root}")
    # 使用 `transformers-cli` 工具上传 `model_dir` 到 Hugging Face 模型库
    print(f"transformers-cli upload {model_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建一个命令行参数解析器对象

    # Required parameters
    parser.add_argument(
        "--fsmt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts,"
            " bpecodes, etc."
        ),
    )
    # 添加一个必需的命令行参数 --fsmt_checkpoint_path，用于指定官方PyTorch检查点文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加另一个必需的命令行参数 --pytorch_dump_folder_path，用于指定输出的PyTorch模型的路径

    args = parser.parse_args()
    # 解析命令行参数并将其存储在args对象中

    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
    # 调用函数convert_fsmt_checkpoint_to_pytorch，传入解析后得到的检查点文件路径和输出模型路径作为参数
```