# `.\models\fsmt\convert_fsmt_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置脚本的编码格式为 utf-8
# 版权声明，此代码版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0，除非符合许可要求，否则不得使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非法律规定或书面同意，否则根据许可分发的软件基于 "AS IS" 基础，不带任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言规定的权限和限制

# 请注意：如果您打算运行此脚本，请确保在 scripts/fsmt/ 目录下查找适当的脚本，以正确完成工作
# 有一套脚本可用于：
# - 下载和准备数据以及运行转换脚本
# - 执行评估以获得配置文件中的最佳超参数
# - 生成模型卡片 - 如果您有同一论文中的多个模型，则非常有用

import argparse
import json
import os
import re
from collections import OrderedDict
from os.path import basename, dirname

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import FSMTConfig, FSMTForConditionalGeneration
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging

# 设置日志级别为警告
logging.set_verbosity_warning()

# 定义 JSON 缩进空格数
json_indent = 2

# 根据在一系列 `num_beams`、`length_penalty` 和 `early_stopping` 上的搜索结果，
# 针对 wmt19 测试数据获得最佳 BLEU 分数，我们将使用以下默认值：
#
# * `num_beams`: 5（得分更高，但需要更多内存/较慢，可以由用户调整）
# * `early_stopping`: `False` 一贯获得更好的得分
# * `length_penalty` 有所变化，因此根据模型分配最佳值
best_score_hparams = {
    # fairseq:
    "wmt19-ru-en": {"length_penalty": 1.1},
    "wmt19-en-ru": {"length_penalty": 1.15},
    "wmt19-en-de": {"length_penalty": 1.0},
    "wmt19-de-en": {"length_penalty": 1.1},
    # allenai:
    "wmt16-en-de-dist-12-1": {"length_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"length_penalty": 0.6},
    "wmt16-en-de-12-1": {"length_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"length_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"length_penalty": 0.6},
}

# 将不同模型重新映射为它们的组织名称
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

# 重新编写字典键
def rewrite_dict_keys(d):
    # (1) 删除单词分割符号，(2) 在单词未分割的地方添加单词结束符号
    # 对字典中的key进行处理，将以@@结尾的key去掉@@，将不以@@结尾的key末尾加上"</w>"，并将处理后的键值对重新组成字典
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    # 定义需要保留的特殊tokens
    keep_keys = "<s> <pad> </s> <unk>".split()
    # 恢复特殊tokens
    for k in keep_keys:
        # 删除处理后的字典中的特殊tokens对应的key
        del d2[f"{k}</w>"]
        # 将原始字典中的特殊tokens对应的键值对加入到处理后的字典中
        d2[k] = d[k]  # 恢复
    # 返回处理后的字典
    return d2
# 将 Fairseq 模型的检查点转换为 PyTorch 格式
def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):
    # 检查 Fairseq 检查点文件是否存在
    assert os.path.exists(fsmt_checkpoint_path)
    # 创建目标文件夹，如果不存在则创建
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    # 打印将结果写入的目录路径
    print(f"Writing results to {pytorch_dump_folder_path}")

    # 处理各种类型的模型

    # 获取检查点文件名和所在文件夹路径
    checkpoint_file = basename(fsmt_checkpoint_path)
    fsmt_folder_path = dirname(fsmt_checkpoint_path)

    # 使用 fairseq 提供的模型并指定参数
    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    models = cls.hub_models()
    kwargs = {"bpe": "fastbpe", "tokenizer": "moses"}
    data_name_or_path = "."
    # 注意：由于模型转储是旧的，fairseq 后来升级了其模型，它对保存的权重进行了大量重写和拆分，
    # 因此我们不能直接在模型文件上使用 torch.load()。参见：fairseq_model.py 中的 upgrade_state_dict(state_dict)
    print(f"using checkpoint {checkpoint_file}")
    # 从预训练模型中加载检查点
    chkpt = hub_utils.from_pretrained(
        fsmt_folder_path, checkpoint_file, data_name_or_path, archive_map=models, **kwargs
    )

    # 获取模型参数
    args = vars(chkpt["args"]["model"])

    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]

    data_root = dirname(pytorch_dump_folder_path)
    model_dir = basename(pytorch_dump_folder_path)

    # 词典文件
    src_dict_file = os.path.join(fsmt_folder_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_folder_path, f"dict.{tgt_lang}.txt")

    # 加载源语言词典并重写词典键
    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-src.json")
    print(f"Generating {src_vocab_file} of {src_vocab_size} of {src_lang} records")
    # 将源语言词汇表写入 JSON 文件
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # 检测是否需要进行小写处理，可以通过检查源语言词汇表中是否有大写字母来推断
    do_lower_case = True
    for k in src_vocab.keys():
        if not k.islower():
            do_lower_case = False
            break

    # 加载目标语言词典并重写词典键
    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    tgt_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-tgt.json")
    print(f"Generating {tgt_vocab_file} of {tgt_vocab_size} of {tgt_lang} records")
    # 将目标语言词汇表写入 JSON 文件
    with open(tgt_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))

    # 合并文件（bpecodes）
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    for fn in ["bpecodes", "code"]:  # 旧版本的 fairseq 将合并文件称为 "code"
        fsmt_merges_file = os.path.join(fsmt_folder_path, fn)
        if os.path.exists(fsmt_merges_file):
            break
    # 打开文件以读取内容，使用 utf-8 编码
    with open(fsmt_merges_file, encoding="utf-8") as fin:
        # 读取文件内容到 merges 变量中
        merges = fin.read()
    # 使用正则表达式将每行结尾的频率数字去除
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    # 打印消息，指示正在生成 merges_file 文件
    print(f"Generating {merges_file}")
    # 打开 merges_file 文件以写入 merges 变量中的内容，使用 utf-8 编码
    with open(merges_file, "w", encoding="utf-8") as fout:
        # 将 merges 变量中的内容写入文件中
        fout.write(merges)
    
    # model config
    # 设置文件路径，指向模型配置文件
    fsmt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")
    
    # 验证 bpe/tokenizer 配置，当前硬编码为 moses+fastbpe -
    # 如果未来模型使用不同类型的 tokenizer，则可能需要修改 tokenizer
    # 检查是否需要扩展 tokenizer 以支持当前的 bpe 类型
    assert args["bpe"] == "fastbpe", f"need to extend tokenizer to support bpe={args['bpe']}"
    assert args["tokenizer"] == "moses", f"need to extend tokenizer to support bpe={args['tokenizer']}"
    
    # 设置模型配置
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
    
    # 设置模型默认的超参数
    model_conf["num_beams"] = 5
    model_conf["early_stopping"] = False
    # 如果模型目录在 best_score_hparams 中且有 length_penalty，则使用该值，否则默认为 1.0
    if model_dir in best_score_hparams and "length_penalty" in best_score_hparams[model_dir]:
        model_conf["length_penalty"] = best_score_hparams[model_dir]["length_penalty"]
    else:
        model_conf["length_penalty"] = 1.0
    
    # 打印消息，指示正在生成 fsmt_model_config_file 文件
    print(f"Generating {fsmt_model_config_file}")
    # 打开 fsmt_model_config_file 文件以写入模型配置内容，使用 utf-8 编码
    with open(fsmt_model_config_file, "w", encoding="utf-8") as f:
        # 将模型配置字典转换为 JSON 格式并写入文件
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))
    
    # tokenizer config
    # 设置文件路径，指向 tokenizer 配置文件
    fsmt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)
    
    # 设置 tokenizer 配置
    tokenizer_conf = {
        "langs": [src_lang, tgt_lang],
        "model_max_length": 1024,
        "do_lower_case": do_lower_case,
    }
    
    # 打印消息，指示正在生成 fsmt_tokenizer_config_file 文件
    print(f"Generating {fsmt_tokenizer_config_file}")
    # 写入tokenizer配置文件
    with open(fsmt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))

    # 获取模型信息
    model = chkpt["models"][0]
    model_state_dict = model.state_dict()

    # 重命名键值，使其以'model.'开头
    model_state_dict = OrderedDict(("model." + k, v) for k, v in model_state_dict.items())

    # 移除不需要的键
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
        model_state_dict.pop(k, None)

    # 创建新的FSMT模型并加载状态字典
    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = FSMTForConditionalGeneration(config)
    model_new.load_state_dict(model_state_dict, strict=False)

    # 保存模型权重
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    # 输出提示信息
    print("Conversion is done!")
    print("\nLast step is to upload the files to s3")
    print(f"cd {data_root}")
    print(f"transformers-cli upload {model_dir}")
# 如果当前脚本被当作主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必选参数
    parser.add_argument(
        "--fsmt_checkpoint_path",  # 参数名：--fsmt_checkpoint_path，表示Fairseq-Style MT模型的检查点文件的路径
        default=None,  # 默认值为None
        type=str,  # 参数类型为字符串
        required=True,  # 参数是必选的
        help=(
            "Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts,"  # 帮助信息
            " bpecodes, etc."
        ),
    )
    # 添加必选参数
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 参数名：--pytorch_dump_folder_path，表示输出PyTorch模型的文件夹路径
        default=None,  # 默认值为None
        type=str,  # 参数类型为字符串
        required=True,  # 参数是必选的
        help="Path to the output PyTorch model."  # 帮助信息
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将Fairseq-Style MT模型的检查点文件转换为PyTorch模型
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
```