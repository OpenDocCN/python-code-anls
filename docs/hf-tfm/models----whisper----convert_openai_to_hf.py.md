# `.\transformers\models\whisper\convert_openai_to_hf.py`

```py
#!/usr/bin/env python
"""Converts a Whisper model in OpenAI format to Hugging Face format."""
# Copyright 2022 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
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

# 导入所需的模块和库
import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any, Optional, Tuple

import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm

from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available

# 定义 Whisper 模型的可用版本及其下载地址
_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

# 定义可用的分词器
_TOKENIZERS = {
    # 多语言模型的下载链接
    "multilingual": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    # 英语模型的下载链接
    "english": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
}
# 上面是一个单独的右大括号，可能是代码块的结束或者是某些条件语句的闭合

def _get_generation_config(
    is_multilingual: bool,
    num_languages: int = 100,
    openai_version: Optional[str] = None,
) -> GenerationConfig:
    """
    Loads the appropriate generation config from HF repo
    """
    # 根据条件确定模型配置文件的来源
    if openai_version is not None:
        repo = f"openai/whisper-{openai_version}"
    elif not is_multilingual:
        repo = "openai/whisper-medium.en"
    elif num_languages < 100:
        repo = "openai/whisper-large-v2"
    else:
        repo = "openai/whisper-large-v3"

    # 从预训练模型中加载生成配置
    gen_cfg = GenerationConfig.from_pretrained(repo)
    # 如果没有指定 OpenAI 版本，则警告用户对齐头未包含在生成配置中
    if openai_version is None:
        gen_cfg.alignment_heads = None
        warnings.warn(
            "Alignment heads have not been included in the generation config, since they are available "
            "only for the original OpenAI checkpoints."
            "If you want to use word-level timestamps with a custom version of Whisper,"
            "see https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb"
            "for the example of how to produce word-level timestamps manually."
        )

    return gen_cfg


def remove_ignore_keys_(state_dict):
    # 要从状态字典中移除的键列表
    ignore_keys = ["layers", "blocks"]
    # 遍历要忽略的键，如果存在于状态字典中，则将其移除
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
    # 映射旧键到新键的字典
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}

def rename_keys(s_dict):
    # 获取状态字典的键列表
    keys = list(s_dict.keys())
    # 遍历状态字典的键
    for key in keys:
        new_key = key
        # 遍历 WHISPER_MAPPING 字典，将键中的旧键替换为新键
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        # 打印出旧键和新键的对应关系
        print(f"{key} -> {new_key}")

        # 用新键替换旧键
        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def make_linear_from_emb(emb):
    # 从嵌入层创建线性层
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _download(url: str, root: str) -> Any:
    # 确保下载目录存在
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    # 检查下载目标是否存在并且不是文件
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    # 检查下载目标是否为文件
    if os.path.isfile(download_target):
        # 读取目标文件的二进制内容
        model_bytes = open(download_target, "rb").read()
        # 检查文件的 SHA256 哈希值是否与预期值匹配
        if insecure_hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            # 如果匹配，返回通过 BytesIO 加载的 Torch 模型
            return torch.load(io.BytesIO(model_bytes))
        else:
            # 如果哈希值不匹配，发出警告并重新下载文件
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 使用 urllib 请求下载目标 URL 的内容，保存到下载目标路径
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 使用 tqdm 显示下载进度
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            # 循环读取源数据并写入到输出文件
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                # 更新 tqdm 进度条
                loop.update(len(buffer))

    # 重新读取下载的文件内容
    model_bytes = open(download_target, "rb").read()
    # 检查下载的模型文件的哈希值是否与预期值匹配
    if insecure_hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        # 如果哈希值不匹配，抛出运行时错误
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    # 返回通过 BytesIO 加载的 Torch 模型
    return torch.load(io.BytesIO(model_bytes))
# 将 OpenAI 的 Whisper 模型转换为 TensorFlow 模型
def convert_openai_whisper_to_tfms(
    checkpoint_path, pytorch_dump_folder_path
) -> Tuple[WhisperForConditionalGeneration, bool, int]:
    # 如果文件名不包含 ".pt"，则下载模型文件
    if ".pt" not in checkpoint_path:
        # 如果没有指定文件夹，则将模型文件下载到当前目录
        root = os.path.dirname(pytorch_dump_folder_path) or "."
        original_checkpoint = _download(_MODELS[checkpoint_path], root)
        openai_version = checkpoint_path
    else:
        # 如果文件名包含 ".pt"，则加载模型文件
        original_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        openai_version = None

    # 提取模型维度信息和模型状态字典
    dimensions = original_checkpoint["dims"]
    state_dict = original_checkpoint["model_state_dict"]
    proj_out_weights = state_dict["decoder.token_embedding.weight"]
    # 移除特定的键并重命名
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    # 设置 tie_embeds 为 True，并提取 FFN 维度
    tie_embeds = True
    ffn_dim = state_dict["decoder.layers.0.fc1.weight"].shape[0]

    # 设置特殊标记的 ID，如果词汇表大小超过 51865，则使用 50257 否则使用 50256
    endoftext_id = 50257 if dimensions["n_vocab"] > 51865 else 50256

    # 配置 Whisper 模型参数
    config = WhisperConfig(
        vocab_size=dimensions["n_vocab"],
        encoder_ffn_dim=ffn_dim,
        decoder_ffn_dim=ffn_dim,
        num_mel_bins=dimensions["n_mels"],
        d_model=dimensions["n_audio_state"],
        max_target_positions=dimensions["n_text_ctx"],
        encoder_layers=dimensions["n_audio_layer"],
        encoder_attention_heads=dimensions["n_audio_head"],
        decoder_layers=dimensions["n_text_layer"],
        decoder_attention_heads=dimensions["n_text_head"],
        max_source_positions=dimensions["n_audio_ctx"],
        eos_token_id=endoftext_id,
        bos_token_id=endoftext_id,
        pad_token_id=endoftext_id,
        decoder_start_token_id=endoftext_id + 1,
    )

    # 创建 WhisperForConditionalGeneration 模型
    model = WhisperForConditionalGeneration(config)
    # 加载模型参数
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    # 如果有缺失的参数，且不在允许的列表中，则抛出 ValueError
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    # 如果 tie_embeds 为 True，则设置模型的 proj_out 属性
    if tie_embeds:
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.proj_out.weight.data = proj_out_weights

    # 确定模型是否多语言，并计算语言数量
    is_multilingual = model.config.vocab_size >= 51865
    num_languages = model.config.vocab_size - 51765 - int(is_multilingual)

    # 从模型检查点确定这些参数，与 Whisper 仓库保持一致
    model.generation_config = _get_generation_config(
        is_multilingual,
        num_languages,
        openai_version,
    )

    # 返回转换后的模型、是否多语言和语言数量
    return model, is_multilingual, num_languages


# 从 https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960 改编
# 将字节编码的 token 进行 BPE 处理
def _bpe(mergeable_ranks, token: bytes, max_rank=None) -> list[bytes]:
    # 将字节编码的 token 拆分为单字节
    parts = [bytes([b]) for b in token]
    # 这个循环不断地合并可合并的相邻字符串，直到无法继续合并为止
    while True:
        # 初始化最小索引和最小合并等级为 None
        min_idx = None
        min_rank = None
        # 遍历相邻字符串对
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # 获取当前相邻字符串对的合并等级
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # 如果合并等级有效且小于当前最小合并等级，更新最小索引和最小合并等级
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        # 如果无法找到可合并的相邻字符串对，或者最小合并等级大于等于最大合并等级，退出循环
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        # 断言找到了可合并的相邻字符串对
        assert min_idx is not None
        # 合并相邻字符串对，并更新 parts 列表
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    # 返回合并后的 parts 列表
    return parts
# 将 TikToken BPE 转换为 Hugging Face 格式
def convert_tiktoken_bpe_to_hf(tiktoken_url: str):
    # 加载 TikToken BPE 数据
    bpe_ranks = load_tiktoken_bpe(tiktoken_url)
    # 创建将字节转换为 Unicode 的函数
    byte_encoder = bytes_to_unicode()

    # 定义将 token 的字节转换为字符串的函数
    def token_bytes_to_string(b):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    # 初始化 merges 和 vocab 字典
    merges = []
    vocab = {}
    # 遍历 token 和 rank，将 token 的字符串形式和 rank 添加到 vocab 字典中
    for token, rank in bpe_ranks.items():
        vocab[token_bytes_to_string(token)] = rank
        # 如果 token 的长度为 1，则跳过后续步骤
        if len(token) == 1:
            continue
        # 合并 token，获取新的 tokens
        merged = tuple(_bpe(bpe_ranks, token, max_rank=rank))
        # 若合并的 token 长度为 2，添加到 merges 中
        if len(merged) == 2:
            merges.append(" ".join(map(token_bytes_to_string, merged)))
    # 返回 vocab 字典和 merges 列表
    return vocab, merges


# 将 TikToken 转换为 Hugging Face 格式
def convert_tiktoken_to_hf(
    multilingual: bool = True, num_languages: int = 100, time_precision=0.02
) -> WhisperTokenizer:
    # 获取 TikToken tokenizer 路径
    tiktoken_tokenizer_path = _TOKENIZERS["multilingual" if multilingual else "english"]
    # 定义特殊 token
    start_of_transcript = ["<|endoftext|>", "<|startoftranscript|>"]
    control_tokens = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    # 生成语言特定 token 列表
    language_tokens = [f"<|{k}|>" for k in list(LANGUAGES)[:num_languages]]
    # 生成时间戳 token 列表
    timestamp_tokens = [("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)]

    # 转换 TikToken BPE 到 Hugging Face 格式
    vocab, merges = convert_tiktoken_bpe_to_hf(tiktoken_tokenizer_path)

    # 使用临时文件夹来创建 vocab 和 merges 文件
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_file = f"{tmpdirname}/vocab.json"
        merge_file = f"{tmpdirname}/merges.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens in merges:
                writer.write(bpe_tokens + "\n")

        # 创建 WhisperTokenizer 对象
        hf_tokenizer = WhisperTokenizer(vocab_file, merge_file)

    # 将特殊 token 和时间戳 token 添加到 tokenizer 中
    hf_tokenizer.add_tokens(start_of_transcript + language_tokens + control_tokens, special_tokens=True)
    hf_tokenizer.add_tokens(timestamp_tokens, special_tokens=False)
    # 返回 Hugging Face 格式的 tokenizer
    return hf_tokenizer


# 主函数入口
if __name__ == "__main__":
    # 创建命令行参数解析对象
    parser = argparse.ArgumentParser()
    # 添加必要的参数
    parser.add_argument("--checkpoint_path", type=str, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 转换 OpenAI Whisper 到 TensorFlow Models
    model, is_multilingual, num_languages = convert_openai_whisper_to_tfms(
        args.checkpoint_path, args.pytorch_dump_folder_path
    )
    # 如果设置了转换预处理器参数
    if args.convert_preprocessor:
        # 尝试检查是否已安装`tiktoken`包，若未安装则抛出异常提示安装
        try:
            if not _is_package_available("tiktoken"):
                raise """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
        # 捕获任何异常，不中断程序执行
        except Exception:
            pass
        # 如果检查到`tiktoken`包可用
        else:
            # 从`tiktoken.load`模块中导入`load_tiktoken_bpe`函数
            from tiktoken.load import load_tiktoken_bpe

            # 转换`tiktoken`到`hf`（Hugging Face）格式
            tokenizer = convert_tiktoken_to_hf(is_multilingual, num_languages)
            # 创建`WhisperFeatureExtractor`实例
            feature_extractor = WhisperFeatureExtractor(
                feature_size=model.config.num_mel_bins,
                # 其余默认参数与`openai/whisper`中硬编码的参数相同
            )
            # 创建`WhisperProcessor`实例
            processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            # 将处理器保存到指定路径
            processor.save_pretrained(args.pytorch_dump_folder_path)

            # 也保存快速分词器
            # 从指定路径加载`WhisperTokenizerFast`实例
            fast_tokenizer = WhisperTokenizerFast.from_pretrained(args.pytorch_dump_folder_path)
            # 将快速分词器保存到指定路径，使用新格式
            fast_tokenizer.save_pretrained(args.pytorch_dump_folder_path, legacy_format=False)

    # 保存模型到指定路径
    model.save_pretrained(args.pytorch_dump_folder_path)
```