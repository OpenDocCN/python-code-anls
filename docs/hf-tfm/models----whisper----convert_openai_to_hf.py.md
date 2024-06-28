# `.\models\whisper\convert_openai_to_hf.py`

```py
#!/usr/bin/env python
"""Converts a Whisper model in OpenAI format to Hugging Face format."""
# 版本和许可声明
# 版权 2022 年由 Hugging Face Inc. 团队和 OpenAI 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

import argparse  # 导入命令行参数解析库
import io  # 导入用于处理字节流的库
import json  # 导入 JSON 格式处理库
import os  # 导入操作系统相关的功能库
import tempfile  # 导入临时文件和目录创建库
import urllib  # 导入处理 URL 的库
import warnings  # 导入警告处理库
from typing import Any, Optional, Tuple  # 引入类型提示支持

import torch  # 导入 PyTorch 深度学习库
from huggingface_hub.utils import insecure_hashlib  # 导入 Hugging Face Hub 的哈希函数支持
from torch import nn  # 导入 PyTorch 的神经网络模块
from tqdm import tqdm  # 导入进度条显示库

from transformers import (  # 导入 Hugging Face Transformers 库中的多个模块和类
    GenerationConfig,  # 生成配置类
    WhisperConfig,  # Whisper 模型配置类
    WhisperFeatureExtractor,  # Whisper 特征提取器类
    WhisperForConditionalGeneration,  # Whisper 条件生成模型类
    WhisperProcessor,  # Whisper 处理器类
    WhisperTokenizer,  # Whisper 分词器类
    WhisperTokenizerFast,  # 快速版 Whisper 分词器类
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode  # 导入 Whisper 分词相关的常量和函数
from transformers.utils.import_utils import _is_package_available  # 导入 Hugging Face Transformers 的包是否可用函数

_MODELS = {  # 预定义 Whisper 模型的下载链接字典
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

_TOKENIZERS = {  # 预定义 Whisper 分词器的配置字典
    # 定义一个包含两个键值对的字典，用于存储语言模型的不同版本的 Token 文件的 URL
    "multilingual": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "english": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
}


def _get_generation_config(
    is_multilingual: bool,
    num_languages: int = 100,
    openai_version: Optional[str] = None,
) -> GenerationConfig:
    """
    Loads the appropriate generation config from HF repo based on provided parameters.

    Args:
        is_multilingual (bool): Flag indicating if multilingual model is used.
        num_languages (int, optional): Number of languages for the model (default is 100).
        openai_version (Optional[str], optional): Version of OpenAI model to load (default is None).

    Returns:
        GenerationConfig: Config object for generation.
    """
    if openai_version is not None:
        repo = f"openai/whisper-{openai_version}"
    elif not is_multilingual:
        repo = "openai/whisper-medium.en"
    elif num_languages < 100:
        repo = "openai/whisper-large-v2"
    else:
        repo = "openai/whisper-large-v3"

    gen_cfg = GenerationConfig.from_pretrained(repo)

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
    """
    Remove specific keys from the provided state_dict.

    Args:
        state_dict (dict): Dictionary containing the model's state.

    Returns:
        None
    """
    ignore_keys = ["layers", "blocks"]
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
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
    """
    Rename keys in the provided dictionary according to pre-defined mapping.

    Args:
        s_dict (dict): Dictionary whose keys need to be renamed.

    Returns:
        dict: Dictionary with renamed keys.
    """
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def make_linear_from_emb(emb):
    """
    Create a linear layer from an embedding layer.

    Args:
        emb (nn.Embedding): Embedding layer.

    Returns:
        nn.Linear: Linear layer initialized with the same weights as the embedding.
    """
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _download(url: str, root: str) -> Any:
    """
    Download a file from a URL to a specified directory.

    Args:
        url (str): URL of the file to download.
        root (str): Directory where the file should be saved.

    Returns:
        Any: Not explicitly returned value.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    # 如果下载目标文件已存在
    if os.path.isfile(download_target):
        # 读取下载目标文件的全部内容
        model_bytes = open(download_target, "rb").read()
        # 使用不安全的哈希算法计算文件内容的 SHA256 值，并检查是否与预期的哈希值匹配
        if insecure_hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            # 如果匹配，则将文件内容作为字节流加载为 Torch 模型并返回
            return torch.load(io.BytesIO(model_bytes))
        else:
            # 如果不匹配，则发出警告，提示哈希值不匹配，需要重新下载文件
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 使用 urllib 请求下载指定 URL 的文件，保存到 download_target
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 使用 tqdm 显示下载进度，设置总大小、显示宽度、单位等参数
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                # 从网络源读取数据块到缓冲区
                buffer = source.read(8192)
                # 如果缓冲区为空则退出循环
                if not buffer:
                    break

                # 将读取的数据块写入到输出文件
                output.write(buffer)
                # 更新 tqdm 进度条，增加已写入数据块的大小
                loop.update(len(buffer))

    # 重新读取下载后的目标文件内容
    model_bytes = open(download_target, "rb").read()
    # 再次使用不安全的哈希算法计算文件内容的 SHA256 值，并检查是否与预期的哈希值匹配
    if insecure_hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        # 如果不匹配，则抛出运行时错误，提示下载的模型文件哈希值不匹配，需要重新尝试加载模型
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not match. Please retry loading the model."
        )

    # 将文件内容作为字节流加载为 Torch 模型并返回
    return torch.load(io.BytesIO(model_bytes))
# 从 OpenAI Whisper 模型的检查点转换为 Transformers 模型格式
def convert_openai_whisper_to_tfms(
    checkpoint_path, pytorch_dump_folder_path
) -> Tuple[WhisperForConditionalGeneration, bool, int]:
    # 检查检查点文件路径是否以 ".pt" 结尾，若不是则下载原始检查点
    if ".pt" not in checkpoint_path:
        # 获取 pytorch_dump_folder_path 的父目录作为下载位置，如果不存在则使用当前目录
        root = os.path.dirname(pytorch_dump_folder_path) or "."
        # 下载指定模型的原始检查点文件
        original_checkpoint = _download(_MODELS[checkpoint_path], root)
        # 获取 OpenAI 模型的版本号
        openai_version = checkpoint_path
    else:
        # 使用 torch.load 加载指定路径的 PyTorch 检查点文件到 CPU
        original_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        openai_version = None

    # 从原始检查点中获取维度信息
    dimensions = original_checkpoint["dims"]
    # 获取模型的状态字典
    state_dict = original_checkpoint["model_state_dict"]
    # 获取解码器的 token embedding 权重
    proj_out_weights = state_dict["decoder.token_embedding.weight"]
    # 移除忽略的键值对
    remove_ignore_keys_(state_dict)
    # 重命名模型中的键名
    rename_keys(state_dict)
    # 标志是否绑定 token embedding
    tie_embeds = True
    # 获取解码器中第一个全连接层的维度
    ffn_dim = state_dict["decoder.layers.0.fc1.weight"].shape[0]

    # 通过判断 vocab 大小来设置 bos/eos/pad token 的 id
    endoftext_id = 50257 if dimensions["n_vocab"] > 51865 else 50256

    # 创建 WhisperConfig 对象，配置模型的参数
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

    # 创建 WhisperForConditionalGeneration 模型对象
    model = WhisperForConditionalGeneration(config)
    # 加载模型的状态字典，并检查是否有丢失的参数
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    # 如果有丢失的参数且不在允许的缺失列表中，则抛出 ValueError
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    # 如果 tie_embeds 为 True，则从 embed_tokens 创建线性投影层
    if tie_embeds:
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        # 否则直接使用给定的 proj_out_weights 作为投影层的权重
        model.proj_out.weight.data = proj_out_weights

    # 根据模型检查点确定模型的生成配置，参考 Whisper 代码库的实现
    is_multilingual = model.config.vocab_size >= 51865
    num_languages = model.config.vocab_size - 51765 - int(is_multilingual)

    # 设置模型的生成配置
    model.generation_config = _get_generation_config(
        is_multilingual,
        num_languages,
        openai_version,
    )

    # 返回转换后的模型对象、是否多语言模型和语言数量
    return model, is_multilingual, num_languages


# 从 https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960 适配而来
def _bpe(mergeable_ranks, token: bytes, max_rank=None) -> list[bytes]:
    # 将字节型 token 拆分为单独的字节部分
    parts = [bytes([b]) for b in token]
    # 返回拆分后的字节列表
    return parts
    # 进入无限循环，直到不再能够合并的情况
    while True:
        # 初始化最小索引和最小合并等级
        min_idx = None
        min_rank = None
        # 遍历 parts 列表中每对相邻元素的索引和元素组成的元组
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # 获取当前两个元素可以合并的等级
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # 如果可以合并的等级存在，并且比当前的最小等级小，则更新最小索引和最小等级
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        # 如果找不到可以合并的等级，或者当前最小等级大于等于指定的最大等级，则跳出循环
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        # 确保最小索引不为空
        assert min_idx is not None
        # 合并 parts 列表中最小索引处的两个元素，并更新 parts 列表
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    # 返回合并完成后的 parts 列表
    return parts
def convert_tiktoken_bpe_to_hf(tiktoken_url: str):
    # 载入指定 URL 的 TikToken BPE 排名数据
    bpe_ranks = load_tiktoken_bpe(tiktoken_url)
    # 创建字节到 Unicode 字符的映射
    byte_encoder = bytes_to_unicode()

    # 将字节表示的 Token 转换为字符串
    def token_bytes_to_string(b):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    # 初始化空列表和字典以保存词汇表和合并规则
    merges = []
    vocab = {}

    # 遍历 BPE 排名数据，将 Token 和其排名转换为字符串形式加入词汇表
    for token, rank in bpe_ranks.items():
        vocab[token_bytes_to_string(token)] = rank
        # 如果 Token 长度为 1，跳过后续步骤
        if len(token) == 1:
            continue
        # 通过 _bpe 函数获取合并后的 Token 对，并将其转换为字符串形式后加入合并规则列表
        merged = tuple(_bpe(bpe_ranks, token, max_rank=rank))
        if len(merged) == 2:  # 考虑空 Token
            merges.append(" ".join(map(token_bytes_to_string, merged)))

    # 返回词汇表和合并规则列表
    return vocab, merges


def convert_tiktoken_to_hf(
    multilingual: bool = True, num_languages: int = 100, time_precision=0.02
) -> WhisperTokenizer:
    # 根据多语言选项确定使用的 TikToken 文件路径
    tiktoken_tokenizer_path = _TOKENIZERS["multilingual" if multilingual else "english"]
    # 定义转录开始标记和控制标记列表
    start_of_transcript = ["<|endoftext|>", "<|startoftranscript|>"]
    control_tokens = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    # 定义非标准化的语言标记列表
    language_tokens = [f"<|{k}|>" for k in list(LANGUAGES)[:num_languages]]
    # 定义标准化的时间戳标记列表
    timestamp_tokens = [("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)]

    # 转换 TikToken 到 Hugging Face 格式的词汇表和合并规则
    vocab, merges = convert_tiktoken_bpe_to_hf(tiktoken_tokenizer_path)

    # 使用临时目录创建词汇表和合并规则文件，并初始化 WhisperTokenizer 对象
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_file = f"{tmpdirname}/vocab.json"
        merge_file = f"{tmpdirname}/merges.txt"
        
        # 将词汇表写入 JSON 文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 将合并规则写入文本文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens in merges:
                writer.write(bpe_tokens + "\n")

        # 初始化 WhisperTokenizer 对象，加载词汇表和合并规则文件
        hf_tokenizer = WhisperTokenizer(vocab_file, merge_file)

    # 向 WhisperTokenizer 对象添加特殊标记和时间戳标记
    hf_tokenizer.add_tokens(start_of_transcript + language_tokens + control_tokens, special_tokens=True)
    hf_tokenizer.add_tokens(timestamp_tokens, special_tokens=False)
    
    # 返回初始化后的 WhisperTokenizer 对象
    return hf_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument("--checkpoint_path", type=str, help="下载的检查点的路径")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="输出 PyTorch 模型的路径.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="是否将预处理器（分词器 + 特征提取器）与模型一起转换.",
    )
    args = parser.parse_args()

    # 转换 OpenAI Whisper 到 TensorFlow Model Specification
    model, is_multilingual, num_languages = convert_openai_whisper_to_tfms(
        args.checkpoint_path, args.pytorch_dump_folder_path
    )
    # 如果命令行参数中包含 convert_preprocessor 标志
    if args.convert_preprocessor:
        try:
            # 检查是否安装了 `tiktoken` 包
            if not _is_package_available("tiktoken"):
                # 如果未安装，抛出异常提醒用户安装 `tiktoken`
                raise """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
        except Exception:
            # 捕获任何异常，不进行处理，继续执行后续代码
            pass
        else:
            # 如果没有抛出异常，导入 `load_tiktoken_bpe` 函数
            from tiktoken.load import load_tiktoken_bpe

            # 根据条件调用 convert_tiktoken_to_hf 函数生成 tokenizer
            tokenizer = convert_tiktoken_to_hf(is_multilingual, num_languages)
            # 创建 WhisperFeatureExtractor 实例，设置特征大小为模型配置的 num_mel_bins 值
            feature_extractor = WhisperFeatureExtractor(
                feature_size=model.config.num_mel_bins,
                # 其余默认参数与 openai/whisper 中硬编码的相同
            )
            # 使用 tokenizer 和 feature_extractor 创建 WhisperProcessor 实例
            processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            # 将 processor 的预训练结果保存到指定的 pytorch_dump_folder_path 中
            processor.save_pretrained(args.pytorch_dump_folder_path)

            # 同时保存快速 tokenizer
            fast_tokenizer = WhisperTokenizerFast.from_pretrained(args.pytorch_dump_folder_path)
            fast_tokenizer.save_pretrained(args.pytorch_dump_folder_path, legacy_format=False)

    # 将模型的预训练结果保存到指定的 pytorch_dump_folder_path 中
    model.save_pretrained(args.pytorch_dump_folder_path)
```