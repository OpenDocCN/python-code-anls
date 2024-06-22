# `.\transformers\models\wav2vec2_bert\convert_wav2vec2_seamless_checkpoint.py`

```py
# 导入必要的库和模块
import argparse
import torch
import torchaudio
from fairseq2.data import Collater
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel, logging

# 设置日志的输出级别为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义一个字典，用于映射源模型和目标模型的权重名称
wav2vec_convert_list = [
    ("encoder_frontend.model_dim_proj", "feature_projection.projection"),
    ("encoder_frontend.post_extract_layer_norm", "feature_projection.layer_norm"),
    ("encoder_frontend.pos_encoder.conv", "encoder.pos_conv_embed.conv"),
    ("encoder.inner.layers", "encoder.layers"),
    ("encoder.inner_layer_norm", "encoder.layer_norm"),
    ("encoder.adaptor_layers", "adapter.layers"),
    ("inner_proj", "intermediate_dense"),
    ("self_attn.output_proj", "self_attn.linear_out"),
    ("output_proj", "output_dense"),
    ("self_attn.k_proj", "self_attn.linear_k"),
    ("self_attn.v_proj", "self_attn.linear_v"),
    ("self_attn.q_proj", "self_attn.linear_q"),
    ("self_attn.sdpa.u_bias", "self_attn.pos_bias_u"),
    ("self_attn.sdpa.v_bias", "self_attn.pos_bias_v"),
    ("self_attn.sdpa.rel_k_embed", "self_attn.distance_embedding"),
    ("self_attn.sdpa.r_proj", "self_attn.linear_pos"),
    ("conv.pointwise_conv1", "conv_module.pointwise_conv1"),
    ("conv.pointwise_conv2", "conv_module.pointwise_conv2"),
    ("conv.depthwise_conv", "conv_module.depthwise_conv"),
    ("conv.layer_norm", "conv_module.depthwise_layer_norm"),
    ("conv_layer_norm", "conv_module.layer_norm"),
    ("encoder.proj1", "intermediate_ffn.intermediate_dense"),
    ("encoder.proj2", "intermediate_ffn.output_dense"),
    ("encoder.layer_norm", "inner_layer_norm"),
    ("masker.temporal_mask_embed", "masked_spec_embed"),
]

# 定义一个集合，用于存储需要删除的键
keys_to_remove = {
    "quantizer.entry_proj",
    "final_proj",
    "final_target_proj",
    "quantizer.entries",
    "quantizer.num_updates",
}

# 定义一个函数，用于计算模型的参数数量
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])

# 定义一个函数，用于转换模型权重
def _convert_model(
    original_model,
    hf_model,
    convert_list,
):
    state_dict = original_model.state_dict()


这段代码的主要功能是转换一个原始的 Wav2Vec2Bert 模型的权重到 Hugging Face 的 Wav2Vec2BertModel 模型。具体来说:

1. 导入必要的库和模块。
2. 设置日志的输出级别为 `info`。
3. 定义一个字典 `wav2vec_convert_list`，用于映射源模型和目标模型的权重名称。
4. 定义一个集合 `keys_to_remove`，用于存储需要删除的键。
5. 定义一个函数 `param_count`，用于计算模型的参数数量。
6. 定义一个函数 `_convert_model`，用于转换模型权重。

这个函数将在后续的代码中被调用。
    # 遍历状态字典的键值对列表
    for k, v in list(state_dict.items()):
        # 复制当前键，准备修改
        new_key = k
        # 遍历要转换的层名称对列表
        for old_layer_name, new_layer_name in convert_list:
            # 如果当前键包含旧层名称，则用新层名称替换
            if old_layer_name in new_key:
                new_key = new_key.replace(old_layer_name, new_layer_name)
    
        # 对 ".layer_norm" 的特殊处理，如果在新键中且前一个字符是数字，则替换为 "final_layer_norm"
        if ".layer_norm" in new_key and new_key.split(".layer_norm")[0][-1].isnumeric():
            new_key = new_key.replace("layer_norm", "final_layer_norm")
    
        # 检查是否应该将当前键添加到新的状态字典中
        add_key = True
        # 遍历需要移除的键列表
        for key in keys_to_remove:
            # 如果当前键包含需要移除的键，则将其从状态字典中移除，并设置 add_key 为 False
            if key in new_key:
                state_dict.pop(k)
                add_key = False
                break
    
        # 如果 add_key 为 True，则将当前键添加到新的状态字典中
        if add_key:
            state_dict[new_key] = state_dict.pop(k)
    
    # 计算状态字典中额外的键，即在模型预训练参数中没有的键
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    # 过滤掉不必要的参数，例如 "num_updates"
    extra_keys = set({k for k in extra_keys if "num_updates" not in k})
    # 计算模型预训练参数中缺失的键
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    
    # 如果存在额外的键，则抛出 ValueError 异常
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    # 如果存在缺失的键，则抛出 ValueError 异常
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    
    # 加载新的状态字典到模型中，严格检查键和形状是否匹配
    hf_model.load_state_dict(state_dict, strict=True)
    # 计算模型参数的数量
    n_params = param_count(hf_model)
    
    # 记录模型加载完成，并打印模型参数数量（以百万为单位）
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")
    
    # 将模型设置为评估模式
    hf_model.eval()
    # 删除状态字典以释放内存
    del state_dict
    
    # 返回加载完成的模型
    return hf_model
# 禁用 PyTorch 中的梯度计算
@torch.no_grad()
def convert_wav2vec2_bert_checkpoint(
    checkpoint_path,  # 检查点路径
    pytorch_dump_folder_path,  # 输出 PyTorch 模型路径
    config_path=None,  # 配置文件路径
    repo_id=None,  # 推送到 HuggingFace 仓库的 ID
):
    """
    将模型的权重复制/粘贴/调整到 transformers 设计中。
    """
    # 如果指定了配置文件路径，则读取配置并设置激活函数为 swish
    if config_path is not None:
        config = Wav2Vec2BertConfig.from_pretrained(config_path, hidden_act="swish")
    # 否则使用默认配置，不应用频谱扩增
    else:
        config = Wav2Vec2BertConfig(apply_spec_augment=False)

    # 创建 Wav2Vec2BertModel 实例
    hf_wav2vec = Wav2Vec2BertModel(config)

    # 加载检查点模型，设置数据类型为 float32
    model = load_conformer_shaw_model(checkpoint_path, dtype=torch.float32)
    # 将模型设置为评估模式
    model.eval()

    # 将检查点模型转换为 HuggingFace 模型
    hf_wav2vec = _convert_model(model, hf_wav2vec, wav2vec_convert_list)

    # 将转换后的模型保存到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了仓库 ID，则将模型推送到 HuggingFace 仓库
    if repo_id:
        hf_wav2vec.push_to_hub(repo_id, create_pr=True)

    # 保存特征提取器
    fe = SeamlessM4TFeatureExtractor(padding_value=1)
    fe._set_processor_class("Wav2Vec2BertProcessor")
    fe.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了仓库 ID，则将特征提取器推送到 HuggingFace 仓库
    if repo_id:
        fe.push_to_hub(repo_id, create_pr=True)

    # 如果指定了音频路径，则进行模型输出验证
    if args.audio_path:
        # 加载音频数据并重采样到特征提取器的采样率
        waveform, sample_rate = torchaudio.load(args.audio_path)
        waveform = torchaudio.functional.resample(waveform, sample_rate, fe.sampling_rate)

        # 创建 Fbank 转换器和 Collater 对象
        fbank_converter = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            dtype=torch.float32,
        )
        collater = Collater(pad_value=1)

        # 对音频数据进行预处理
        decoded_audio = {"waveform": waveform.T, "sample_rate": fe.sampling_rate, "format": -1}
        src = collater(fbank_converter(decoded_audio))["fbank"]
        seqs, padding_mask = get_seqs_and_padding_mask(src)

        # 获取原始模型的输出
        with torch.inference_mode():
            seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
            original_output, padding_mask = model.encoder(seqs, padding_mask)

        # 评估转换后的模型
        hf_wav2vec.eval()

        # 将音频数据输入转换后的模型
        inputs = fe(waveform, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = hf_wav2vec(**inputs)

        # 比较原始模型和转换后模型的输出
        torch.testing.assert_close(original_output, outputs.last_hidden_state, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--checkpoint_path", default="conformer_shaw", type=str, help="Path to seamless communication checkpoint"
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--repo_id", default=None, type=str, help="Push to this repo id if precised.")
    parser.add_argument(
        "--audio_path",
        default=None,
        type=str,
        help="If specified, check that the original model and the converted model produce the same outputs.",
    )

    args = parser.parse_args()
    # 调用函数将 wav2vec2 模型的检查点转换为适用于 BERT 模型的检查点
    convert_wav2vec2_bert_checkpoint(
        # 指定原始 wav2vec2 模型的检查点路径
        args.checkpoint_path,
        # 指定转换后的检查点保存路径
        args.pytorch_dump_folder_path,
        # 指定转换所需的配置文件路径
        args.config_path,
        # 指定转换所需的资源库 ID
        args.repo_id
    )
```