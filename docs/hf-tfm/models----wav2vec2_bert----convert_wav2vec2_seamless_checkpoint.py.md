# `.\models\wav2vec2_bert\convert_wav2vec2_seamless_checkpoint.py`

```
# 定义函数用于计算模型参数总数，不包括特定键名的参数
def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


# 定义私有函数，用于转换模型参数
def _convert_model(
    original_model,
    hf_model,
    convert_list,
):
    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 遍历状态字典中的键值对列表
    for k, v in list(state_dict.items()):
        # 复制键，准备进行重命名
        new_key = k
        # 遍历转换列表，将符合条件的旧层名替换为新层名
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_key:
                new_key = new_key.replace(old_layer_name, new_layer_name)

        # 手动处理层归一化的情况
        if ".layer_norm" in new_key and new_key.split(".layer_norm")[0][-1].isnumeric():
            new_key = new_key.replace("layer_norm", "final_layer_norm")

        # 检查是否需要移除当前键
        add_key = True
        for key in keys_to_remove:
            if key in new_key:
                # 如果键中包含需要移除的关键词，则从状态字典中移除该键值对
                state_dict.pop(k)
                add_key = False
                break

        # 如果不需要移除，则将更新后的键值对添加回状态字典中
        if add_key:
            state_dict[new_key] = state_dict.pop(k)

    # 计算多余的键（存在于状态字典中但不在预期模型中的）
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    # 过滤掉不必要的参数（如包含"num_updates"的键）
    extra_keys = set({k for k in extra_keys if "num_updates" not in k})
    # 计算缺失的键（存在于预期模型中但不在状态字典中的）
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())

    # 如果存在多余的键，则抛出数值错误异常
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    # 如果存在缺失的键，则抛出数值错误异常
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")

    # 使用更新后的状态字典加载预训练模型的状态
    hf_model.load_state_dict(state_dict, strict=True)
    # 计算加载后模型的参数数量
    n_params = param_count(hf_model)

    # 记录模型加载完成并输出参数数量（以百万为单位）
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    # 将模型设置为评估模式
    hf_model.eval()
    # 删除状态字典，释放内存
    del state_dict

    # 返回加载并配置好的模型
    return hf_model
# 使用 @torch.no_grad() 装饰器，确保在模型推断过程中不进行梯度计算
@torch.no_grad()
# 定义函数 convert_wav2vec2_bert_checkpoint，用于将模型权重从 Wav2Vec2 转换到 Transformers 设计
def convert_wav2vec2_bert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了 config_path，则从预训练的配置文件加载 Wav2Vec2BertConfig，并设置隐藏层激活函数为 "swish"
    if config_path is not None:
        config = Wav2Vec2BertConfig.from_pretrained(config_path, hidden_act="swish")
    else:
        # 否则创建一个新的 Wav2Vec2BertConfig 对象，关闭 spec-augment
        config = Wav2Vec2BertConfig(apply_spec_augment=False)

    # 根据配置创建 Wav2Vec2BertModel 模型对象
    hf_wav2vec = Wav2Vec2BertModel(config)

    # 加载 Conformer 模型，将其类型转换为 torch.float32，并设为评估模式
    model = load_conformer_shaw_model(checkpoint_path, dtype=torch.float32)
    model.eval()

    # 将 Conformer 模型的权重转换到 hf_wav2vec 模型中，使用预定义的转换列表 wav2vec_convert_list
    hf_wav2vec = _convert_model(model, hf_wav2vec, wav2vec_convert_list)

    # 将转换后的 hf_wav2vec 模型保存到指定的 PyTorch 转储文件夹中
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了 repo_id，则将 hf_wav2vec 模型推送到指定的仓库，并创建 pull request
    if repo_id:
        hf_wav2vec.push_to_hub(repo_id, create_pr=True)

    # 创建 SeamlessM4TFeatureExtractor 特征提取器对象，设置填充值为 1
    fe = SeamlessM4TFeatureExtractor(padding_value=1)
    # 将特征提取器的处理器类设为 "Wav2Vec2BertProcessor"
    fe._set_processor_class("Wav2Vec2BertProcessor")
    # 将特征提取器保存到指定的 PyTorch 转储文件夹中
    fe.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了 repo_id，则将特征提取器推送到指定的仓库，并创建 pull request
    if repo_id:
        fe.push_to_hub(repo_id, create_pr=True)

    # 如果提供了 args.audio_path，则加载音频文件，并进行必要的预处理和特征提取
    if args.audio_path:
        # 加载音频文件，并获取波形和采样率
        waveform, sample_rate = torchaudio.load(args.audio_path)
        # 使用特征提取器的采样率对波形进行重新采样
        waveform = torchaudio.functional.resample(waveform, sample_rate, fe.sampling_rate)

        # 创建 WaveformToFbankConverter 对象，将波形转换为 FBANK 特征
        fbank_converter = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            dtype=torch.float32,
        )
        # 创建 Collater 对象，用于对 FBANK 特征进行填充
        collater = Collater(pad_value=1)

        # 构建解码后的音频字典 decoded_audio
        decoded_audio = {"waveform": waveform.T, "sample_rate": fe.sampling_rate, "format": -1}
        # 对解码后的音频数据应用特征提取器，并获取 FBANK 特征及其填充掩码
        src = collater(fbank_converter(decoded_audio))["fbank"]
        seqs, padding_mask = get_seqs_and_padding_mask(src)

        # 在推断模式下运行模型的前端编码器和编码器，获取原始输出和填充掩码
        with torch.inference_mode():
            seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
            original_output, padding_mask = model.encoder(seqs, padding_mask)

        # 将 hf_wav2vec 模型设为评估模式
        hf_wav2vec.eval()

        # 使用特征提取器对音频进行编码，并通过 hf_wav2vec 模型获取输出
        inputs = fe(waveform, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = hf_wav2vec(**inputs)

        # 使用 torch.testing.assert_close 检查原始模型输出和转换后模型输出的相似性
        torch.testing.assert_close(original_output, outputs.last_hidden_state, atol=5e-3, rtol=5e-3)


# 如果当前脚本作为主程序运行，则解析命令行参数
if __name__ == "__main__":
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

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数 convert_wav2vec2_bert_checkpoint，将指定的参数传递给它
    convert_wav2vec2_bert_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.repo_id
    )
```