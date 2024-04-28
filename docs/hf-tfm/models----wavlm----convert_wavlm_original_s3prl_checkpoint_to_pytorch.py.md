# `.\transformers\models\wavlm\convert_wavlm_original_s3prl_checkpoint_to_pytorch.py`

```
# 导入需要的库
import argparse
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMConfig,
    WavLMForAudioFrameClassification,
    WavLMForSequenceClassification,
    WavLMForXVector,
    logging,
)

# 设置日志输出级别为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义函数将预训练的分类模型转换为 WavLMForSequenceClassification 模型
def convert_classification(base_model_name, hf_config, downstream_dict):
    model = WavLMForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict["projector.weight"]
    model.projector.bias.data = downstream_dict["projector.bias"]
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    return model

# 定义函数将预训练的音频帧分类模型转换为 WavLMForAudioFrameClassification 模型
def convert_diarization(base_model_name, hf_config, downstream_dict):
    model = WavLMForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    return model

# 定义函数将预训练的 XVector 模型转换为 WavLMForXVector 模型
def convert_xvector(base_model_name, hf_config, downstream_dict):
    model = WavLMForXVector.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict["connector.weight"]
    model.projector.bias.data = downstream_dict["connector.bias"]
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.weight"]
        model.tdnn[i].kernel.bias.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.bias"]
    model.feature_extractor.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.weight"]
    model.feature_extractor.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.bias"]
    model.classifier.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.weight"]
    model.classifier.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.bias"]
    model.objective.weight.data = downstream_dict["objective.W"]
    return model

# 定义函数将 S3PRL 模型转换为 Hubert 模型
@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    # 函数定义需要转换的过程
    pass
    # 加载模型的权重并将其拷贝/粘贴/微调到 transformers 的设计中
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 从检查点中获取下游模型的权重
    downstream_dict = checkpoint["Downstream"]

    # 从预训练配置中创建 WavLMConfig 对象
    hf_config = WavLMConfig.from_pretrained(config_path)

    # 从预训练模型中创建 Wav2Vec2FeatureExtractor 对象
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 获取模型的架构
    arch = hf_config.architectures[0]
    
    # 根据架构类型选择不同的转换方法
    if arch.endswith("ForSequenceClassification"):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    # 如果配置中使用了加权层求和，则将模型的层权重更新为检查点中的权重
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 将特征提取器保存到指定路径
    hf_feature_extractor.save_pretrained(model_dump_path)
    # 将模型保存到指定路径
    hf_model.save_pretrained(model_dump_path)
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：预训练基础模型的名称
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    # 添加参数：分类器配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    # 添加参数：s3prl检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
  # 添加参数：转换后模型保存路径
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    # 解析参数
    args = parser.parse_args()
    # 调用函数以转换s3prl检查点为Hugging Face模型
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```