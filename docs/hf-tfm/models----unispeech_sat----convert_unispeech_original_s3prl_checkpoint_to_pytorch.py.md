# `.\transformers\models\unispeech_sat\convert_unispeech_original_s3prl_checkpoint_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明
# 根据Apache许可2.0版，除非符合该许可，否则不得使用此文件
# 可以在以下网址取得许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得根据许可分发的软件
# 根据许可的"按原样"基础分发，不附带任何担保或条件，无论是明示的或暗示的
# 请参阅许可中关于特定语言的限制和权利的规定
"""Convert Hubert checkpoint."""


# 导入相关包
import argparse
import torch
# 从transformers包中导入以下内容
from transformers import (
    UniSpeechSatConfig,
    UniSpeechSatForAudioFrameClassification,
    UniSpeechSatForSequenceClassification,
    UniSpeechSatForXVector,
    Wav2Vec2FeatureExtractor,
    logging,
)


#设置日志级别
logging.set_verbosity_info()
# 获取默认日志器
logger = logging.get_logger(__name__)


# 转换用于序列分类的模型
def convert_classification(base_model_name, hf_config, downstream_dict):
    # 从给定的模型名称和配置中加载预训练的UniSpeechSatForSequenceClassification模型
    model = UniSpeechSatForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的投影层权重
    model.projector.weight.data = downstream_dict["projector.weight"]
    # 设置模型的投影层偏置
    model.projector.bias.data = downstream_dict["projector.bias"]
    # 设置模型的分类器权重
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    # 设置模型的分类器偏置
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    # 返回转换后的模型
    return model


# 转换用于音频帧分类的模型
def convert_diarization(base_model_name, hf_config, downstream_dict):
    # 从给定的模型名称和配置中加载预训练的UniSpeechSatForAudioFrameClassification模型
    model = UniSpeechSatForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的分类器权重
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    # 设置模型的分类器偏置
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    # 返回转换后的模型
    return model


# 转换用于X向量的模型
def convert_xvector(base_model_name, hf_config, downstream_dict):
    # 从给定的模型名称和配置中加载预训练的UniSpeechSatForXVector模型
    model = UniSpeechSatForXVector.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的投影层权重
    model.projector.weight.data = downstream_dict["connector.weight"]
    # 设置模型的投影层偏置
    model.projector.bias.data = downstream_dict["connector.bias"]
    # 对于每一个tdnn模块，设置对应的权重和偏置
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[
            f"model.framelevel_feature_extractor.module.{i}.kernel.weight"
        ]
        model.tdnn[i].kernel.bias.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.bias"]
    # 设置模型的特征提取器权重
    model.feature_extractor.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.weight"]
    # 设置模型的特征提取器偏置
    model.feature_extractor.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.bias"]
    # 设置模型的分类器权重
    model.classifier.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.weight"]
    # 设置模型的分类器偏置
    model.classifier.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.bias"]
    # 设置模型的目标权重
    model.objective.weight.data = downstream_dict["objective.W"]
    # 返回转换后的模型
    return model


# 带有torch.no_grad()装饰器的函数
@torch.no_grad()
# 将S3PRL模型的权重转换为transformers设计的模型，用于拷贝/粘贴/调整模型的权重。
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    # 使用torch.load加载checkpoint文件，map_location指定为CPU
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 从checkpoint中获取下游任务的字典
    downstream_dict = checkpoint["Downstream"]

    # 从预训练的config路径加载UniSpeechSatConfig
    hf_config = UniSpeechSatConfig.from_pretrained(config_path)
    # 从预训练的base_model_name加载Wav2Vec2FeatureExtractor，返回attention mask，并且不进行归一化
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 获取模型架构，通常为列表中的第一个
    arch = hf_config.architectures[0]
    # 根据模型架构的不同类型进行不同的转换
    if arch.endswith("ForSequenceClassification"):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        # 如果架构类型不支持，抛出NotImplementedError异常
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    # 如果配置中使用加权层和和，则加载Featurizer中的权重
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 将特征提取器保存到模型转储路径中
    hf_feature_extractor.save_pretrained(model_dump_path)
    # 将模型保存到模型转储路径中
    hf_model.save_pretrained(model_dump_path)


if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加四个参数，分别为base_model_name、config_path、checkpoint_path、model_dump_path
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    # 解析参数
    args = parser.parse_args()
    # 调用convert_s3prl_checkpoint函数，传入参数
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```  
```