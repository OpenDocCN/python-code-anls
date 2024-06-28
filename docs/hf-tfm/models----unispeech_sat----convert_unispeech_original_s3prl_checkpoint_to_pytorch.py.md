# `.\models\unispeech_sat\convert_unispeech_original_s3prl_checkpoint_to_pytorch.py`

```py
# 设置脚本的编码格式为 UTF-8
# 版权声明和许可信息，此处使用的是 Apache License 2.0
# 只允许在符合许可证的情况下使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“现状”分发软件
# 没有明示或暗示的任何保证或条件。详见许可证条款。

"""Convert Hubert checkpoint."""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数

import torch  # PyTorch 库

from transformers import (  # 从 transformers 库中导入以下模块和类
    UniSpeechSatConfig,  # UniSpeechSatConfig 配置类
    UniSpeechSatForAudioFrameClassification,  # 用于音频帧分类的模型类
    UniSpeechSatForSequenceClassification,  # 用于序列分类的模型类
    UniSpeechSatForXVector,  # 用于生成 x-vector 的模型类
    Wav2Vec2FeatureExtractor,  # Wav2Vec2 的特征提取器类
    logging,  # 日志记录模块
)

logging.set_verbosity_info()  # 设置日志记录的详细程度为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def convert_classification(base_model_name, hf_config, downstream_dict):
    # 从预训练模型和配置创建 UniSpeechSatForSequenceClassification 模型
    model = UniSpeechSatForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的投影层权重和偏置
    model.projector.weight.data = downstream_dict["projector.weight"]
    model.projector.bias.data = downstream_dict["projector.bias"]
    # 设置模型的分类器权重和偏置
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    return model  # 返回转换后的模型


def convert_diarization(base_model_name, hf_config, downstream_dict):
    # 从预训练模型和配置创建 UniSpeechSatForAudioFrameClassification 模型
    model = UniSpeechSatForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的分类器权重和偏置
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    return model  # 返回转换后的模型


def convert_xvector(base_model_name, hf_config, downstream_dict):
    # 从预训练模型和配置创建 UniSpeechSatForXVector 模型
    model = UniSpeechSatForXVector.from_pretrained(base_model_name, config=hf_config)
    # 设置模型的投影层权重和偏置
    model.projector.weight.data = downstream_dict["connector.weight"]
    model.projector.bias.data = downstream_dict["connector.bias"]
    
    # 遍历模型中的每个 TDNN 层，设置其权重和偏置
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[
            f"model.framelevel_feature_extractor.module.{i}.kernel.weight"
        ]
        model.tdnn[i].kernel.bias.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.bias"]

    # 设置特征提取器的权重和偏置
    model.feature_extractor.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.weight"]
    model.feature_extractor.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.bias"]
    # 设置分类器的权重和偏置
    model.classifier.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.weight"]
    model.classifier.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.bias"]
    # 设置目标函数的权重
    model.objective.weight.data = downstream_dict["objective.W"]
    return model  # 返回转换后的模型


@torch.no_grad()
# 定义函数，用于将 S3PRL 模型的检查点转换为 transformers 设计的模型
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 加载检查点文件，将其映射到 CPU 上
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 从检查点中获取 Downstream 字典
    downstream_dict = checkpoint["Downstream"]

    # 从预训练配置文件加载 UniSpeechSatConfig
    hf_config = UniSpeechSatConfig.from_pretrained(config_path)
    
    # 从预训练模型加载 Wav2Vec2FeatureExtractor
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 获取模型的架构名称
    arch = hf_config.architectures[0]
    
    # 根据模型架构选择相应的转换函数，转换成 Hugging Face 的模型
    if arch.endswith("ForSequenceClassification"):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        # 如果架构不被支持，抛出未实现错误
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    # 如果配置要求使用加权层求和，则加载 Featurizer 中的权重数据
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 将特征提取器的配置保存到模型导出路径
    hf_feature_extractor.save_pretrained(model_dump_path)
    
    # 将转换后的 Hugging Face 模型保存到模型导出路径
    hf_model.save_pretrained(model_dump_path)


# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：预训练基础模型名称
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    
    # 添加命令行参数：分类器配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    
    # 添加命令行参数：S3PRL 检查点文件路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    
    # 添加命令行参数：转换后模型保存路径
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用转换函数，传入命令行参数指定的参数
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```