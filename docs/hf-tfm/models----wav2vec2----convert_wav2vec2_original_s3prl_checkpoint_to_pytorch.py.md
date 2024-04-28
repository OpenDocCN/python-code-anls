# `.\transformers\models\wav2vec2\convert_wav2vec2_original_s3prl_checkpoint_to_pytorch.py`

```
# 设置 UTF-8 编码
# 版权声明
# 根据 Apache 许可证 2.0 版许可使用
# 您可以在符合许可证的情况下使用本文件
# 您可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意
# 分发的软件都是基于“如其所是”的基础分发的
# 没有任何明示或暗示的保证或条件
# 有关权限详细信息，请查看许可证
# 限制许可证下的特定语言控制权限和获取权限
"""转换 Hubert 检查点。"""


# 导入所需模块
import argparse
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    logging,
)


# 设置日志输出级别为 info
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)


# 转换分类模型
def convert_classification(base_model_name, hf_config, downstream_dict):
    # 从预训练模型中加载 Wav2Vec2ForSequenceClassification 模型
    model = Wav2Vec2ForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置projector的权重
    model.projector.weight.data = downstream_dict["projector.weight"]
    # 设置projector的偏置
    model.projector.bias.data = downstream_dict["projector.bias"]
    # 设置classifier的权重
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    # 设置classifier的偏置
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    return model


# 转换音频帧分类模型
def convert_diarization(base_model_name, hf_config, downstream_dict):
    # 从预训练模型中加载 Wav2Vec2ForAudioFrameClassification 模型
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置classifier的权重
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    # 设置classifier的偏置
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    return model


# 转换 X-Vector 模型
def convert_xvector(base_model_name, hf_config, downstream_dict):
    # 从预训练模型中加载 Wav2Vec2ForXVector 模型
    model = Wav2Vec2ForXVector.from_pretrained(base_model_name, config=hf_config)
    # 设置projector的权重
    model.projector.weight.data = downstream_dict["connector.weight"]
    # 设置projector的偏置
    model.projector.bias.data = downstream_dict["connector.bias"]
    
    # 设置各个 TDNN 模块的权重和偏置
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
    return model


# 转换 S3PRL 检查点的函数，无需梯度计算
@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    将模型的权重复制/粘贴/调整到 transformers 设计中。
    """
    # 加载检查点文件中的权重，并放置在 CPU 上
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 获取下游任务的字典
    downstream_dict = checkpoint["Downstream"]

    # 从预训练配置文件中加载 HF 配置
    hf_config = Wav2Vec2Config.from_pretrained(config_path)
    # 从预训练特征提取器加载 HF 特征提取器
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 获取 HF 模型的架构
    arch = hf_config.architectures[0]
    # 根据架构类型进行不同的转换
    if arch.endswith("ForSequenceClassification"):
        # 将模型转换为分类模型
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        # 将模型转换为语音分框模型
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        # 将模型转换为 XVector 模型
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        # 抛出未实现的错误，提示不支持当前架构类型的转换
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    # 如果配置要求使用加权层求和
    if hf_config.use_weighted_layer_sum:
        # 设置 HF 模型的层权重为检查点中的权重
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 将 HF 特征提取器保存到模型转储路径
    hf_feature_extractor.save_pretrained(model_dump_path)
    # 将 HF 模型保存到模型转储路径
    hf_model.save_pretrained(model_dump_path)
```  
# 如果当前脚本作为独立运行的程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定预训练基模型的名称
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    # 添加命令行参数，用于指定分类器配置文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    # 添加命令行参数，用于指定S3PRL检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    # 添加命令行参数，用于指定最终转换后模型的保存路径
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将S3PRL检查点转换为Hugging Face模型
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```