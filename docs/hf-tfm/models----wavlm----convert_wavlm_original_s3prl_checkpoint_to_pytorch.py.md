# `.\models\wavlm\convert_wavlm_original_s3prl_checkpoint_to_pytorch.py`

```
# 导入需要的库和模块
import argparse  # 导入参数解析模块

import torch  # 导入 PyTorch 库

# 从 transformers 库中导入需要的类和函数
from transformers import (
    Wav2Vec2FeatureExtractor,  # 导入音频特征提取器类
    WavLMConfig,  # 导入 WavLM 模型的配置类
    WavLMForAudioFrameClassification,  # 导入用于音频帧分类的 WavLM 模型类
    WavLMForSequenceClassification,  # 导入用于序列分类的 WavLM 模型类
    WavLMForXVector,  # 导入用于 XVector 的 WavLM 模型类
    logging,  # 导入日志记录模块
)

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取当前文件的日志记录器对象
logger = logging.get_logger(__name__)


# 定义转换序列分类模型的函数
def convert_classification(base_model_name, hf_config, downstream_dict):
    # 从预训练模型名称和配置创建 WavLM 序列分类模型
    model = WavLMForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型投影器的权重为下游任务的投影器权重
    model.projector.weight.data = downstream_dict["projector.weight"]
    # 设置模型投影器的偏置为下游任务的投影器偏置
    model.projector.bias.data = downstream_dict["projector.bias"]
    # 设置模型分类器的权重为下游任务的线性层权重
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    # 设置模型分类器的偏置为下游任务的线性层偏置
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    return model  # 返回转换后的模型


# 定义转换音频帧分类模型的函数
def convert_diarization(base_model_name, hf_config, downstream_dict):
    # 从预训练模型名称和配置创建 WavLM 音频帧分类模型
    model = WavLMForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    # 设置模型分类器的权重为下游任务的线性层权重
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    # 设置模型分类器的偏置为下游任务的线性层偏置
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    return model  # 返回转换后的模型


# 定义转换 XVector 模型的函数
def convert_xvector(base_model_name, hf_config, downstream_dict):
    # 从预训练模型名称和配置创建 WavLM XVector 模型
    model = WavLMForXVector.from_pretrained(base_model_name, config=hf_config)
    # 设置模型投影器的权重为下游任务的投影器权重
    model.projector.weight.data = downstream_dict["connector.weight"]
    # 设置模型投影器的偏置为下游任务的投影器偏置
    model.projector.bias.data = downstream_dict["connector.bias"]
    # 遍历和设置每个 TDNN 层的卷积核权重和偏置
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[
            f"model.framelevel_feature_extractor.module.{i}.kernel.weight"
        ]
        model.tdnn[i].kernel.bias.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.bias"]

    # 设置模型语音层特征提取器的第一个线性层权重和偏置
    model.feature_extractor.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.weight"]
    model.feature_extractor.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.bias"]
    # 设置模型语音层特征提取器的第二个线性层权重和偏置
    model.classifier.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.weight"]
    model.classifier.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.bias"]
    # 设置模型目标函数的权重
    model.objective.weight.data = downstream_dict["objective.W"]
    return model  # 返回转换后的模型


# 定义用于转换 S3PRL 检查点的函数，这个函数没有实现，只有文档字符串
@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    此函数用于从 S3PRL 模型检查点转换模型到其他格式，但是这里没有具体的实现代码。
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 加载模型检查点，指定在CPU上进行加载
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 从检查点中提取下游任务相关的信息
    downstream_dict = checkpoint["Downstream"]

    # 从预训练配置文件中加载 Wav2Vec2 模型的配置
    hf_config = WavLMConfig.from_pretrained(config_path)

    # 从预训练模型中加载 Wav2Vec2 特征提取器
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 获取模型架构名称
    arch = hf_config.architectures[0]
    
    # 根据模型架构名称选择合适的转换函数转换模型
    if arch.endswith("ForSequenceClassification"):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        # 抛出异常，表示不支持当前模型架构的权重转换
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    # 如果配置要求使用加权层求和，加载模型的加权层参数
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 将特征提取器保存到指定路径
    hf_feature_extractor.save_pretrained(model_dump_path)
    
    # 将转换后的模型保存到指定路径
    hf_model.save_pretrained(model_dump_path)
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数，用于指定huggingface预训练基础模型的名称
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    
    # 添加命令行参数，用于指定huggingface分类器配置文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    
    # 添加命令行参数，用于指定s3prl检查点文件的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    
    # 添加命令行参数，用于指定最终转换模型的保存路径
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    
    # 解析命令行参数，并将它们保存在args对象中
    args = parser.parse_args()
    
    # 调用函数，将指定的参数传递给函数
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```