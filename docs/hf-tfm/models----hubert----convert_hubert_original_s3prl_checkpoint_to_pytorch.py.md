# `.\models\hubert\convert_hubert_original_s3prl_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定目的的适用性和
# 适销性。请查看许可证以获取特定语言的权限和限制
"""将 Hubert 检查点转换为 transformers 模型。"""


# 导入必要的库
import argparse
import torch
from transformers import HubertConfig, HubertForSequenceClassification, Wav2Vec2FeatureExtractor, logging

# 设置日志级别为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 支持的模型列表
SUPPORTED_MODELS = ["UtteranceLevel"]


# 禁用梯度计算
@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    复制/粘贴/调整模型的权重以符合 transformers 设计。
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 检查是否支持当前模型
    if checkpoint["Config"]["downstream_expert"]["modelrc"]["select"] not in SUPPORTED_MODELS:
        raise NotImplementedError(f"The supported s3prl models are {SUPPORTED_MODELS}")

    # 获取下游模型的权重
    downstream_dict = checkpoint["Downstream"]

    # 从预训练配置文件创建 Hubert 配置
    hf_congfig = HubertConfig.from_pretrained(config_path)
    # 从预训练模型创建 Hubert 分类模型
    hf_model = HubertForSequenceClassification.from_pretrained(base_model_name, config=hf_congfig)
    # 从预训练模型创建 Wav2Vec2 特征提取器
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    # 如果配置中使用了加权层求和，则设置权重
    if hf_congfig.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    # 设置投影层的权重和偏置
    hf_model.projector.weight.data = downstream_dict["projector.weight"]
    hf_model.projector.bias.data = downstream_dict["projector.bias"]
    # 设置分类器的权重和偏置
    hf_model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    hf_model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]

    # 保存特征提取器到指定路径
    hf_feature_extractor.save_pretrained(model_dump_path)
    # 保存模型到指定路径
    hf_model.save_pretrained(model_dump_path)


# 主函数入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
```