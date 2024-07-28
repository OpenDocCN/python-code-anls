# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\model.py`

```py
"""
This code is adapted from
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/model.py
"""

# 导入PyTorch的神经网络模块和张量类型
import torch.nn as nn
from torch import Tensor

# 从本地模块中导入特征提取器：ResNetFeatureExtractor和VGGFeatureExtractor
from .modules.feature_extraction import (
    ResNetFeatureExtractor,
    VGGFeatureExtractor,
)

# 从本地模块中导入预测模块：Attention
from .modules.prediction import Attention

# 从本地模块中导入序列建模模块：BidirectionalLSTM
from .modules.sequence_modeling import BidirectionalLSTM

# 从本地模块中导入转换模块：TpsSpatialTransformerNetwork
from .modules.transformation import TpsSpatialTransformerNetwork


class Model(nn.Module):
    # 初始化函数，接受一个字典 opt2val 作为参数
    def __init__(self, opt2val: dict):
        # 调用父类的初始化方法
        super(Model, self).__init__()

        # 从参数字典中获取各个配置值
        input_channel = opt2val["input_channel"]
        output_channel = opt2val["output_channel"]
        hidden_size = opt2val["hidden_size"]
        vocab_size = opt2val["vocab_size"]
        num_fiducial = opt2val["num_fiducial"]
        imgH = opt2val["imgH"]
        imgW = opt2val["imgW"]
        FeatureExtraction = opt2val["FeatureExtraction"]
        Transformation = opt2val["Transformation"]
        SequenceModeling = opt2val["SequenceModeling"]
        Prediction = opt2val["Prediction"]

        # 根据不同的 Transformation 配置选择性初始化空间变换网络
        if Transformation == "TPS":
            self.Transformation = TpsSpatialTransformerNetwork(
                F=num_fiducial,
                I_size=(imgH, imgW),
                I_r_size=(imgH, imgW),
                I_channel_num=input_channel,
            )
        else:
            # 如果未指定 Transformation 模块，则打印警告信息
            print("No Transformation module specified")

        # 根据 FeatureExtraction 配置选择性初始化特征提取器
        if FeatureExtraction == "VGG":
            extractor = VGGFeatureExtractor
        else:  # 默认使用 ResNet 特征提取器
            extractor = ResNetFeatureExtractor
        self.FeatureExtraction = extractor(
            input_channel,
            output_channel,
            opt2val,
        )
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1))  # 将最后一维度 (imgH/16-1) 转换为 1

        # 根据 SequenceModeling 配置选择性初始化序列建模器
        if SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output,
                    hidden_size,
                    hidden_size,
                ),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
            )
            self.SequenceModeling_output = hidden_size
        else:
            # 如果未指定 SequenceModeling 模块，则打印警告信息，并将输出设置为特征提取器的输出
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        # 根据 Prediction 配置选择性初始化预测模块
        if Prediction == "CTC":
            self.Prediction = nn.Linear(
                self.SequenceModeling_output,
                vocab_size,
            )
        elif Prediction == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output,
                hidden_size,
                vocab_size,
            )
        elif Prediction == "Transformer":  # TODO: 尚未实现 Transformer 预测模块
            pass
        else:
            # 如果 Prediction 既不是 CTC 也不是 Attn，则抛出异常
            raise Exception("Prediction is neither CTC or Attn")
    def forward(self, x: Tensor):
        """
        :param x: (batch, input_channel, height, width)
        :return: prediction tensor of shape (batch, T, num_classes)
        """
        # Transformation stage
        x = self.Transformation(x)  # 对输入进行变换处理

        # Feature extraction stage
        visual_feature = self.FeatureExtraction(
            x)  # 提取视觉特征，输出形状为 (batch, output_channel=512, h=3, w)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(
            0, 3, 1, 2))  # 自适应平均池化，调整维度为 (batch, w, channel=512, h=1)
        visual_feature = visual_feature.squeeze(3)  # 压缩维度，得到 (batch, w, channel=512)

        # Sequence modeling stage
        self.SequenceModeling.eval()  # 将 SequenceModeling 设置为评估模式
        contextual_feature = self.SequenceModeling(visual_feature)  # 序列建模，得到上下文特征

        # Prediction stage
        prediction = self.Prediction(
            contextual_feature.contiguous())  # 进行预测，输出形状为 (batch, T, num_classes)

        return prediction  # 返回预测结果
```