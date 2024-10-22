# `.\diffusers\pipelines\kandinsky\text_encoder.py`

```py
# 导入 PyTorch 库
import torch
# 从 transformers 库导入预训练模型类和配置类
from transformers import PreTrainedModel, XLMRobertaConfig, XLMRobertaModel


# 定义 MCLIPConfig 类，继承自 XLMRobertaConfig
class MCLIPConfig(XLMRobertaConfig):
    # 设置模型类型为 "M-CLIP"
    model_type = "M-CLIP"

    # 初始化方法，接受变换器和图像维度大小的参数
    def __init__(self, transformerDimSize=1024, imageDimSize=768, **kwargs):
        # 将变换器维度大小赋值给实例变量
        self.transformerDimensions = transformerDimSize
        # 将图像维度大小赋值给实例变量
        self.numDims = imageDimSize
        # 调用父类的初始化方法
        super().__init__(**kwargs)


# 定义 MultilingualCLIP 类，继承自 PreTrainedModel
class MultilingualCLIP(PreTrainedModel):
    # 指定配置类为 MCLIPConfig
    config_class = MCLIPConfig

    # 初始化方法，接受配置和其他参数
    def __init__(self, config, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)
        # 创建 XLMRobertaModel 实例，使用传入的配置
        self.transformer = XLMRobertaModel(config)
        # 创建线性变换层，输入特征为变换器维度，输出特征为图像维度
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    # 定义前向传播方法，接受输入 ID 和注意力掩码
    def forward(self, input_ids, attention_mask):
        # 获取变换器的嵌入表示，使用输入 ID 和注意力掩码
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        # 根据注意力掩码计算加权嵌入表示
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        # 返回线性变换后的嵌入表示和原始嵌入表示
        return self.LinearTransformation(embs2), embs
```