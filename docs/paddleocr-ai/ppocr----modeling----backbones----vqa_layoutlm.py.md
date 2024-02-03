# `.\PaddleOCR\ppocr\modeling\backbones\vqa_layoutlm.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from paddle import nn

# 导入不同的 LayoutXLM 模型和任务类
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMForTokenClassification, LayoutXLMForRelationExtraction
# 导入 LayoutLM 模型和任务类
from paddlenlp.transformers import LayoutLMModel, LayoutLMForTokenClassification
# 导入 LayoutLMv2 模型和任务类
from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2ForTokenClassification, LayoutLMv2ForRelationExtraction
# 导入自动模型
from paddlenlp.transformers import AutoModel

# 定义可以导出的类
__all__ = ["LayoutXLMForSer", "LayoutLMForSer"]

# 预训练模型字典，包含不同模型和对应的预训练模型名称
pretrained_model_dict = {
    LayoutXLMModel: {
        "base": "layoutxlm-base-uncased",
        "vi": "vi-layoutxlm-base-uncased",
    },
    LayoutLMModel: {
        "base": "layoutlm-base-uncased",
    },
    LayoutLMv2Model: {
        "base": "layoutlmv2-base-uncased",
        "vi": "vi-layoutlmv2-base-uncased",
    },
}

# 定义 NLPBaseModel 类，继承自 nn.Layer
class NLPBaseModel(nn.Layer):
    # 初始化 NLPBaseModel 类，接受基础模型类、模型类、模式、类型、是否预训练、检查点等参数
    def __init__(self,
                 base_model_class,
                 model_class,
                 mode="base",
                 type="ser",
                 pretrained=True,
                 checkpoints=None,
                 **kwargs):
        # 调用父类的初始化方法
        super(NLPBaseModel, self).__init__()
        # 如果提供了检查点，则加载已训练的模型
        if checkpoints is not None:
            self.model = model_class.from_pretrained(checkpoints)
        else:
            # 否则加载预训练模型
            pretrained_model_name = pretrained_model_dict[base_model_class][mode]
            if pretrained is True:
                base_model = base_model_class.from_pretrained(pretrained_model_name)
            else:
                base_model = base_model_class.from_pretrained(pretrained)
            # 根据类型创建模型，设置类别数和丢弃率
            if type == "ser":
                self.model = model_class(base_model, num_classes=kwargs["num_classes"], dropout=None)
            else:
                self.model = model_class(base_model, dropout=None)
        # 设置输出通道数为1
        self.out_channels = 1
        # 使用视觉骨干网络
        self.use_visual_backbone = True
# 定义一个 LayoutLMForSer 类，继承自 NLPBaseModel 类
class LayoutLMForSer(NLPBaseModel):
    # 初始化方法，接受参数 num_classes, pretrained, checkpoints, mode 和 kwargs
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        # 调用父类的初始化方法，传入 LayoutLMModel, LayoutLMForTokenClassification, mode, "ser", pretrained, checkpoints 和 num_classes 参数
        super(LayoutLMForSer, self).__init__(
            LayoutLMModel,
            LayoutLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes, )
        # 初始化 use_visual_backbone 属性为 False
        self.use_visual_backbone = False

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 调用模型的 forward 方法，传入 input_ids, bbox, attention_mask, token_type_ids 和 position_ids 参数
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            position_ids=None,
            output_hidden_states=False)
        # 返回模型输出
        return x


# 定义一个 LayoutLMv2ForSer 类，继承自 NLPBaseModel 类
class LayoutLMv2ForSer(NLPBaseModel):
    # 初始化方法，接受参数 num_classes, pretrained, checkpoints, mode 和 kwargs
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        # 调用父类的初始化方法，传入 LayoutLMv2Model, LayoutLMv2ForTokenClassification, mode, "ser", pretrained, checkpoints 和 num_classes 参数
        super(LayoutLMv2ForSer, self).__init__(
            LayoutLMv2Model,
            LayoutLMv2ForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes)
        # 如果模型的 layoutlmv2 属性存在，并且 use_visual_backbone 属性为 False，则将 use_visual_backbone 属性设置为 False
        if hasattr(self.model.layoutlmv2, "use_visual_backbone"
                   ) and self.model.layoutlmv2.use_visual_backbone is False:
            self.use_visual_backbone = False

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 如果 use_visual_backbone 属性为 True，则将 image 设置为 x 的第五个元素，否则设置为 None
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None
        # 调用模型的 forward 方法，传入 input_ids, bbox, attention_mask, token_type_ids, image, position_ids, head_mask 和 labels 参数
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        # 如果处于训练状态，则返回包含 backbone_out 和其他输出的字典，否则直接返回模型输出
        if self.training:
            res = {"backbone_out": x[0]}
            res.update(x[1])
            return res
        else:
            return x


# 定义一个 LayoutXLMForSer 类，继承自 NLPBaseModel 类
    # 初始化 LayoutXLMForSer 类
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        # 调用 LayoutXLMForSer 类的父类的初始化方法
        super(LayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes)
        # 检查模型是否有可视化的背景，并设置相应的属性
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    # 前向传播函数
    def forward(self, x):
        # 如果模型使用可视化背景，则获取输入中的图像数据
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None
        # 调用模型进行前向传播
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        # 如果处于训练状态，则返回包含背景输出的字典
        if self.training:
            res = {"backbone_out": x[0]}
            res.update(x[1])
            return res
        # 否则直接返回模型输出
        else:
            return x
# 定义 LayoutLMv2ForRe 类，继承自 NLPBaseModel 类
class LayoutLMv2ForRe(NLPBaseModel):
    # 初始化方法，接受预训练参数、检查点、模式等参数
    def __init__(self, pretrained=True, checkpoints=None, mode="base",
                 **kwargs):
        # 调用父类的初始化方法，传入 LayoutLMv2Model、LayoutLMv2ForRelationExtraction、mode、"re"、pretrained、checkpoints 参数
        super(LayoutLMv2ForRe, self).__init__(
            LayoutLMv2Model, LayoutLMv2ForRelationExtraction, mode, "re",
            pretrained, checkpoints)
        # 如果模型中有 layoutlmv2 属性，并且其 use_visual_backbone 属性为 False，则设置 use_visual_backbone 为 False
        if hasattr(self.model.layoutlmv2, "use_visual_backbone"
                   ) and self.model.layoutlmv2.use_visual_backbone is False:
            self.use_visual_backbone = False

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 调用模型的 forward 方法，传入 input_ids、bbox、attention_mask、token_type_ids、image、position_ids、head_mask、labels、entities、relations 参数
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=x[4],
            position_ids=None,
            head_mask=None,
            labels=None,
            entities=x[5],
            relations=x[6])
        # 返回前向传播结果
        return x


# 定义 LayoutXLMForRe 类，继承自 NLPBaseModel 类
class LayoutXLMForRe(NLPBaseModel):
    # 初始化方法，接受预训练参数、检查点、模式等参数
    def __init__(self, pretrained=True, checkpoints=None, mode="base",
                 **kwargs):
        # 调用父类的初始化方法，传入 LayoutXLMModel、LayoutXLMForRelationExtraction、mode、"re"、pretrained、checkpoints 参数
        super(LayoutXLMForRe, self).__init__(
            LayoutXLMModel, LayoutXLMForRelationExtraction, mode, "re",
            pretrained, checkpoints)
        # 如果模型中有 layoutxlm 属性，并且其 use_visual_backbone 属性为 False，则设置 use_visual_backbone 为 False
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 如果 use_visual_backbone 为 True，则设置 image、entities、relations 分别为 x[4]、x[5]、x[6]
        if self.use_visual_backbone is True:
            image = x[4]
            entities = x[5]
            relations = x[6]
        # 否则，设置 image 为 None，entities 为 x[4]，relations 为 x[5]
        else:
            image = None
            entities = x[4]
            relations = x[5]
        # 调用模型的 forward 方法，传入 input_ids、bbox、attention_mask、token_type_ids、image、position_ids、head_mask、labels、entities、relations 参数
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None,
            entities=entities,
            relations=relations)
        # 返回前向传播结果
        return x
```