# `.\PaddleOCR\ppocr\modeling\architectures\distillation_model.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”基础分发，没有任何形式的保证
# 无论是明示的还是暗示的。有关特定语言的详细信息，请参见许可证
# 管理权限和限制
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle.nn 模块
from paddle import nn
# 从 ppocr.modeling.transforms 模块中导入 build_transform 函数
from ppocr.modeling.transforms import build_transform
# 从 ppocr.modeling.backbones 模块中导入 build_backbone 函数
from ppocr.modeling.backbones import build_backbone
# 从 ppocr.modeling.necks 模块中导入 build_neck 函数
from ppocr.modeling.necks import build_neck
# 从 ppocr.modeling.heads 模块中导入 build_head 函数
from ppocr.modeling.heads import build_head
# 从 .base_model 模块中导入 BaseModel 类
from .base_model import BaseModel
# 从 ppocr.utils.save_load 模块中导入 load_pretrained_params 函数
from ppocr.utils.save_load import load_pretrained_params

# 定义 DistillationModel 类，继承自 nn.Layer
__all__ = ['DistillationModel']


class DistillationModel(nn.Layer):
    def __init__(self, config):
        """
        初始化函数，用于OCR蒸馏模块。
        参数:
            config (dict): 模块的超参数。
        """
        # 调用父类的初始化函数
        super().__init__()
        # 初始化模型列表和模型名称列表
        self.model_list = []
        self.model_name_list = []
        # 遍历配置中的模型
        for key in config["Models"]:
            # 获取当前模型的配置
            model_config = config["Models"][key]
            freeze_params = False
            pretrained = None
            # 检查是否需要冻结参数
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            # 检查是否有预训练模型
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            # 创建基础模型
            model = BaseModel(model_config)
            # 如果有预训练模型，则加载参数
            if pretrained is not None:
                load_pretrained_params(model, pretrained)
            # 如果需要冻结参数，则设置参数不可训练
            if freeze_params:
                for param in model.parameters():
                    param.trainable = False
            # 将模型添加到模型列表中
            self.model_list.append(self.add_sublayer(key, model))
            self.model_name_list.append(key)

    def forward(self, x, data=None):
        # 初始化结果字典
        result_dict = dict()
        # 遍历模型名称列表
        for idx, model_name in enumerate(self.model_name_list):
            # 将模型的输出结果添加到结果字典中
            result_dict[model_name] = self.model_list[idx](x, data)
        # 返回结果字典
        return result_dict
```