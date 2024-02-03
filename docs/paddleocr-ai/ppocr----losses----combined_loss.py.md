# `.\PaddleOCR\ppocr\losses\combined_loss.py`

```
# 导入 paddle 库
import paddle
# 导入 paddle 中的神经网络模块
import paddle.nn as nn

# 从当前目录下的 rec_ctc_loss 文件中导入 CTCLoss 类
from .rec_ctc_loss import CTCLoss
# 从当前目录下的 center_loss 文件中导入 CenterLoss 类
from .center_loss import CenterLoss
# 从当前目录下的 ace_loss 文件中导入 ACELoss 类
from .ace_loss import ACELoss
# 从当前目录下的 rec_sar_loss 文件中导入 SARLoss 类
from .rec_sar_loss import SARLoss

# 从当前目录下的 distillation_loss 文件中导入不同的 distillation loss 类
from .distillation_loss import DistillationCTCLoss, DistillCTCLogits
from .distillation_loss import DistillationSARLoss, DistillationNRTRLoss
from .distillation_loss import DistillationDMLLoss, DistillationKLDivLoss, DistillationDKDLoss
from .distillation_loss import DistillationDistanceLoss, DistillationDBLoss, DistillationDilaDBLoss
from .distillation_loss import DistillationVQASerTokenLayoutLMLoss, DistillationSERDMLLoss
from .distillation_loss import DistillationLossFromOutput
from .distillation_loss import DistillationVQADistanceLoss

# 定义 CombinedLoss 类，继承自 nn.Layer 类
class CombinedLoss(nn.Layer):
    """
    CombinedLoss:
        a combionation of loss function
    """
    # 初始化函数，接受一个损失配置列表作为参数
    def __init__(self, loss_config_list=None):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化损失函数列表和权重列表
        self.loss_func = []
        self.loss_weight = []
        # 检查损失配置列表是否为列表类型
        assert isinstance(loss_config_list, list), (
            'operator config should be a list')
        # 遍历损失配置列表
        for config in loss_config_list:
            # 检查配置是否为字典且只有一个键值对
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            # 获取配置中的名称和参数
            name = list(config)[0]
            param = config[name]
            # 检查参数中是否包含权重
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            # 将权重添加到权重列表中，并从参数中移除
            self.loss_weight.append(param.pop("weight"))
            # 根据名称和参数创建损失函数对象，并添加到损失函数列表中
            self.loss_func.append(eval(name)(**param))

    # 前向传播函数，接受输入、批次和其他关键字参数
    def forward(self, input, batch, **kargs):
        # 初始化损失字典和总损失值
        loss_dict = {}
        loss_all = 0.
        # 遍历损失函数列表
        for idx, loss_func in enumerate(self.loss_func):
            # 计算损失值
            loss = loss_func(input, batch, **kargs)
            # 如果损失值为张量，则转换为字典形式
            if isinstance(loss, paddle.Tensor):
                loss = {"loss_{}_{}".format(str(loss), idx): loss}

            # 获取当前损失函数对应的权重
            weight = self.loss_weight[idx]

            # 对损失值乘以权重
            loss = {key: loss[key] * weight for key in loss}

            # 如果损失字典中包含"loss"键，则将其加到总损失值中
            if "loss" in loss:
                loss_all += loss["loss"]
            else:
                # 否则将所有损失值相加
                loss_all += paddle.add_n(list(loss.values()))
            # 更新损失字典
            loss_dict.update(loss)
        # 将总损失值添加到损失字典中
        loss_dict["loss"] = loss_all
        # 返回损失字典
        return loss_dict
```