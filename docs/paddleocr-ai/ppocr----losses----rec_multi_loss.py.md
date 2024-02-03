# `.\PaddleOCR\ppocr\losses\rec_multi_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 Paddle 库
import paddle
from paddle import nn

# 导入自定义的损失函数
from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss
from .rec_nrtr_loss import NRTRLoss

# 定义一个多损失函数类
class MultiLoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化损失函数字典和损失函数列表
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        # 获取权重参数
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        # 遍历损失函数列表
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                # 如果参数不为空，则更新参数
                if param is not None:
                    kwargs.update(param)
                # 根据损失函数名称动态创建损失函数对象
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss
    # 定义一个方法用于计算前向传播的损失值
    def forward(self, predicts, batch):
        # 初始化总损失字典
        self.total_loss = {}
        # 初始化总损失值
        total_loss = 0.0
        # 遍历损失函数字典
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            # 如果损失函数为CTCLoss
            if name == 'CTCLoss':
                # 计算CTCLoss的损失值并乘以权重1
                loss = loss_func(predicts['ctc'],
                                 batch[:2] + batch[3:])['loss'] * self.weight_1
            # 如果损失函数为SARLoss
            elif name == 'SARLoss':
                # 计算SARLoss的损失值并乘以权重2
                loss = loss_func(predicts['sar'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            # 如果损失函数为NRTRLoss
            elif name == 'NRTRLoss':
                # 计算NRTRLoss的损失值并乘以权重2
                loss = loss_func(predicts['nrtr'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            # 如果损失函数不在支持的列表中，则抛出NotImplementedError
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            # 将每个损失值存入总损失字典中
            self.total_loss[name] = loss
            # 累加总损失值
            total_loss += loss
        # 将总损失值存入总损失字典中
        self.total_loss['loss'] = total_loss
        # 返回总损失字典
        return self.total_loss
```