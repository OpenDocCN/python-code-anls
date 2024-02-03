# `.\PaddleOCR\ppocr\losses\rec_rfl_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 代码来源于：
# https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_common/models/loss/cross_entropy_loss.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 中导入 nn 模块
from paddle import nn

# 从当前目录下的 basic_loss 文件中导入 CELoss 和 DistanceLoss 类
from .basic_loss import CELoss, DistanceLoss

# 定义 RFLLoss 类，继承自 nn.Layer 类
class RFLLoss(nn.Layer):
    # 初始化方法
    def __init__(self, ignore_index=-100, **kwargs):
        # 调用父类的初始化方法
        super().__init__()

        # 创建均方误差损失函数对象，传入参数 kwargs
        self.cnt_loss = nn.MSELoss(**kwargs)
        # 创建交叉熵损失函数对象，设置忽略索引为 ignore_index
        self.seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # 定义一个前向传播函数，接受预测结果和批次数据作为输入参数
    def forward(self, predicts, batch):

        # 初始化总损失字典和总损失值
        self.total_loss = {}
        total_loss = 0.0
        
        # 判断预测结果是否为元组或列表
        if isinstance(predicts, tuple) or isinstance(predicts, list):
            cnt_outputs, seq_outputs = predicts
        else:
            cnt_outputs, seq_outputs = predicts, None
        
        # 获取批次数据中的内容 [image, label, length, cnt_label]
        if cnt_outputs is not None:
            # 计算分类损失
            cnt_loss = self.cnt_loss(cnt_outputs, paddle.cast(batch[3], paddle.float32))
            self.total_loss['cnt_loss'] = cnt_loss
            total_loss += cnt_loss

        if seq_outputs is not None:
            # 获取目标值和标签长度
            targets = batch[1].astype("int64")
            label_lengths = batch[2].astype('int64')
            batch_size, num_steps, num_classes = seq_outputs.shape[0], seq_outputs.shape[1], seq_outputs.shape[2]
            
            # 断言目标值的形状和输入的形状匹配
            assert len(targets.shape) == len(list(seq_outputs.shape)) - 1, \
                "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

            # 处理输入和目标值的形状
            inputs = seq_outputs[:, :-1, :]
            targets = targets[:, 1:]

            inputs = paddle.reshape(inputs, [-1, inputs.shape[-1]])
            targets = paddle.reshape(targets, [-1])
            
            # 计算序列损失
            seq_loss = self.seq_loss(inputs, targets)
            self.total_loss['seq_loss'] = seq_loss
            total_loss += seq_loss

        # 计算总损失并添加到总损失字典中
        self.total_loss['loss'] = total_loss
        # 返回总损失字典
        return self.total_loss
```