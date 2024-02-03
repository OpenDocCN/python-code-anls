# `.\PaddleOCR\ppocr\losses\rec_vl_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/wangyuxin87/VisionLAN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 中导入 nn 模块
from paddle import nn

# 定义 VLLoss 类，继承自 nn.Layer
class VLLoss(nn.Layer):
    # 初始化函数，接受 mode、weight_res、weight_mas 等参数
    def __init__(self, mode='LF_1', weight_res=0.5, weight_mas=0.5, **kwargs):
        super(VLLoss, self).__init__()
        # 定义损失函数为交叉熵损失，reduction 参数为 "mean"
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(reduction="mean")
        # 断言 mode 参数在指定范围内
        assert mode in ['LF_1', 'LF_2', 'LA']
        # 初始化 mode、weight_res、weight_mas 属性
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    # 将标签展平的函数
    def flatten_label(self, target):
        label_flatten = []
        label_length = []
        # 遍历目标张量的每一行
        for i in range(0, target.shape[0]):
            # 将当前行的标签转换为列表
            cur_label = target[i].tolist()
            # 将当前行的标签添加到 label_flatten 中，直到遇到 0
            label_flatten += cur_label[:cur_label.index(0) + 1]
            # 记录当前行标签的长度
            label_length.append(cur_label.index(0) + 1)
        # 将 label_flatten 转换为张量，数据类型为 int64
        label_flatten = paddle.to_tensor(label_flatten, dtype='int64')
        # 将 label_length 转换为张量，数据类型为 int32
        label_length = paddle.to_tensor(label_length, dtype='int32')
        return (label_flatten, label_length)

    # 将输入源展平的函数
    def _flatten(self, sources, lengths):
        # 拼接所有输入源，长度由 lengths 决定
        return paddle.concat([t[:l] for t, l in zip(sources, lengths)]
    # 定义一个前向传播函数，接受预测结果和批处理数据作为参数
    def forward(self, predicts, batch):
        # 获取第一个预测结果
        text_pre = predicts[0]
        # 将目标值转换为int64类型
        target = batch[1].astype('int64')
        # 将目标值展平并获取长度信息
        label_flatten, length = self.flatten_label(target)
        # 将第一个预测结果展平
        text_pre = self._flatten(text_pre, length)
        
        # 根据模式选择不同的处理方式
        if self.mode == 'LF_1':
            # 计算损失函数
            loss = self.loss_func(text_pre, label_flatten)
        else:
            # 获取额外的预测结果和目标值
            text_rem = predicts[1]
            text_mas = predicts[2]
            target_res = batch[2].astype('int64')
            target_sub = batch[3].astype('int64')
            # 将额外的目标值展平并获取长度信息
            label_flatten_res, length_res = self.flatten_label(target_res)
            label_flatten_sub, length_sub = self.flatten_label(target_sub)
            # 将额外的预测结果展平
            text_rem = self._flatten(text_rem, length_res)
            text_mas = self._flatten(text_mas, length_sub)
            # 计算不同部分的损失函数并加权求和
            loss_ori = self.loss_func(text_pre, label_flatten)
            loss_res = self.loss_func(text_rem, label_flatten_res)
            loss_mas = self.loss_func(text_mas, label_flatten_sub)
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        
        # 返回损失值作为字典
        return {'loss': loss}
```