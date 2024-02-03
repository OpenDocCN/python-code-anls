# `.\PaddleOCR\ppocr\losses\text_focus_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/FudanVI/FudanOCR/blob/main/scene-text-telescope/loss/text_focus_loss.py

import paddle.nn as nn
import paddle
import numpy as np
import pickle as pkl

# 标准字母表
standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
# 标准字母表对应的索引字典
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index

# 加载混淆矩阵
def load_confuse_matrix(confuse_dict_path):
    # 打开混淆字典文件
    f = open(confuse_dict_path, 'rb')
    # 从文件中加载数据
    data = pkl.load(f)
    f.close()
    # 按照数字、小写字母、大写字母的顺序切分数据
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1, 62))
    pad = np.ones((63, 1))
    # 重新排列数据
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data == np.inf] = 1
    # 转换为 Paddle Tensor
    rearrange_data = paddle.to_tensor(rearrange_data)

    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    # upper_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # 调整混淆矩阵，确保小写字母的混淆概率不小于对应大写字母的混淆概率
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j + 26])
    rearrange_data = rearrange_data[:37, :37]

    return rearrange_data

# 权重交叉熵损失函数
def weight_cross_entropy(pred, gt, weight_table):
    batch = gt.shape[0]
    # 从权重表中获取对应标签的权重
    weight = weight_table[gt]
    # 对预测结果进行指数运算
    pred_exp = paddle.exp(pred)
    # 计算加权后的预测结果
    pred_exp_weight = weight * pred_exp
    # 初始化损失值
    loss = 0
    # 遍历每个样本的真实标签
    for i in range(len(gt)):
        # 计算交叉熵损失
        loss -= paddle.log(pred_exp_weight[i][gt[i]] / paddle.sum(pred_exp_weight, 1)[i])
    # 返回平均损失
    return loss / batch
class TelescopeLoss(nn.Layer):
    # 定义TelescopeLoss类，继承自nn.Layer
    def __init__(self, confuse_dict_path):
        # 初始化函数，接受混淆矩阵路径作为参数
        super(TelescopeLoss, self).__init__()
        # 调用父类的初始化函数
        self.weight_table = load_confuse_matrix(confuse_dict_path)
        # 加载混淆矩阵文件，存储在weight_table中
        self.mse_loss = nn.MSELoss()
        # 创建均方误差损失函数对象
        self.ce_loss = nn.CrossEntropyLoss()
        # 创建交叉熵损失函数对象
        self.l1_loss = nn.L1Loss()
        # 创建L1损失函数对象

    def forward(self, pred, data):
        # 前向传播函数，接受预测值和数据作为参数
        sr_img = pred["sr_img"]
        # 获取预测中的sr_img
        hr_img = pred["hr_img"]
        # 获取预测中的hr_img
        sr_pred = pred["sr_pred"]
        # 获取预测中的sr_pred
        text_gt = pred["text_gt"]
        # 获取预测中的text_gt

        word_attention_map_gt = pred["word_attention_map_gt"]
        # 获取预测中的word_attention_map_gt
        word_attention_map_pred = pred["word_attention_map_pred"]
        # 获取预测中的word_attention_map_pred
        mse_loss = self.mse_loss(sr_img, hr_img)
        # 计算sr_img和hr_img之间的均方误差损失
        attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
        # 计算word_attention_map_gt和word_attention_map_pred之间的L1损失
        recognition_loss = weight_cross_entropy(sr_pred, text_gt, self.weight_table)
        # 计算sr_pred和text_gt之间的加权交叉熵损失
        loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
        # 计算总损失，包括均方误差损失、注意力损失和识别损失
        return {
            "mse_loss": mse_loss,
            "attention_loss": attention_loss,
            "loss": loss
        }
        # 返回损失字典，包括均方误差损失、注意力损失和总损失
```