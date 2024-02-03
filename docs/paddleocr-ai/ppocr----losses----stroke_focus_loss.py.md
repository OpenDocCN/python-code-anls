# `.\PaddleOCR\ppocr\losses\stroke_focus_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于"原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
# 引用代码来源
# https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/loss/stroke_focus_loss.py
import cv2
import sys
import time
import string
import random
import numpy as np
import paddle.nn as nn
import paddle

# 定义 StrokeFocusLoss 类，继承自 nn.Layer
class StrokeFocusLoss(nn.Layer):
    def __init__(self, character_dict_path=None, **kwargs):
        # 调用父类的构造函数
        super(StrokeFocusLoss, self).__init__(character_dict_path)
        # 初始化均方误差损失
        self.mse_loss = nn.MSELoss()
        # 初始化交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()
        # 初始化 L1 损失
        self.l1_loss = nn.L1Loss()
        # 英文数字字符集
        self.english_stroke_alphabet = '0123456789'
        # 英文数字字符字典
        self.english_stroke_dict = {}
        # 构建英文数字字符字典
        for index in range(len(self.english_stroke_alphabet)):
            self.english_stroke_dict[self.english_stroke_alphabet[
                index]] = index

        # 读取字符字典文件中的内容
        stroke_decompose_lines = open(character_dict_path, 'r').readlines()
        # 初始化字符字典
        self.dic = {}
        # 遍历每一行内容
        for line in stroke_decompose_lines:
            line = line.strip()
            # 拆分字符和序列
            character, sequence = line.split()
            # 将字符和序列存入字典
            self.dic[character] = sequence
    # 前向传播函数，接收预测结果和数据作为输入
    def forward(self, pred, data):

        # 从预测结果中获取超分辨率图像和高分辨率图像
        sr_img = pred["sr_img"]
        hr_img = pred["hr_img"]

        # 计算均方误差损失
        mse_loss = self.mse_loss(sr_img, hr_img)
        
        # 从预测结果中获取真实词注意力图和预测词注意力图
        word_attention_map_gt = pred["word_attention_map_gt"]
        word_attention_map_pred = pred["word_attention_map_pred"]

        # 计算注意力损失，使用 L1 损失函数
        attention_loss = paddle.nn.functional.l1_loss(word_attention_map_gt,
                                                      word_attention_map_pred)

        # 计算总损失，包括均方误差损失和注意力损失，注意力损失权重为50，总损失乘以100
        loss = (mse_loss + attention_loss * 50) * 100

        # 返回损失字典，包括均方误差损失、注意力损失和总损失
        return {
            "mse_loss": mse_loss,
            "attention_loss": attention_loss,
            "loss": loss
        }
```