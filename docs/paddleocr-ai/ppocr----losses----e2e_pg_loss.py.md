# `.\PaddleOCR\ppocr\losses\e2e_pg_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
from paddle import nn
import paddle

# 从 det_basic_loss 模块中导入 DiceLoss 类
from .det_basic_loss import DiceLoss
# 从 ppocr.utils.e2e_utils.extract_batchsize 模块中导入 pre_process 函数

# 定义 PGLoss 类，继承自 nn.Layer 类
class PGLoss(nn.Layer):
    # 初始化函数，接受 tcl_bs, max_text_length, max_text_nums, pad_num, eps 等参数
    def __init__(self,
                 tcl_bs,
                 max_text_length,
                 max_text_nums,
                 pad_num,
                 eps=1e-6,
                 **kwargs):
        # 调用父类的初始化函数
        super(PGLoss, self).__init__()
        # 初始化 PGLoss 类的属性
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        # 创建 DiceLoss 类的实例，传入 eps 参数
        self.dice_loss = DiceLoss(eps=eps)
    # 计算边界损失函数，输入参数为预测的边界框、真实的边界框、置信度得分和掩码
    def border_loss(self, f_border, l_border, l_score, l_mask):
        # 将真实边界框拆分为坐标信息和归一化信息
        l_border_split, l_border_norm = paddle.tensor.split(
            l_border, num_or_sections=[4, 1], axis=1)
        # 将预测边界框作为边界框的输出
        f_border_split = f_border
        # 获取真实边界框归一化信息的形状
        b, c, h, w = l_border_norm.shape
        # 将真实边界框归一化信息进行扩展
        l_border_norm_split = paddle.expand(
            x=l_border_norm, shape=[b, 4 * c, h, w])
        # 获取置信度得分的形状
        b, c, h, w = l_score.shape
        # 将置信度得分进行扩展
        l_border_score = paddle.expand(x=l_score, shape=[b, 4 * c, h, w])
        # 获取掩码的形状
        b, c, h, w = l_mask.shape
        # 将掩码进行扩展
        l_border_mask = paddle.expand(x=l_mask, shape=[b, 4 * c, h, w])
        # 计算边界框的差值
        border_diff = l_border_split - f_border_split
        # 计算绝对值的边界框差值
        abs_border_diff = paddle.abs(border_diff)
        # 判断边界框差值是否小于1.0
        border_sign = abs_border_diff < 1.0
        # 将判断结果转换为float32类型
        border_sign = paddle.cast(border_sign, dtype='float32')
        # 停止梯度的传播
        border_sign.stop_gradient = True
        # 计算边界框内部损失
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                         (abs_border_diff - 0.5) * (1.0 - border_sign)
        # 计算边界框外部损失
        border_out_loss = l_border_norm_split * border_in_loss
        # 计算边界损失
        border_loss = paddle.sum(border_out_loss * l_border_score * l_border_mask) / \
                      (paddle.sum(l_border_score * l_border_mask) + 1e-5)
        # 返回边界损失值
        return border_loss
    # 计算方向损失函数，输入参数包括前景方向、后景方向、得分、掩码
    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        # 将后景方向张量按照指定维度分割成两部分，分别为l_direction_split和l_direction_norm
        l_direction_split, l_direction_norm = paddle.tensor.split(
            l_direction, num_or_sections=[2, 1], axis=1)
        # 前景方向不需要分割，直接赋值给f_direction_split
        f_direction_split = f_direction
        # 获取l_direction_norm的形状信息
        b, c, h, w = l_direction_norm.shape
        # 将l_direction_norm进行扩展，使其形状与l_direction_split相同
        l_direction_norm_split = paddle.expand(
            x=l_direction_norm, shape=[b, 2 * c, h, w])
        # 获取l_score的形状信息
        b, c, h, w = l_score.shape
        # 将l_score进行扩展，使其形状与l_direction_split相同
        l_direction_score = paddle.expand(x=l_score, shape=[b, 2 * c, h, w])
        # 获取l_mask的形状信息
        b, c, h, w = l_mask.shape
        # 将l_mask进行扩展，使其形状与l_direction_split相同
        l_direction_mask = paddle.expand(x=l_mask, shape=[b, 2 * c, h, w])
        # 计算方向差异
        direction_diff = l_direction_split - f_direction_split
        # 计算方向差异的绝对值
        abs_direction_diff = paddle.abs(direction_diff)
        # 判断方向差异的绝对值是否小于1.0，得到布尔张量
        direction_sign = abs_direction_diff < 1.0
        # 将布尔张量转换为float32类型
        direction_sign = paddle.cast(direction_sign, dtype='float32')
        # 设置direction_sign不参与梯度计算
        direction_sign.stop_gradient = True
        # 计算方向内损失
        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + \
                            (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        # 计算方向外损失
        direction_out_loss = l_direction_norm_split * direction_in_loss
        # 计算方向损失
        direction_loss = paddle.sum(direction_out_loss * l_direction_score * l_direction_mask) / \
                         (paddle.sum(l_direction_score * l_direction_mask) + 1e-5)
        # 返回方向损失
        return direction_loss
    # 定义 CTC 损失函数，计算字符级别的 CTC 损失
    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        # 将输入特征张量的维度进行转置，将通道维度放到最后
        f_char = paddle.transpose(f_char, [0, 2, 3, 1])
        # 将目标位置张量进行形状重塑，变为二维张量
        tcl_pos = paddle.reshape(tcl_pos, [-1, 3])
        # 将目标位置张量的数据类型转换为整型
        tcl_pos = paddle.cast(tcl_pos, dtype=int)
        # 根据目标位置从输入特征张量中提取对应位置的特征
        f_tcl_char = paddle.gather_nd(f_char, tcl_pos)
        # 将提取的特征进行形状重塑，增加一个维度
        f_tcl_char = paddle.reshape(
            f_tcl_char, [-1, 64, self.pad_num + 1])  # len(Lexicon_Table)+1
        # 将特征张量按照最后一个维度进行分割，分为前景和背景特征
        f_tcl_char_fg, f_tcl_char_bg = paddle.split(
            f_tcl_char, [self.pad_num, 1], axis=2)
        # 根据掩码将背景特征进行处理，保留有效区域，无效区域填充为固定值
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        # 获取掩码张量的形状信息
        b, c, l = tcl_mask.shape
        # 根据掩码张量的形状信息扩展前景掩码，使其与特征张量形状相匹配
        tcl_mask_fg = paddle.expand(x=tcl_mask, shape=[b, c, self.pad_num * l])
        tcl_mask_fg.stop_gradient = True
        # 根据前景掩码将前景特征进行处理，保留有效区域，无效区域填充为固定值
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (
            -20.0)
        # 将前景和背景特征拼接在一起，形成最终的特征张量
        f_tcl_char_mask = paddle.concat([f_tcl_char_fg, f_tcl_char_bg], axis=2)
        # 将特征张量进行转置，调整维度顺序
        f_tcl_char_ld = paddle.transpose(f_tcl_char_mask, (1, 0, 2))
        # 获取特征张量的形状信息
        N, B, _ = f_tcl_char_ld.shape
        # 创建输入长度张量，表示每个样本的序列长度
        input_lengths = paddle.to_tensor([N] * B, dtype='int64')
        # 计算 CTC 损失，传入特征张量、标签、输入长度、标签长度等参数
        cost = paddle.nn.functional.ctc_loss(
            log_probs=f_tcl_char_ld,
            labels=tcl_label,
            input_lengths=input_lengths,
            label_lengths=label_t,
            blank=self.pad_num,
            reduction='none')
        # 对损失进行求均值操作
        cost = cost.mean()
        # 返回计算得到的损失值
        return cost
    # 定义一个前向传播函数，接受模型预测结果和标签数据作为输入
    def forward(self, predicts, labels):
        # 将标签数据解包为 images, tcl_maps, tcl_label_maps, border_maps, direction_maps, training_masks, label_list, pos_list, pos_mask
        images, tcl_maps, tcl_label_maps, border_maps \
            , direction_maps, training_masks, label_list, pos_list, pos_mask = labels
        # 对所有批次进行循环
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list, pos_list, pos_mask, self.max_text_length,
            self.max_text_nums, self.pad_num, self.tcl_bs)
        
        # 从预测结果中获取 f_score, f_border, f_direction, f_char
        f_score, f_border, f_direction, f_char = predicts['f_score'], predicts['f_border'], predicts['f_direction'], \
                                                 predicts['f_char']
        
        # 计算得分损失
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        # 计算边界损失
        border_loss = self.border_loss(f_border, border_maps, tcl_maps,
                                       training_masks)
        # 计算方向损失
        direction_loss = self.direction_loss(f_direction, direction_maps,
                                             tcl_maps, training_masks)
        # 计算 CTC 损失
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        # 计算总损失
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss

        # 将各个损失值保存在字典中
        losses = {
            'loss': loss_all,
            "score_loss": score_loss,
            "border_loss": border_loss,
            "direction_loss": direction_loss,
            "ctc_loss": ctc_loss
        }
        # 返回损失字典
        return losses
```