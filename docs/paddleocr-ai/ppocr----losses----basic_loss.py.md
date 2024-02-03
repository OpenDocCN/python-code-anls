# `.\PaddleOCR\ppocr\losses\basic_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import L1Loss
from paddle.nn import MSELoss as L2Loss
from paddle.nn import SmoothL1Loss

# 定义交叉熵损失函数类
class CELoss(nn.Layer):
    def __init__(self, epsilon=None):
        super().__init__()
        # 如果 epsilon 不为 None 且在 (0, 1) 之间，则将其设为 None
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    # 标签平滑函数
    def _labelsmoothing(self, target, class_num):
        # 如果目标形状的最后一个维度不等于类别数，则进行 one-hot 编码
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        # 对 one-hot 编码的目标进行标签平滑
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    # 前向传播函数
    def forward(self, x, label):
        loss_dict = {}
        # 如果 epsilon 不为 None
        if self.epsilon is not None:
            class_num = x.shape[-1]
            # 对标签进行标签平滑
            label = self._labelsmoothing(label, class_num)
            # 对 x 进行 log_softmax 操作
            x = -F.log_softmax(x, axis=-1)
            # 计算交叉熵损失
            loss = paddle.sum(x * label, axis=-1)
        else:
            # 如果标签的最后一个维度与 x 的最后一个维度相等
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            # 计算交叉熵损失
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        return loss

# KLJS 损失函数类
class KLJSLoss(object):
    # 初始化函数，设置损失函数模式为 KL 或 JS
    def __init__(self, mode='kl'):
        # 断言损失函数模式只能是 kl, KL, js, JS 中的一个
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'KL', 'js', 'JS']"
        # 设置损失函数模式
        self.mode = mode

    # 调用函数，计算 KL 或 JS 损失
    def __call__(self, p1, p2, reduction="mean", eps=1e-5):

        # 如果损失函数模式为 kl
        if self.mode.lower() == 'kl':
            # 计算 KL 损失
            loss = paddle.multiply(p2,
                                   paddle.log((p2 + eps) / (p1 + eps) + eps))
            loss += paddle.multiply(p1,
                                    paddle.log((p1 + eps) / (p2 + eps) + eps))
            loss *= 0.5
        # 如果损失函数模式为 js
        elif self.mode.lower() == "js":
            # 计算 JS 损失
            loss = paddle.multiply(
                p2, paddle.log((2 * p2 + eps) / (p1 + p2 + eps) + eps))
            loss += paddle.multiply(
                p1, paddle.log((2 * p1 + eps) / (p1 + p2 + eps) + eps))
            loss *= 0.5
        else:
            # 抛出数值错误，损失函数模式应为 kl 或 js
            raise ValueError(
                "The mode.lower() if KLJSLoss should be one of ['kl', 'js']")

        # 如果指定 reduction 为 mean
        if reduction == "mean":
            # 计算平均损失
            loss = paddle.mean(loss, axis=[1, 2])
        # 如果指定 reduction 为 none 或为空
        elif reduction == "none" or reduction is None:
            # 返回损失
            return loss
        else:
            # 计算总损失
            loss = paddle.sum(loss, axis=[1, 2])

        # 返回损失
        return loss
class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False):
        # 初始化 DMLLoss 类
        super().__init__()
        # 如果指定了激活函数，则确保是 "softmax" 或 "sigmoid"
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        # 根据激活函数类型初始化激活函数对象
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        # 初始化 KLJSLoss 对象
        self.jskl_loss = KLJSLoss(mode="kl")

    def _kldiv(self, x, target):
        # 设置一个极小值
        eps = 1.0e-10
        # 计算 KL 散度损失
        loss = target * (paddle.log(target + eps) - x)
        # 计算批次均值损失
        loss = paddle.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        # 如果存在激活函数，则对输出进行激活并添加一个极小值
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log:
            # 对于识别蒸馏，特征图需要对数，计算 KL 散度损失
            log_out1 = paddle.log(out1)
            log_out2 = paddle.log(out2)
            loss = (
                self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            # 对于检测蒸馏，不需要对数，计算 JSD 损失
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Layer):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", **kargs):
        # 初始化 DistanceLoss 类
        super().__init__()
        # 确保损失模式是 "l1", "l2", 或 "smooth_l1"
        assert mode in ["l1", "l2", "smooth_l1"]
        # 根据损失模式初始化损失函数对象
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        # 计算并返回损失值
        return self.loss_func(x, y)


class LossFromOutput(nn.Layer):
    def __init__(self, key='loss', reduction='none'):
        # 初始化 LossFromOutput 类
        super().__init__()
        self.key = key
        self.reduction = reduction
    # 定义一个前向传播函数，接受预测结果和批次数据作为输入
    def forward(self, predicts, batch):
        # 将预测结果赋值给损失值
        loss = predicts
        # 如果指定了关键字并且预测结果是字典类型，则取出指定关键字对应的值作为损失值
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        # 如果指定了减少方式为均值，则计算损失值的均值
        if self.reduction == 'mean':
            loss = paddle.mean(loss)
        # 如果指定了减少方式为求和，则计算损失值的总和
        elif self.reduction == 'sum':
            loss = paddle.sum(loss)
        # 返回包含损失值的字典
        return {'loss': loss}
class KLDivLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self):
        super().__init__()

    def _kldiv(self, x, target, mask=None):
        # 设置一个很小的常数，避免对数函数中出现零值
        eps = 1.0e-10
        # 计算 KL 散度损失
        loss = target * (paddle.log(target + eps) - x)
        if mask is not None:
            # 将损失展平并按行求和
            loss = loss.flatten(0, 1).sum(axis=1)
            # 根据掩码选择损失值并计算平均值
            loss = loss.masked_select(mask).mean()
        else:
            # 计算批次平均损失
            loss = paddle.sum(loss) / loss.shape[0]
        return loss

    def forward(self, logits_s, logits_t, mask=None):
        # 对 logits_s 进行 log_softmax 操作
        log_out_s = F.log_softmax(logits_s, axis=-1)
        # 对 logits_t 进行 softmax 操作
        out_t = F.softmax(logits_t, axis=-1)
        # 计算 KL 散度损失
        loss = self._kldiv(log_out_s, out_t, mask)
        return loss


class DKDLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self, temperature=1.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def _cat_mask(self, t, mask1, mask2):
        # 计算两个掩码对应位置的加权和
        t1 = (t * mask1).sum(axis=1, keepdim=True)
        t2 = (t * mask2).sum(axis=1, keepdim=True)
        # 拼接两个加权和
        rt = paddle.concat([t1, t2], axis=1)
        return rt

    def _kl_div(self, x, label, mask=None):
        # 计算 KL 散度损失
        y = (label * (paddle.log(label + 1e-10) - x)).sum(axis=1)
        if mask is not None:
            # 根据掩码选择损失值并计算平均值
            y = y.masked_select(mask).mean()
        else:
            # 计算平均损失
            y = y.mean()
        return y
    # 定义前向传播函数，计算蒸馏损失
    def forward(self, logits_student, logits_teacher, target, mask=None):
        # 将目标值转换为独热编码的掩码
        gt_mask = F.one_hot(
            target.reshape([-1]), num_classes=logits_student.shape[-1])
        # 计算非目标值的掩码
        other_mask = 1 - gt_mask
        # 将学生模型的logits展平
        logits_student = logits_student.flatten(0, 1)
        # 将教师模型的logits展平
        logits_teacher = logits_teacher.flatten(0, 1)
        # 对学生模型的logits进行softmax操作
        pred_student = F.softmax(logits_student / self.temperature, axis=1)
        # 对教师模型的logits进行softmax操作
        pred_teacher = F.softmax(logits_teacher / self.temperature, axis=1)
        # 将学生模型的预测结果与目标值掩码和非目标值掩码拼接
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        # 将教师模型的预测结果与目标值掩码和非目标值掩码拼接
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        # 对学生模型的预测结果取对数
        log_pred_student = paddle.log(pred_student)
        # 计算TCKD损失
        tckd_loss = self._kl_div(log_pred_student,
                                 pred_teacher) * (self.temperature**2)
        # 对教师模型的部分预测结果进行softmax操作
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, axis=1)
        # 对学生模型的部分预测结果取对数
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, axis=1)
        # 计算NCKD损失
        nckd_loss = self._kl_div(log_pred_student_part2,
                                 pred_teacher_part2) * (self.temperature**2)

        # 计算最终损失，结合TCKD损失和NCKD损失
        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        # 返回损失值
        return loss
```