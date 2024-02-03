# `.\PaddleOCR\ppocr\losses\distillation_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
#
# 导入所需的库
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2

# 导入自定义的损失函数
from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss
from .rec_ce_loss import CELoss
from .basic_loss import DMLLoss, KLDivLoss, DKDLoss
from .basic_loss import DistanceLoss
from .basic_loss import LossFromOutput
from .det_db_loss import DBLoss
from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss
from .vqa_token_layoutlm_loss import VQASerTokenLayoutLMLoss

# 定义一个函数用于计算损失字典中所有损失的总和
def _sum_loss(loss_dict):
    # 如果损失字典中已经包含了总损失，则直接返回
    if "loss" in loss_dict.keys():
        return loss_dict
    else:
        # 否则初始化总损失为0
        loss_dict["loss"] = 0.
        # 遍历损失字典，计算总损失
        for k, value in loss_dict.items():
            if k == "loss":
                continue
            else:
                loss_dict["loss"] += value
        return loss_dict

# 定义一个继承自 DMLLoss 的类，用于蒸馏损失
class DistillationDMLLoss(DMLLoss):
    """
    """
    # 初始化函数，设置模型名称对列表、激活函数、是否使用对数、关键字、是否多头、区分头、映射名称、名称
    def __init__(self,
                 model_name_pairs=[],
                 act=None,
                 use_log=False,
                 key=None,
                 multi_head=False,
                 dis_head='ctc',
                 maps_name=None,
                 name="dml"):
        # 调用父类的初始化函数，设置激活函数和是否使用对数
        super().__init__(act=act, use_log=use_log)
        # 断言模型名称对列表是列表类型
        assert isinstance(model_name_pairs, list)
        # 设置关键字、是否多头、区分头、模型名称对列表、名称、映射名称
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    # 检查模型名称对列表的函数
    def _check_model_name_pairs(self, model_name_pairs):
        # 如果模型名称对列表不是列表类型，则返回空列表
        if not isinstance(model_name_pairs, list):
            return []
        # 如果模型名称对列表的第一个元素是列表类型且第一个元素的第一个元素是字符串类型，则返回模型名称对列表
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    # 检查映射名称的函数
    def _check_maps_name(self, maps_name):
        # 如果映射名称为None，则返回None
        if maps_name is None:
            return None
        # 如果映射名称是字符串类型，则返回包含映射名称的列表
        elif type(maps_name) == str:
            return [maps_name]
        # 如果映射名称是列表类型，则返回包含映射名称的列表
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    # 切片输出的函数
    def _slice_out(self, outs):
        # 创建新的输出字典
        new_outs = {}
        # 遍历映射名称列表
        for k in self.maps_name:
            # 如果映射名称为"thrink_maps"，则将输出的第一维切片并存入新的输出字典
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            # 如果映射名称为"threshold_maps"，则将输出的第二维切片并存入新的输出字典
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            # 如果映射名称为"binary_maps"，则将输出的第三维切片并存入新的输出字典
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        # 返回新的输出字典
        return new_outs
    # 定义一个前向传播函数，接受模型预测结果和批次数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对列表
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取模型预测结果中的两个输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果指定了关键字，则从输出中提取对应的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            # 如果未指定映射名称
            if self.maps_name is None:
                # 如果是多头模型
                if self.multi_head:
                    # 调用父类的前向传播方法计算损失
                    loss = super().forward(out1[self.dis_head], out2[self.dis_head])
                else:
                    # 调用父类的前向传播方法计算损失
                    loss = super().forward(out1, out2)
                # 如果损失是字典类型
                if isinstance(loss, dict):
                    # 将损失值添加到损失字典中
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = loss[key]
                else:
                    # 将损失值添加到损失字典中
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                # 对输出进行切片
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                # 遍历切片后的输出
                for _c, k in enumerate(outs1.keys()):
                    # 调用父类的前向传播方法计算损失
                    loss = super().forward(outs1[k], outs2[k])
                    # 如果损失是字典类型
                    if isinstance(loss, dict):
                        # 将损失值添加到损失字典中
                        for key in loss:
                            loss_dict["{}_{}_{}_{}_{}".format(key, pair[0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        # 将损失值添加到损失字典中
                        loss_dict["{}_{}_{}".format(self.name, self.maps_name[_c], idx)] = loss

        # 对损失字典中的损失值进行求和
        loss_dict = _sum_loss(loss_dict)

        # 返回损失字典
        return loss_dict
class DistillationKLDivLoss(KLDivLoss):
    """
    继承自 KLDivLoss 类的蒸馏 KL 散度损失类
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 multi_head=False,
                 dis_head='ctc',
                 maps_name=None,
                 name="kl_div"):
        # 初始化函数，设置参数
        super().__init__()
        # 断言 model_name_pairs 是列表类型
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        # 检查并设置模型名称对
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        # 检查并设置映射名称
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        # 检查模型名称对是否符合要求
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        # 检查映射名称是否符合要求
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        # 从输出中切片出指定的映射
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs
    # 定义一个前向传播函数，接受模型预测结果和批次数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对列表中的每个索引和元素对
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取模型预测结果中对应模型名称对的输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果指定了关键字，则从输出中获取对应键的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            # 如果未指定映射名称
            if self.maps_name is None:
                # 如果是多头模型
                if self.multi_head:
                    # 获取当前批次中的最大长度
                    max_len = batch[3].max()
                    # 获取目标值
                    tgt = batch[2][:, 1:2 + max_len]
                    tgt = tgt.reshape([-1])
                    # 生成非填充掩码
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(
                            tgt.shape, dtype=tgt.dtype))
                    # 计算损失
                    loss = super().forward(out1[self.dis_head],
                                           out2[self.dis_head], non_pad_mask)
                else:
                    # 计算损失
                    loss = super().forward(out1, out2)
                # 如果损失是字典类型
                if isinstance(loss, dict):
                    # 将损失值添加到损失字典中
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1],
                                                       idx)] = loss[key]
                else:
                    # 将损失值添加到损失字典中
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                # 对输出进行切片
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                # 遍历切片后的输出
                for _c, k in enumerate(outs1.keys()):
                    # 计算损失
                    loss = super().forward(outs1[k], outs2[k])
                    # 如果损失是字典类型
                    if isinstance(loss, dict):
                        # 将损失值添加到损失字典中
                        for key in loss:
                            loss_dict["{}_{}_{}_{}_{}".format(key, pair[
                                0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        # 将损失值添加到损失字典中
                        loss_dict["{}_{}_{}".format(self.name, self.maps_name[
                            _c], idx)] = loss

        # 对损失字典中的损失值进行求和
        loss_dict = _sum_loss(loss_dict)

        # 返回损失字典
        return loss_dict
class DistillationDKDLoss(DKDLoss):
    """
    继承自 DKDLoss 类的蒸馏损失函数类 DistillationDKDLoss
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 multi_head=False,
                 dis_head='ctc',
                 maps_name=None,
                 name="dkd",
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0):
        # 调用父类 DKDLoss 的构造函数，传入温度、alpha 和 beta 参数
        super().__init__(temperature, alpha, beta)
        # 断言 model_name_pairs 是列表类型
        assert isinstance(model_name_pairs, list)
        # 初始化实例变量
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        # 调用 _check_model_name_pairs 方法，检查并设置 model_name_pairs
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        # 调用 _check_maps_name 方法，检查并设置 maps_name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        # 如果 model_name_pairs 不是列表类型，则返回空列表
        if not isinstance(model_name_pairs, list):
            return []
        # 如果 model_name_pairs 的第一个元素是列表类型且第一个元素的第一个元素是字符串类型，则返回 model_name_pairs
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        # 如果 maps_name 为 None，则返回 None
        if maps_name is None:
            return None
        # 如果 maps_name 是字符串类型，则返回包含 maps_name 的列表
        elif type(maps_name) == str:
            return [maps_name]
        # 如果 maps_name 是列表类型，则返回 maps_name
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        # 初始化新的输出字典
        new_outs = {}
        # 遍历 maps_name
        for k in self.maps_name:
            # 根据不同的键值进行切片操作
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

class DistillationNRTRDMLLoss(DistillationDMLLoss):
    """
    继承自 DistillationDMLLoss 类的蒸馏损失函数类 DistillationNRTRDMLLoss
    """
    # 定义一个方法用于计算模型的前向传播，接收模型的预测结果和批次数据作为参数
    def forward(self, predicts, batch):
        # 初始化一个空字典用于存储损失值
        loss_dict = dict()
        # 遍历模型名称对列表中的索引和值
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取模型预测结果中对应模型名称的输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果指定了关键字，则从输出中获取对应关键字的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]

            # 如果是多头模型
            if self.multi_head:
                # 获取当前批次中的最大长度
                max_len = batch[3].max()
                # 获取目标值
                tgt = batch[2][:, 1:2 + max_len]
                tgt = tgt.reshape([-1])
                # 创建一个非填充掩码
                non_pad_mask = paddle.not_equal(
                    tgt, paddle.zeros(
                        tgt.shape, dtype=tgt.dtype))
                # 计算损失值
                loss = super().forward(out1[self.dis_head], out2[self.dis_head],
                                       non_pad_mask)
            else:
                # 计算损失值
                loss = super().forward(out1, out2)
            # 如果损失值是字典类型
            if isinstance(loss, dict):
                # 遍历损失值字典，将损失值存储到loss_dict中
                for key in loss:
                    loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1],
                                                   idx)] = loss[key]
            else:
                # 将损失值存储到loss_dict中
                loss_dict["{}_{}".format(self.name, idx)] = loss

        # 对损失值进行求和
        loss_dict = _sum_loss(loss_dict)

        # 返回损失值字典
        return loss_dict
class DistillationKLDivLoss(KLDivLoss):
    """
    继承自 KLDivLoss 类的蒸馏 KL 散度损失类
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 multi_head=False,
                 dis_head='ctc',
                 maps_name=None,
                 name="kl_div"):
        # 初始化函数，设置参数
        super().__init__()
        # 断言 model_name_pairs 是列表类型
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        # 检查并设置模型名称对
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        # 检查并设置映射名称
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        # 检查模型名称对是否符合要求
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        # 检查映射名称是否符合要求
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        # 从输出中切片出指定的映射
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs
    # 定义一个前向传播函数，接受模型预测结果和批次数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对列表中的每个索引和元素对
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取模型预测结果中对应模型名称对的输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果指定了关键字，则从输出中获取对应键的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            # 如果未指定映射名称
            if self.maps_name is None:
                # 如果是多头模型
                if self.multi_head:
                    # 获取当前批次中的最大长度
                    max_len = batch[3].max()
                    # 获取目标值
                    tgt = batch[2][:, 1:2 + max_len]
                    tgt = tgt.reshape([-1])
                    # 生成非填充掩码
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(
                            tgt.shape, dtype=tgt.dtype))
                    # 计算损失
                    loss = super().forward(out1[self.dis_head],
                                           out2[self.dis_head], non_pad_mask)
                else:
                    # 计算损失
                    loss = super().forward(out1, out2)
                # 如果损失是字典类型
                if isinstance(loss, dict):
                    # 将损失值添加到损失字典中
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1],
                                                       idx)] = loss[key]
                else:
                    # 将损失值添加到损失字典中
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                # 对输出进行切片
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                # 遍历切片后的输出
                for _c, k in enumerate(outs1.keys()):
                    # 计算损失
                    loss = super().forward(outs1[k], outs2[k])
                    # 如果损失是字典类型
                    if isinstance(loss, dict):
                        # 将损失值添加到损失字典中
                        for key in loss:
                            loss_dict["{}_{}_{}_{}_{}".format(key, pair[
                                0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        # 将损失值添加到损失字典中
                        loss_dict["{}_{}_{}".format(self.name, self.maps_name[
                            _c], idx)] = loss

        # 对损失字典中的损失值进行求和
        loss_dict = _sum_loss(loss_dict)

        # 返回损失字典
        return loss_dict
# 定义一个继承自DKDLoss的DistillationDKDLoss类
class DistillationDKDLoss(DKDLoss):
    """
    """

    # 初始化方法，接受多个参数
    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 multi_head=False,
                 dis_head='ctc',
                 maps_name=None,
                 name="dkd",
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0):
        # 调用父类的初始化方法
        super().__init__(temperature, alpha, beta)
        # 断言model_name_pairs是一个列表
        assert isinstance(model_name_pairs, list)
        # 初始化实例变量
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        # 调用_check_model_name_pairs方法，检查model_name_pairs
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        # 调用_check_maps_name方法，检查maps_name
        self.maps_name = self._check_maps_name(maps_name)

    # 检查model_name_pairs的方法
    def _check_model_name_pairs(self, model_name_pairs):
        # 如果model_name_pairs不是列表，则返回空列表
        if not isinstance(model_name_pairs, list):
            return []
        # 如果model_name_pairs的第一个元素是列表且第一个元素的第一个元素是字符串，则返回model_name_pairs
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    # 检查maps_name的方法
    def _check_maps_name(self, maps_name):
        # 如果maps_name为None，则返回None
        if maps_name is None:
            return None
        # 如果maps_name是字符串，则返回包含maps_name的列表
        elif type(maps_name) == str:
            return [maps_name]
        # 如果maps_name是列表，则返回maps_name
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    # 切片方法，根据maps_name对outs进行切片
    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

# 定义一个继承自CTCLoss的DistillationCTCLoss类
class DistillationCTCLoss(CTCLoss):
    # 初始化函数，设置模型名称列表、关键字、是否多头、损失函数名称等属性
    def __init__(self,
                 model_name_list=[],
                 key=None,
                 multi_head=False,
                 name="loss_ctc"):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型名称列表
        self.model_name_list = model_name_list
        # 设置关键字
        self.key = key
        # 设置损失函数名称
        self.name = name
        # 设置是否多头
        self.multi_head = multi_head

    # 前向传播函数，计算损失
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称列表
        for idx, model_name in enumerate(self.model_name_list):
            # 获取模型预测结果
            out = predicts[model_name]
            # 如果有关键字，则从预测结果中获取对应部分
            if self.key is not None:
                out = out[self.key]
            # 如果是多头模型
            if self.multi_head:
                # 确保预测结果中包含 'ctc'，否则抛出异常
                assert 'ctc' in out, 'multi head has multi out'
                # 计算损失
                loss = super().forward(out['ctc'], batch[:2] + batch[3:])
            else:
                # 计算损失
                loss = super().forward(out, batch)
            # 如果损失是字典类型
            if isinstance(loss, dict):
                # 遍历损失字典，将损失值添加到总损失字典中
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name,
                                                idx)] = loss[key]
            else:
                # 将损失值添加到总损失字典中
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        # 返回总损失字典
        return loss_dict
# 定义一个继承自SARLoss的DistillationSARLoss类
class DistillationSARLoss(SARLoss):
    # 初始化函数，接受一些参数
    def __init__(self,
                 model_name_list=[],
                 key=None,
                 multi_head=False,
                 name="loss_sar",
                 **kwargs):
        # 从kwargs中获取ignore_index参数，默认值为92
        ignore_index = kwargs.get('ignore_index', 92)
        # 调用父类的初始化函数，传入ignore_index参数
        super().__init__(ignore_index=ignore_index)
        # 初始化model_name_list、key、name和multi_head属性
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    # 前向传播函数，接受predicts和batch作为输入
    def forward(self, predicts, batch):
        # 初始化一个空字典用于存储损失值
        loss_dict = dict()
        # 遍历model_name_list中的模型名称和索引
        for idx, model_name in enumerate(self.model_name_list):
            # 获取模型预测结果
            out = predicts[model_name]
            # 如果key不为None，则从out中获取对应key的值
            if self.key is not None:
                out = out[self.key]
            # 如果multi_head为True，则执行以下操作
            if self.multi_head:
                # 断言'sar'在out中，如果不在则抛出异常
                assert 'sar' in out, 'multi head has multi out'
                # 调用父类的forward函数，传入'sar'对应的值和batch的部分数据
                loss = super().forward(out['sar'], batch[:1] + batch[2:])
            else:
                # 调用父类的forward函数，传入out和batch
                loss = super().forward(out, batch)
            # 如果loss是字典类型，则遍历其中的键值对
            if isinstance(loss, dict):
                for key in loss:
                    # 将损失值存入loss_dict中，键名包含name、model_name和索引
                    loss_dict["{}_{}_{}".format(self.name, model_name,
                                                idx)] = loss[key]
            else:
                # 将损失值存入loss_dict中，键名包含name和model_name
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        # 返回损失值字典
        return loss_dict

# 定义一个继承自CELoss的DistillationNRTRLoss类
class DistillationNRTRLoss(CELoss):
    # 初始化函数，接受一些参数
    def __init__(self,
                 model_name_list=[],
                 key=None,
                 multi_head=False,
                 smoothing=True,
                 name="loss_nrtr",
                 **kwargs):
        # 调用父类的初始化函数，传入smoothing参数
        super().__init__(smoothing=smoothing)
        # 初始化model_name_list、key、name和multi_head属性
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head
    # 定义一个前向传播函数，接受模型预测结果和批处理数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称列表中的每个模型
        for idx, model_name in enumerate(self.model_name_list):
            # 获取当前模型的预测结果
            out = predicts[model_name]
            # 如果指定了关键字，则从预测结果中获取对应的值
            if self.key is not None:
                out = out[self.key]
            # 如果是多头模型
            if self.multi_head:
                # 确保预测结果中包含 'gtc' 键，用于多头模型的多输出
                assert 'gtc' in out, 'multi head has multi out'
                # 调用父类的前向传播函数计算损失
                loss = super().forward(out['gtc'], batch[:1] + batch[2:])
            else:
                # 调用父类的前向传播函数计算损失
                loss = super().forward(out, batch)
            # 如果损失是字典类型
            if isinstance(loss, dict):
                # 遍历损失字典，将损失值添加到总损失字典中
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name,
                                                idx)] = loss[key]
            else:
                # 将损失值添加到总损失字典中
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        # 返回总损失字典
        return loss_dict
class DistillationDBLoss(DBLoss):
    # 定义一个继承自DBLoss类的DistillationDBLoss类
    def __init__(self,
                 model_name_list=[],
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 name="db",
                 **kwargs):
        # 初始化方法，设置类的属性
        super().__init__()
        # 调用父类的初始化方法
        self.model_name_list = model_name_list
        # 设置模型名称列表属性
        self.name = name
        # 设置名称属性
        self.key = None
        # 设置关键字属性为None

    def forward(self, predicts, batch):
        # 前向传播方法，计算损失
        loss_dict = {}
        # 初始化损失字典
        for idx, model_name in enumerate(self.model_name_list):
            # 遍历模型名称列表
            out = predicts[model_name]
            # 获取预测结果
            if self.key is not None:
                out = out[self.key]
            # 如果关键字不为None，则从预测结果中获取对应的值
            loss = super().forward(out, batch)
            # 调用父类的前向传播方法计算损失

            if isinstance(loss, dict):
                # 如果损失是字典类型
                for key in loss.keys():
                    # 遍历损失字典的键
                    if key == "loss":
                        continue
                    # 如果键为"loss"则跳过
                    name = "{}_{}_{}".format(self.name, model_name, key)
                    # 根据名称格式化生成新的名称
                    loss_dict[name] = loss[key]
                    # 将损失值添加到损失字典中
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
                # 将损失值添加到损失字典中

        loss_dict = _sum_loss(loss_dict)
        # 调用_sum_loss函数对损失字典进行求和
        return loss_dict
        # 返回损失字典


class DistillationDilaDBLoss(DBLoss):
    # 定义一个继承自DBLoss类的DistillationDilaDBLoss类
    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 name="dila_dbloss"):
        # 初始化方法，设置类的属性
        super().__init__()
        # 调用父类的初始化方法
        self.model_name_pairs = model_name_pairs
        # 设置模型名称对属性
        self.name = name
        # 设置名称属性
        self.key = key
        # 设置关键字属性
    # 定义一个前向传播函数，接受模型预测结果和批次数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对列表
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取学生和教师模型的预测结果
            stu_outs = predicts[pair[0]]
            tch_outs = predicts[pair[1]]
            # 如果指定了关键字，则获取对应的预测结果
            if self.key is not None:
                stu_preds = stu_outs[self.key]
                tch_preds = tch_outs[self.key]

            # 获取学生模型的收缩图和二值图
            stu_shrink_maps = stu_preds[:, 0, :, :]
            stu_binary_maps = stu_preds[:, 2, :, :]

            # 对教师模型的收缩图进行膨胀操作
            dilation_w = np.array([[1, 1], [1, 1]])
            th_shrink_maps = tch_preds[:, 0, :, :]
            th_shrink_maps = th_shrink_maps.numpy() > 0.3  # 设置阈值为0.3
            dilate_maps = np.zeros_like(th_shrink_maps).astype(np.float32)
            for i in range(th_shrink_maps.shape[0]):
                dilate_maps[i] = cv2.dilate(
                    th_shrink_maps[i, :, :].astype(np.uint8), dilation_w)
            th_shrink_maps = paddle.to_tensor(dilate_maps)

            # 获取批次数据中的标签阈值图、标签阈值掩码、标签收缩图和标签收缩掩码
            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = batch[1:]

            # 计算收缩图损失
            bce_loss = self.alpha * self.bce_loss(
                stu_shrink_maps, th_shrink_maps, label_shrink_mask)
            loss_binary_maps = self.dice_loss(stu_binary_maps, th_shrink_maps,
                                              label_shrink_mask)

            # 构建损失字典的键值
            k = "{}_{}_{}".format(self.name, pair[0], pair[1])
            # 将损失值添加到损失字典中
            loss_dict[k] = bce_loss + loss_binary_maps

        # 对损失字典中的损失值进行求和
        loss_dict = _sum_loss(loss_dict)
        # 返回损失字典
        return loss_dict
# 定义一个继承自DistanceLoss的DistillationDistanceLoss类
class DistillationDistanceLoss(DistanceLoss):
    """
    """

    # 初始化方法，接受参数mode、model_name_pairs、key、name和**kargs
    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 name="loss_distance",
                 **kargs):
        # 调用父类的初始化方法，传入mode和**kargs参数
        super().__init__(mode=mode, **kargs)
        # 断言model_name_pairs是一个列表
        assert isinstance(model_name_pairs, list)
        # 设置key属性为传入的key参数
        self.key = key
        # 设置model_name_pairs属性为传入的model_name_pairs参数
        self.model_name_pairs = model_name_pairs
        # 设置name属性为传入的name参数加上"_l2"
        self.name = name + "_l2"

    # 前向传播方法，接受参数predicts和batch
    def forward(self, predicts, batch):
        # 初始化一个空字典用于存储损失值
        loss_dict = dict()
        # 遍历model_name_pairs列表中的索引和元素
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取预测结果中pair[0]和pair[1]对应的值
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果key不为None，则将out1和out2分别取key对应的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            # 计算out1和out2之间的距离损失
            loss = super().forward(out1, out2)
            # 如果损失是一个字典，则遍历字典中的键值对
            if isinstance(loss, dict):
                for key in loss:
                    # 将损失值存入loss_dict中，键名格式为"self.name_key_idx"
                    loss_dict["{}_{}_{}".format(self.name, key, idx)] = loss[key]
            else:
                # 将损失值存入loss_dict中，键名格式为"self.name_pair[0]_pair[1]_idx"
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)] = loss
        # 返回损失值字典
        return loss_dict


# 定义一个继承自VQASerTokenLayoutLMLoss的DistillationVQASerTokenLayoutLMLoss类
class DistillationVQASerTokenLayoutLMLoss(VQASerTokenLayoutLMLoss):
    # 初始化方法，接受参数num_classes、model_name_list、key和name
    def __init__(self,
                 num_classes,
                 model_name_list=[],
                 key=None,
                 name="loss_ser"):
        # 调用父类的初始化方法，传入num_classes参数
        super().__init__(num_classes=num_classes)
        # 设置model_name_list属性为传入的model_name_list参数
        self.model_name_list = model_name_list
        # 设置key属性为传入的key参数
        self.key = key
        # 设置name属性为传入的name参数
        self.name = name

    # 前向传播方法，接受参数predicts和batch
    def forward(self, predicts, batch):
        # 初始化一个空字典用于存储损失值
        loss_dict = dict()
        # 遍历model_name_list列表中的索引和元素
        for idx, model_name in enumerate(self.model_name_list):
            # 获取预测结果中model_name对应的值
            out = predicts[model_name]
            # 如果key不为None，则将out取key对应的值
            if self.key is not None:
                out = out[self.key]
            # 计算out和batch之间的损失
            loss = super().forward(out, batch)
            # 将损失值存入loss_dict中，键名格式为"self.name_model_name"
            loss_dict["{}_{}".format(self.name, model_name)] = loss["loss"]
        # 返回损失值字典
        return loss_dict


# 定义一个继承自LossFromOutput的DistillationLossFromOutput类
class DistillationLossFromOutput(LossFromOutput):
    # 初始化 LossRe 类，设置默认参数和属性
    def __init__(self,
                 reduction="none",
                 model_name_list=[],
                 dist_key=None,
                 key="loss",
                 name="loss_re"):
        # 调用父类的初始化方法，设置 key 和 reduction
        super().__init__(key=key, reduction=reduction)
        # 设置模型名称列表
        self.model_name_list = model_name_list
        # 设置名称
        self.name = name
        # 设置分布键
        self.dist_key = dist_key

    # 前向传播方法，计算损失
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称列表
        for idx, model_name in enumerate(self.model_name_list):
            # 获取模型预测结果
            out = predicts[model_name]
            # 如果有分布键，取出对应的值
            if self.dist_key is not None:
                out = out[self.dist_key]
            # 计算损失
            loss = super().forward(out, batch)
            # 将损失值存入损失字典中
            loss_dict["{}_{}".format(self.name, model_name)] = loss["loss"]
        # 返回损失字典
        return loss_dict
# 定义一个继承自 DMLLoss 的类 DistillationSERDMLLoss
class DistillationSERDMLLoss(DMLLoss):
    """
    """

    # 初始化方法，接受多个参数
    def __init__(self,
                 act="softmax",
                 use_log=True,
                 num_classes=7,
                 model_name_pairs=[],
                 key=None,
                 name="loss_dml_ser"):
        # 调用父类的初始化方法
        super().__init__(act=act, use_log=use_log)
        # 断言 model_name_pairs 是列表类型
        assert isinstance(model_name_pairs, list)
        # 初始化类的属性
        self.key = key
        self.name = name
        self.num_classes = num_classes
        self.model_name_pairs = model_name_pairs

    # 前向传播方法，接受 predicts 和 batch 两个参数
    def forward(self, predicts, batch):
        # 初始化一个空字典用于存储损失值
        loss_dict = dict()
        # 遍历 model_name_pairs 中的索引和值
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取 predicts 中对应的两个输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 如果 key 不为 None，则从输出中获取指定 key 的值
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            # 将输出进行 reshape 操作
            out1 = out1.reshape([-1, out1.shape[-1]])
            out2 = out2.reshape([-1, out2.shape[-1])

            # 获取 batch 中的 attention_mask
            attention_mask = batch[2]
            # 如果 attention_mask 不为 None，则根据其值筛选出有效的输出
            if attention_mask is not None:
                active_output = attention_mask.reshape([-1, ]) == 1
                out1 = out1[active_output]
                out2 = out2[active_output]

            # 将计算得到的损失值存入 loss_dict 中
            loss_dict["{}_{}".format(self.name, idx)] = super().forward(out1, out2)

        # 返回损失值字典
        return loss_dict


# 定义一个继承自 DistanceLoss 的类 DistillationVQADistanceLoss
class DistillationVQADistanceLoss(DistanceLoss):
    # 初始化方法，接受多个参数
    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 index=None,
                 name="loss_distance",
                 **kargs):
        # 调用父类的初始化方法
        super().__init__(mode=mode, **kargs)
        # 断言 model_name_pairs 是列表类型
        assert isinstance(model_name_pairs, list)
        # 初始化类的属性
        self.key = key
        self.index = index
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"
    # 定义一个前向传播函数，接受模型预测结果和批处理数据作为输入
    def forward(self, predicts, batch):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对列表中的每个索引和名称对
        for idx, pair in enumerate(self.model_name_pairs):
            # 获取模型预测结果中的两个输出
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # 获取注意力掩码
            attention_mask = batch[2]
            # 如果存在关键字，则从输出中提取相应部分
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
                # 如果存在索引，则从输出中提取相应部分
                if self.index is not None:
                    out1 = out1[:, self.index, :, :]
                    out2 = out2[:, self.index, :, :]
                # 如果存在注意力掩码，则根据最大长度截取输出
                if attention_mask is not None:
                    max_len = attention_mask.shape[-1]
                    out1 = out1[:, :max_len]
                    out2 = out2[:, :max_len]
                # 重塑输出形状
                out1 = out1.reshape([-1, out1.shape[-1]])
                out2 = out2.reshape([-1, out2.shape[-1]])
            # 如果存在注意力掩码，则根据掩码筛选输出
            if attention_mask is not None:
                active_output = attention_mask.reshape([-1, ]) == 1
                out1 = out1[active_output]
                out2 = out2[active_output]
            # 计算损失
            loss = super().forward(out1, out2)
            # 如果损失是字典类型，则将每个键值对添加到损失字典中
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}nohu_{}".format(self.name, key,
                                                    idx)] = loss[key]
            # 否则将损失添加到损失字典中
            else:
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                               idx)] = loss
        # 返回损失字典
        return loss_dict
class CTCDKDLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self, temperature=0.5, alpha=1.0, beta=1.0):
        # 初始化函数，设置温度、alpha、beta等参数
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6
        self.t = temperature
        self.act = nn.Softmax(axis=-1)
        self.use_log = True

    def kl_loss(self, p1, p2):  # predict, label
        # 计算 KL 散度损失
        loss = paddle.multiply(
            p2, paddle.log((p2 + self.eps) / (p1 + self.eps) + self.eps))
        bs = loss.shape[0]
        loss = paddle.sum(loss) / bs
        return loss

    def _cat_mask(self, t, mask1, mask2):
        # 将两个掩码合并
        t1 = (t * mask1).sum(axis=1, keepdim=True)
        t2 = (t * mask2).sum(axis=1, keepdim=True)
        rt = paddle.concat([t1, t2], axis=1)
        return rt

    def multi_label_mask(self, targets):
        # 生成多标签掩码
        targets = targets.astype("int32")
        res = F.one_hot(targets, num_classes=11465)
        mask = paddle.clip(paddle.sum(res, axis=1), 0, 1)
        mask[:, 0] = 0  # 忽略 CTC 空白标签
        return mask
    # 定义前向传播函数，接收学生模型的logits、教师模型的logits、目标值和掩码作为输入
    def forward(self, logits_student, logits_teacher, targets, mask=None):

        # 生成多标签掩码
        gt_mask = self.multi_label_mask(targets)
        # 生成其他掩码
        other_mask = paddle.ones_like(gt_mask) - gt_mask

        # 对学生模型的logits进行softmax计算，除以温度参数
        pred_student = F.softmax(logits_student / self.temperature, axis=-1)
        # 对教师模型的logits进行softmax计算，除以温度参数
        pred_teacher = F.softmax(logits_teacher / self.temperature, axis=-1)

        # 将学生模型的预测结果按照第一维度求平均
        pred_student = paddle.mean(pred_student, axis=1)
        # 将教师模型的预测结果按照第一维度求平均
        pred_teacher = paddle.mean(pred_teacher, axis=1)

        # 将学生模型的预测结果与多标签掩码和其他掩码拼接
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        # 将教师模型的预测结果与多标签掩码和其他掩码拼接
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)

        # 计算TCKD损失，即学生模型预测结果与教师模型预测结果的KL散度
        tckd_loss = self.kl_loss(pred_student, pred_teacher)

        # 将多标签掩码扩展为与logits_teacher相同形状的张量
        gt_mask_ex = paddle.expand_as(gt_mask.unsqueeze(axis=1), logits_teacher)
        # 对教师模型的logits减去一个大的数值再进行softmax计算
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask_ex, axis=-1)
        # 对学生模型的logits减去一个大的数值再进行softmax计算
        pred_student_part2 = F.softmax(
            logits_student / self.temperature - 1000.0 * gt_mask_ex, axis=-1)

        # 将教师模型的预测结果按照第一维度求平均
        pred_teacher_part2 = paddle.mean(pred_teacher_part2, axis=1)
        # 将学生模型的预测结果按照第一维度求平均
        pred_student_part2 = paddle.mean(pred_student_part2, axis=1)

        # 计算NCKD损失，即学生模型第二部分预测结果与教师模型第二部分预测结果的KL散度
        nckd_loss = self.kl_loss(pred_student_part2, pred_teacher_part2)
        
        # 计算总损失，包括TCKD损失和NCKD损失，乘以相应的权重系数
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        # 返回总损失
        return loss
class KLCTCLogits(nn.Layer):
    # 定义 KLCTCLogits 类，继承自 nn.Layer
    def __init__(self, weight=1.0, reduction='mean', mode="mean"):
        # 初始化函数，设置权重、减少方式、模式等参数
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-6
        self.t = 0.5
        self.act = nn.Softmax(axis=-1)
        self.use_log = True
        self.mode = mode
        self.ctc_dkd_loss = CTCDKDLoss()

    def kl_loss(self, p1, p2):  # predict, label
        # 计算 KL 散度损失函数
        loss = paddle.multiply(
            p2, paddle.log((p2 + self.eps) / (p1 + self.eps) + self.eps))
        bs = loss.shape[0]
        loss = paddle.sum(loss) / bs
        return loss

    def forward_meanmax(self, stu_out, tea_out):
        # 计算 forward_meanmax 函数
        stu_out = paddle.mean(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.mean(F.softmax(tea_out / self.t, axis=-1), axis=1)
        loss = self.kl_loss(stu_out, tea_out)
        return loss

    def forward_meanlog(self, stu_out, tea_out):
        # 计算 forward_meanlog 函数
        stu_out = paddle.mean(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.mean(F.softmax(tea_out / self.t, axis=-1), axis=1)
        if self.use_log is True:
            # 如果使用 log，则计算 KL 散度损失函数
            log_out1 = paddle.log(stu_out)
            log_out2 = paddle.log(tea_out)
            loss = (
                self._kldiv(log_out1, tea_out) + self._kldiv(log_out2, stu_out)
            ) / 2.0
        return loss

    def forward_sum(self, stu_out, tea_out):
        # 计算 forward_sum 函数
        stu_out = paddle.sum(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.sum(F.softmax(tea_out / self.t, axis=-1), axis=1)
        stu_out = paddle.log(stu_out)
        bs = stu_out.shape[0]
        loss = tea_out * (paddle.log(tea_out + self.eps) - stu_out)
        loss = paddle.sum(loss, axis=1) / loss.shape[0]
        return loss
    # 计算 KL 散度损失函数
    def _kldiv(self, x, target):
        # 设置一个很小的值，避免出现除零错误
        eps = 1.0e-10
        # 计算 KL 散度损失
        loss = target * (paddle.log(target + eps) - x)
        # 对损失进行求和和平均，然后再除以样本数量
        loss = paddle.sum(paddle.mean(loss, axis=1)) / loss.shape[0]
        return loss

    # 前向传播函数
    def forward(self, stu_out, tea_out, targets=None):
        # 根据模式选择不同的前向传播方式
        if self.mode == "log":
            return self.forward_log(stu_out, tea_out)
        elif self.mode == "mean":
            # 创建一个与 stu_out 相同形状的全为1的张量
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            # 将第三维度的第一个元素设置为-1
            blank_mask[:, :, 0] = -1
            # 将 stu_out 和 tea_out 乘以 blank_mask
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanmax(stu_out, tea_out)
        elif self.mode == "sum":
            return self.forward_sum(stu_out, tea_out)
        elif self.mode == "meanlog":
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanlog(stu_out, tea_out)
        elif self.mode == "ctcdkd":
            # 忽略 CTC 空白标签的 logits
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.ctc_dkd_loss(stu_out, tea_out, targets)
        else:
            raise ValueError("error!!!!!!")

    # 对数模式下的前向传播函数
    def forward_log(self, out1, out2):
        # 如果激活函数不为空，则对 out1 和 out2 进行激活并加上一个很小的值
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        # 如果使用对数，则需要对特征图进行对数操作
        if self.use_log is True:
            log_out1 = paddle.log(out1)
            log_out2 = paddle.log(out2)
            # 计算 KL 散度损失的平均值
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0

        return loss
class DistillCTCLogits(KLCTCLogits):
    # 定义一个继承自KLCTCLogits类的DistillCTCLogits类
    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 name="ctc_logits",
                 reduction="mean"):
        # 初始化方法，接受model_name_pairs、key、name、reduction等参数
        super().__init__(reduction=reduction)
        # 调用父类的初始化方法，传入reduction参数
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        # 将model_name_pairs参数传入_check_model_name_pairs方法，并将返回值赋给self.model_name_pairs
        self.key = key
        # 将key参数赋给self.key
        self.name = name
        # 将name参数赋给self.name

    def _check_model_name_pairs(self, model_name_pairs):
        # 定义一个方法_check_model_name_pairs，用于检查model_name_pairs参数
        if not isinstance(model_name_pairs, list):
            return []
            # 如果model_name_pairs不是列表，则返回空列表
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
            # 如果model_name_pairs的第一个元素是列表且第一个元素的第一个元素是字符串，则返回model_name_pairs
        else:
            return [model_name_pairs]
            # 否则将model_name_pairs转换为列表后返回

    def forward(self, predicts, batch):
        # 定义一个前向传播方法forward，接受predicts和batch参数
        loss_dict = dict()
        # 初始化一个空字典loss_dict
        for idx, pair in enumerate(self.model_name_pairs):
            # 遍历self.model_name_pairs中的索引和元素
            out1 = predicts[pair[0]]
            # 获取predicts中pair[0]对应的值，赋给out1
            out2 = predicts[pair[1]]
            # 获取predicts中pair[1]对应的值，赋给out2

            if self.key is not None:
                out1 = out1[self.key]['ctc']
                out2 = out2[self.key]['ctc']
                # 如果self.key不为None，则将out1和out2更新为其对应的'ctc'值

            ctc_label = batch[1]
            # 获取batch中索引为1的元素，赋给ctc_label
            loss = super().forward(out1, out2, ctc_label)
            # 调用父类的forward方法，传入out1、out2和ctc_label参数，将返回值赋给loss
            if isinstance(loss, dict):
                # 如果loss是字典类型
                for key in loss:
                    # 遍历loss中的键
                    loss_dict["{}_{}_{}".format(self.name, model_name,
                                                idx)] = loss[key]
                    # 将loss中的值添加到loss_dict中，键为格式化后的字符串
            else:
                loss_dict["{}_{}".format(self.name, idx)] = loss
                # 如果loss不是字典类型，则将loss添加到loss_dict中，键为格式化后的字符串
        return loss_dict
        # 返回loss_dict
```