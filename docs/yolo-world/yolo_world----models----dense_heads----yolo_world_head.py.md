# `.\YOLO-World\yolo_world\models\dense_heads\yolo_world_head.py`

```py
# 导入所需的库和模块
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (
    multi_apply,
    unpack_gt_instances,
    filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer

# 注册模型类为MODELS
@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """
    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        # 初始化偏置参数
        self.bias = nn.Parameter(torch.zeros([]))
        # 初始化logit_scale参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        # 对输入x进行L2范数归一化
        x = F.normalize(x, dim=1, p=2)
        # 对输入w进行L2范数归一化
        w = F.normalize(w, dim=-1, p=2)
        # 使用torch.einsum计算张量乘积
        x = torch.einsum('bchw,bkc->bkhw', x, w)
        # 对结果乘以logit_scale的指数并加上偏置
        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    # 定义一个名为ContrastiveHead的类，继承自nn.Module类
    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:
        # 调用父类的初始化方法
        super().__init__(init_cfg=init_cfg)
        # 根据norm_cfg中的参数构建规范化层
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        # 初始化偏置参数为0
        self.bias = nn.Parameter(torch.zeros([]))
        # 初始化logit_scale参数为-1.0，用于稳定性
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    # 定义前向传播函数
    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        # 对输入x进行规范化
        x = self.norm(x)
        # 对输入w进行L2范数规范化
        w = F.normalize(w, dim=-1, p=2)
        # 使用torch.einsum进行张量乘法操作
        x = torch.einsum('bchw,bkc->bkhw', x, w)
        # 对结果乘以logit_scale的指数，并加上偏置
        x = x * self.logit_scale.exp() + self.bias
        # 返回结果
        return x
# 注册 YOLO-World 的头部模块到模型注册表中
@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 **kwargs) -> None:
        # 初始化头部模块的属性
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # 调用父类的初始化权重方法
        super().init_weights()
        # 针对每个类别预测器和类别对比器进行初始化
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # 重置偏置
            # 如果类别对比器有偏置属性
            if hasattr(cls_contrast, 'bias'):
                # 使用常数初始化类别对比器的偏置
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        # 确保图像特征的数量等于级别数量
        assert len(img_feats) == self.num_levels
        # 将文本特征复制到每个级别的文本特征列表中
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        # 调用 multi_apply 方法进行前向传播
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts)
    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        # 获取输入特征的形状信息
        b, _, h, w = img_feat.shape
        # 使用分类预测模型对图像特征进行预测
        cls_embed = cls_pred(img_feat)
        # 使用对比损失模型对分类嵌入进行预测
        cls_logit = cls_contrast(cls_embed, txt_feat)
        # 使用回归预测模型对图像特征进行预测
        bbox_dist_preds = reg_pred(img_feat)
        # 如果回归最大值大于1
        if self.reg_max > 1:
            # 重新调整bbox_dist_preds的形状
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: get_flops脚本无法处理矩阵乘法的情况，稍后需要修复
            # 计算bbox_preds，softmax后与proj矩阵相乘
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            # 调整bbox_preds的形状
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        # 如果是训练模式，返回分类预测、bbox预测和bbox距离预测
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        # 如果是推理模式，返回分类预测和bbox预测
        else:
            return cls_logit, bbox_preds
@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """注册YOLO-World头部模块，并继承自YOLOv8Head"""

    """YOLO-World头部"""
    def __init__(self, world_size=-1, *args, **kwargs) -> None:
        """初始化函数，设置world_size参数"""
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    """YOLO World v8头部。"""
    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """对上游网络的特征执行前向传播和损失计算"""

        outs = self(img_feats, txt_feats)
        # 快速版本
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """执行头部的前向传播，然后从特征和数据样本中计算损失和预测。"""
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """从上游网络前向传递特征。"""
        return self.head_module(img_feats, txt_feats)
    # 对象方法，用于对输入的图像特征、文本特征和批量数据样本进行前向传播，预测检测结果
    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        # 从批量数据样本中提取图像元信息
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # 对输入的图像特征和文本特征进行前向传播
        outs = self(img_feats, txt_feats)
        # 根据前向传播的结果和图像元信息进行预测，返回预测结果
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        # 返回预测结果
        return predictions

    # 对象方法，用于进行带有测试时间数据增强的测试
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        # 抛出未实现的错误，提示该方法尚未实现
        raise NotImplementedError('aug_test is not implemented yet.')
```