# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\rcnn_vl.py`

```
# 版权所有 (c) Facebook, Inc. 及其附属机构
import logging  # 导入 logging 模块以进行日志记录
import numpy as np  # 导入 numpy 作为 np 用于数值计算
from typing import Dict, List, Optional, Tuple  # 从 typing 模块导入类型提示
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

from detectron2.config import configurable  # 从 detectron2 导入可配置装饰器
from detectron2.structures import ImageList, Instances  # 从 detectron2 导入图像列表和实例结构
from detectron2.utils.events import get_event_storage  # 从 detectron2 导入获取事件存储的函数

from detectron2.modeling.backbone import Backbone, build_backbone  # 从 detectron2 导入 Backbone 类及其构建函数
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY  # 从 detectron2 导入元架构注册表

from detectron2.modeling.meta_arch import GeneralizedRCNN  # 从 detectron2 导入广义 R-CNN 类

from detectron2.modeling.postprocessing import detector_postprocess  # 从 detectron2 导入检测器后处理函数
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image  # 从 detectron2 导入单图像快速 R-CNN 推理函数
from contextlib import contextmanager  # 从上下文管理库导入上下文管理器
from itertools import count  # 从 itertools 导入计数器函数

@META_ARCH_REGISTRY.register()  # 在元架构注册表中注册该类
class VLGeneralizedRCNN(GeneralizedRCNN):  # 定义 VLGeneralizedRCNN 类，继承自 GeneralizedRCNN
    """
    广义 R-CNN。任何包含以下三个组件的模型：
    1. 每个图像的特征提取（即骨干网络）
    2. 区域提议生成
    3. 每个区域的特征提取和预测
    """
    # 定义前向传播方法，接受批处理输入
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        参数：
            batched_inputs: 一个列表，包含 :class:`DatasetMapper` 的批处理输出。
                列表中的每个项包含一张图像的输入。
                目前，列表中的每个项是一个字典，包含：
    
                * image: Tensor，图像格式为 (C, H, W)。
                * instances（可选）：真实标签 :class:`Instances`
                * proposals（可选）：预计算的 :class:`Instances`。
    
                原始字典中包含的其他信息，例如：
    
                * "height", "width"（int）：模型的输出分辨率，在推理中使用。
                  详细信息请参见 :meth:`postprocess`。
    
        返回：
            list[dict]:
                每个字典是一个输入图像的输出。
                字典包含一个键 "instances"，其值为 :class:`Instances`。
                :class:`Instances` 对象具有以下键：
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # 如果不是训练模式，则执行推理
        if not self.training:
            return self.inference(batched_inputs)
    
        # 预处理输入图像
        images = self.preprocess_image(batched_inputs)
        # 如果输入中有真实实例，则将其转移到设备上
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
    
        # 从图像中获取特征
        input = self.get_batch(batched_inputs, images)
        features = self.backbone(input)
    
        # 如果存在提案生成器，则生成提案和提案损失
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            # 否则，确保输入中包含提案，并将其转移到设备上
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
    
        # 从 ROI 头部获取检测器损失
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # 如果可视化周期大于 0，则进行可视化
        if self.vis_period > 0:
            storage = get_event_storage()
            # 每隔 vis_period 次迭代进行可视化
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
    
        # 创建损失字典，更新检测器损失和提案损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # 返回损失字典
        return losses
    
    # 定义推理方法，接受批处理输入和其他参数
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        执行给定输入的推断。

        参数：
            batched_inputs (list[dict]): 与 :meth:`forward` 中相同
            detected_instances (None 或 list[Instances]): 如果不是 None，包含每张图像的 `Instances` 对象。`Instances`
                对象包含图像中的“pred_boxes”和“pred_classes”，即已知的框。
                推断将跳过边界框的检测，仅预测其他每个 ROI 的输出。
            do_postprocess (bool): 是否对输出进行后处理。

        返回：
            当 do_postprocess=True 时，与 :meth:`forward` 中相同。
            否则，返回包含原始网络输出的 list[Instances]。
        """
        # 确保当前不处于训练模式
        assert not self.training

        # 对输入图像进行预处理
        images = self.preprocess_image(batched_inputs)
        # 从预处理的图像中提取特征
        # features = self.backbone(images.tensor)
        # 获取批处理输入
        input = self.get_batch(batched_inputs, images)
        # 从主干网络提取特征
        features = self.backbone(input)

        # 如果没有检测到实例
        if detected_instances is None:
            # 如果存在提议生成器
            if self.proposal_generator is not None:
                # 生成提议并丢弃第二个返回值
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                # 确保批处理输入中包含提议
                assert "proposals" in batched_inputs[0]
                # 从批处理输入中提取提议并转移到设备
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            # 通过 ROI 头部处理图像、特征和提议
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            # 将检测到的实例转移到设备
            detected_instances = [x.to(self.device) for x in detected_instances]
            # 使用给定的边界框进行前向推断
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # 如果需要进行后处理
        if do_postprocess:
            # 确保不在脚本模式下
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            # 对结果进行后处理并返回
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            # 返回原始结果
            return results

    # 定义获取批处理输入的方法
    def get_batch(self, examples, images):
        # 如果示例数量大于等于1且第一个示例中没有“bbox”，则为仅图像模式
        if len(examples) >= 1 and "bbox" not in examples[0]:  # image_only
            # 返回图像的张量
            return {"images": images.tensor}

        # 返回输入
        return input
    # 批量推断方法，处理一组输入
        def _batch_inference(self, batched_inputs, detected_instances=None):
            """
            在输入列表上执行推断，
            使用的批量大小为 self.batch_size（例如，2），而不是列表的长度。
    
            输入和输出与 :meth:`GeneralizedRCNN.inference` 的格式相同
            """
            # 如果没有提供检测实例，则初始化为与输入列表长度相同的 None 列表
            if detected_instances is None:
                detected_instances = [None] * len(batched_inputs)
    
            # 初始化输出列表和输入、实例临时存储
            outputs = []
            inputs, instances = [], []
            # 遍历输入和检测实例，使用索引
            for idx, input, instance in zip(count(), batched_inputs, detected_instances):
                # 将当前输入添加到 inputs 列表
                inputs.append(input)
                # 将当前实例添加到 instances 列表
                instances.append(instance)
                # 如果 inputs 长度为 2 或到达输入列表的最后一个元素
                if len(inputs) == 2 or idx == len(batched_inputs) - 1:
                    # 执行推断，扩展到输出列表中
                    outputs.extend(
                        self.inference(
                            inputs,
                            # 如果第一个实例不为 None，则传递实例列表
                            instances if instances[0] is not None else None,
                            do_postprocess=True,  # 设置为 True 进行后处理
                        )
                    )
                    # 清空输入和实例列表，为下批处理做准备
                    inputs, instances = [], []
            # 返回最终的输出列表
            return outputs
```