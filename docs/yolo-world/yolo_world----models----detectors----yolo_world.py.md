# `.\YOLO-World\yolo_world\models\detectors\yolo_world.py`

```
# 导入所需的模块和类
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

# 注册YOLOWorldDetector类到MODELS模块
@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    # 初始化函数，接受一些参数
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        # 初始化类的属性
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)

    # 计算损失函数的方法，接受输入和数据样本
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        # 设置bbox_head的类别数为训练类别数
        self.bbox_head.num_classes = self.num_train_classes
        # 提取图像特征和文本特征
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # 计算损失
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        # 返回损失
        return losses
    # 预测模型的方法，接受批量输入和数据样本，返回带有后处理的结果列表
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        # 提取图像特征和文本特征
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # 设置边界框头部的类别数为文本特征的第一个维度大小
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        
        # 使用图像特征、文本特征和数据样本进行预测，返回结果列表
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        # 将预测结果添加到数据样本中
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        # 返回更新后的数据样本
        return batch_data_samples

    # 网络前向传播过程，通常包括骨干网络、颈部和头部的前向传播，不包含任何后处理
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        
        # 提取图像特征和文本特征
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        
        # 进行边界框头部的前向传播，返回结果
        results = self.bbox_head.forward(img_feats, txt_feats)
        
        # 返回结果
        return results
    # 定义一个方法用于提取特征，接受两个输入参数：batch_inputs（张量）和batch_data_samples（样本列表），返回一个元组
    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # 如果batch_data_samples是字典类型，则获取其中的'texts'键对应的值
        if isinstance(batch_data_samples, dict):
            texts = batch_data_samples['texts']
        # 如果batch_data_samples是列表类型，则遍历其中的数据样本，获取每个数据样本的文本信息
        elif isinstance(batch_data_samples, list):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        # 如果batch_data_samples既不是字典类型也不是列表类型，则抛出类型错误异常
        else:
            raise TypeError('batch_data_samples should be dict or list.')

        # 调用backbone模型提取图像和文本特征
        img_feats, txt_feats = self.backbone(batch_inputs, texts)
        # 如果模型包含neck部分
        if self.with_neck:
            # 如果使用多模态neck
            if self.mm_neck:
                # 将图像特征和文本特征输入到neck模块中进行处理
                img_feats = self.neck(img_feats, txt_feats)
            else:
                # 只将图像特征输入到neck模块中进行处理
                img_feats = self.neck(img_feats)
        # 返回提取的图像特征和文本特征
        return img_feats, txt_feats
```