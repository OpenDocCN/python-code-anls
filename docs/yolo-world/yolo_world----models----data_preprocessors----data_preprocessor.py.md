# `.\YOLO-World\yolo_world\models\data_preprocessors\data_preprocessor.py`

```
# 版权声明
# 导入必要的库和模块
from typing import Optional, Union
import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import BaseDataElement
from mmyolo.registry import MODELS

# 定义数据类型
CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]

# 注册YOLOWDetDataPreprocessor类到MODELS模块
@MODELS.register_module()
class YOLOWDetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolow_collate`
    """

    # 初始化函数，接受参数和关键字参数
    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        # 调用父类的初始化函数，并传入参数和关键字参数
        super().__init__(*args, non_blocking=non_blocking, **kwargs)
    # 执行基于“DetDataPreprocessor”的归一化、填充和bgr2rgb转换
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        # 如果不是训练状态，则直接调用父类的forward方法
        if not training:
            return super().forward(data, training)

        # 对数据进行类型转换
        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # TODO: 支持多尺度训练
        # 如果启用通道转换且输入数据通道数为3，则进行通道转换
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        # 如果启用归一化，则对输入数据进行归一化处理
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        # 如果存在批量增强操作，则逐个应用
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        # 生成图像元信息列表
        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'texts': data_samples['texts'],
            'img_metas': img_metas
        }
        # 如果数据样本中包含'masks'，则添加到输出中
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        # 如果数据样本中包含'is_detection'，则添加到输出中
        if 'is_detection' in data_samples:
            data_samples_output['is_detection'] = data_samples['is_detection']

        # 返回处理后的数据
        return {'inputs': inputs, 'data_samples': data_samples_output}
```