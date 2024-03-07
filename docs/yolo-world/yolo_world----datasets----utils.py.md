# `.\YOLO-World\yolo_world\datasets\utils.py`

```
# 导入必要的库和模块
from typing import Sequence
import torch
from mmengine.dataset import COLLATE_FUNCTIONS

# 注册自定义的数据集拼接函数
@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    # 初始化空列表用于存储数据
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    
    # 遍历数据批次
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        # 获取 ground truth 边界框和标签
        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        
        # 如果数据中包含 masks，则转换为张量并添加到 batch_masks 列表中
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to_tensor(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        
        # 创建 batch_idx 用于标识数据批次，拼接边界框和标签
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    # 构建拼接后的结果字典
    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    
    # 如果存在 masks 数据，则添加到结果字典中
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    # 根据是否使用多尺度训练，将输入数据添加到结果字典中
    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)

    # 如果数据中包含文本信息，则添加到结果字典中
    if hasattr(data_batch[0]['data_samples'], 'texts'):
        batch_texts = [meta['data_samples'].texts for meta in data_batch]
        collated_results['data_samples']['texts'] = batch_texts
    # 检查第一个数据批次中的'data_samples'是否具有'is_detection'属性
    if hasattr(data_batch[0]['data_samples'], 'is_detection'):
        # 如果具有'data_samples'中的'is_detection'属性，则提取每个数据批次中'data_samples'的'is_detection'值
        batch_detection = [meta['data_samples'].is_detection
                           for meta in data_batch]
        # 将提取的'data_samples'中的'is_detection'值转换为torch张量，并存储在collated_results字典中
        collated_results['data_samples']['is_detection'] = torch.tensor(
            batch_detection)

    # 返回整理后的结果字典
    return collated_results
```