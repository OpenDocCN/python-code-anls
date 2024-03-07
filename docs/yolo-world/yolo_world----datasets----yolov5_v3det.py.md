# `.\YOLO-World\yolo_world\datasets\yolov5_v3det.py`

```
# 导入所需的模块和函数
import copy
import json
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path

from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import CocoDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

# 定义需要忽略的文件列表
v3det_ignore_list = [
    'a00013820/26_275_28143226914_ff3a247c53_c.jpg',
    'n03815615/12_1489_32968099046_be38fa580e_c.jpg',
    'n04550184/19_1480_2504784164_ffa3db8844_c.jpg',
    'a00008703/2_363_3576131784_dfac6fc6ce_c.jpg',
    'n02814533/28_2216_30224383848_a90697f1b3_c.jpg',
    'n12026476/29_186_15091304754_5c219872f7_c.jpg',
    'n01956764/12_2004_50133201066_72e0d9fea5_c.jpg',
    'n03785016/14_2642_518053131_d07abcb5da_c.jpg',
    'a00011156/33_250_4548479728_9ce5246596_c.jpg',
    'a00009461/19_152_2792869324_db95bebc84_c.jpg',
]

# 注册 V3DetDataset 类
@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Objects365 v1 dataset for detection."""

    METAINFO = {'classes': 'classes', 'palette': None}

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

# 注册 YOLOv5V3DetDataset 类，继承自 BatchShapePolicyDataset 和 V3DetDataset
@DATASETS.register_module()
class YOLOv5V3DetDataset(BatchShapePolicyDataset, V3DetDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
```