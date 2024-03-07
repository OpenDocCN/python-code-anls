# `.\YOLO-World\yolo_world\datasets\yolov5_obj365v2.py`

```
# 导入 Objects365V2Dataset 类
from mmdet.datasets import Objects365V2Dataset

# 导入 BatchShapePolicyDataset 类
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
# 导入 DATASETS 注册表
from mmyolo.registry import DATASETS

# 注册 YOLOv5Objects365V2Dataset 类到 DATASETS 注册表
@DATASETS.register_module()
class YOLOv5Objects365V2Dataset(BatchShapePolicyDataset, Objects365V2Dataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    # 空的类定义，继承自 BatchShapePolicyDataset 和 Objects365V2Dataset
    pass
```