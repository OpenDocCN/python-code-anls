# `.\YOLO-World\yolo_world\datasets\yolov5_obj365v1.py`

```py
# 导入需要的模块
from mmdet.datasets import Objects365V1Dataset

# 导入自定义的数据集类
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

# 注册YOLOv5Objects365V1Dataset类到DATASETS模块
@DATASETS.register_module()
class YOLOv5Objects365V1Dataset(BatchShapePolicyDataset, Objects365V1Dataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
```