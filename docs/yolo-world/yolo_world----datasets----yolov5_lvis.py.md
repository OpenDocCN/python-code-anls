# `.\YOLO-World\yolo_world\datasets\yolov5_lvis.py`

```py
# 导入需要的模块
from mmdet.datasets import LVISV1Dataset

# 导入自定义的数据集类
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

# 注册YOLOv5 LVIS数据集类，继承自BatchShapePolicyDataset和LVISV1Dataset
@DATASETS.register_module()
class YOLOv5LVISV1Dataset(BatchShapePolicyDataset, LVISV1Dataset):
    """Dataset for YOLOv5 LVIS Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    # 空的类定义，没有额外的方法或属性
    pass
```