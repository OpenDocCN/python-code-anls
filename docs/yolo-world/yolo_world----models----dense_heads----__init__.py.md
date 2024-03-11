# `.\YOLO-World\yolo_world\models\dense_heads\__init__.py`

```py
# 导入 YOLOWorldHead 和 YOLOWorldHeadModule 类
from .yolo_world_head import YOLOWorldHead, YOLOWorldHeadModule
# 导入 YOLOWorldSegHead 和 YOLOWorldSegHeadModule 类
from .yolo_world_seg_head import YOLOWorldSegHead, YOLOWorldSegHeadModule

# 定义 __all__ 列表，包含需要导出的类名
__all__ = [
    'YOLOWorldHead', 'YOLOWorldHeadModule', 'YOLOWorldSegHead',
    'YOLOWorldSegHeadModule'
]
```