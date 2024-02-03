# `.\segment-anything\segment_anything\__init__.py`

```
# 导入构建 SAM 模型的函数和 SAM 预测器类
from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
# 导入 SAM 预测器类
from .predictor import SamPredictor
# 导入 SAM 自动遮罩生成器类
from .automatic_mask_generator import SamAutomaticMaskGenerator
```