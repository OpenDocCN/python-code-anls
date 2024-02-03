# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\__init__.py`

```py
# 设置文件编码为 UTF-8
# 设置时间为 2019年12月4日 10点53分
# 设置作者为 zhoujun
# 导入当前目录下的 iaa_augment 模块
from .iaa_augment import IaaAugment
# 导入当前目录下的 augment 模块
from .augment import *
# 导入当前目录下的 random_crop_data 模块中的 EastRandomCropData 和 PSERandomCrop 类
from .random_crop_data import EastRandomCropData, PSERandomCrop
# 导入当前目录下的 make_border_map 模块中的 MakeBorderMap 类
from .make_border_map import MakeBorderMap
# 导入当前目录下的 make_shrink_map 模块中的 MakeShrinkMap 类
from .make_shrink_map import MakeShrinkMap
```