# `.\PaddleOCR\ppocr\postprocess\ct_postprocess.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发在“按原样”基础上，
# 没有任何明示或暗示的保证或条件
# 有关特定语言的权限和限制，请参阅许可证
"""
# 引用来源
# https://github.com/shengtao96/CentripetalText/blob/main/test.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import cv2
import paddle
import pyclipper

class CTPostProcess(object):
    """
    The post process for Centripetal Text (CT).
    """

    def __init__(self, min_score=0.88, min_area=16, box_type='poly', **kwargs):
        # 初始化函数，设置最小得分、最小区域和框类型等参数
        self.min_score = min_score
        self.min_area = min_area
        self.box_type = box_type

        # 创建一个 2x300x300 的整数数组，用于存储坐标信息
        self.coord = np.zeros((2, 300, 300), dtype=np.int32)
        for i in range(300):
            for j in range(300):
                # 将坐标信息填充到数组中
                self.coord[0, i, j] = j
                self.coord[1, i, j] = i
```